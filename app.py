from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
import sys
import gc

from langchain_openai import OpenAI
from src.helper import get_embeddings
from pinecone import Pinecone

app = Flask(__name__)
load_dotenv()

os.environ.update({
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:16",
    "TOKENIZERS_PARALLELISM": "false",
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "PINECONE_API_KEY": os.getenv('PINECONE_API_KEY'),
    "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY'),
    "PINECONE_ENVIRONMENT": os.getenv('PINECONE_ENVIRONMENT')
})

llm = None
embeddings = None
pc = None
pinecone_index = None

try:
    llm = OpenAI(
        temperature=0.4,
        max_tokens=200,
        model="gpt-3.5-turbo-instruct"
    )
    embeddings = get_embeddings()
    
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENVIRONMENT'))
    pinecone_index = pc.Index("liberiahealthresponseai")
    
except Exception as e:
    print(f"ERROR: Failed to initialize LLM, Embeddings, or Pinecone client on startup: {e}", file=sys.stderr)

def clear_memory():
    gc.collect()
    if 'torch' in sys.modules:
        import torch
        torch.cuda.empty_cache()

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    if llm is None or embeddings is None or pinecone_index is None:
        return jsonify({"response": "Service not ready due to initialization error. Please check logs."}), 500

    try:
        msg = request.form.get("msg", "").strip()
        if not msg:
            return jsonify({"response": "Please provide a message"})

        try:
            query_embedding = embeddings.embed_query(msg)
            
            results = pinecone_index.query(
                vector=query_embedding,
                top_k=2,
                include_values=False
            )
            
            context = "\n".join([r['id'] for r in results['matches']])
            prompt = f"Context: {context}\nQuestion: {msg}\nAnswer:"
            
            response = llm.invoke(prompt)
        except Exception as e:
            print(f"Embedding/Pinecone/LLM call error: {str(e)}", file=sys.stderr)
            response = llm.invoke(f"Question: {msg}\nAnswer:")
        
        clear_memory()
        return jsonify({"response": response})
        
    except MemoryError:
        clear_memory()
        return jsonify({"response": "Service overloaded. Please try simpler questions."}), 503
    except Exception as e:
        print(f"Unhandled error in chat: {str(e)}", file=sys.stderr)
        return jsonify({"response": "Technical difficulty. Please try again."}), 500

#if __name__ == '__main__':
 #   port = int(os.environ.get("PORT", 8080))
  #  app.run(host="0.0.0.0", port=port, debug=True, threaded=False)
