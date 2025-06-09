from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
import sys
import gc

app = Flask(__name__)
load_dotenv()

# Memory optimization settings
os.environ.update({
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:16",
    "TOKENIZERS_PARALLELISM": "false",
    "TF_CPP_MIN_LOG_LEVEL": "3",  # Reduce TensorFlow logging
    "PINECONE_API_KEY": os.getenv('PINECONE_API_KEY'),
    "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY')
})

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
    try:
        msg = request.form.get("msg", "").strip()
        if not msg:
            return jsonify({"response": "Please provide a message"})

        # Load minimal components only when needed
        from langchain_openai import OpenAI
        from langchain.chains import RetrievalQA
        from src.helper import get_embeddings
        
        # Initialize with minimal memory footprint
        llm = OpenAI(
            temperature=0.4,
            max_tokens=200,  # Further reduced
            model="gpt-3.5-turbo-instruct"
        )
        
        # Use direct Pinecone retrieval without langchain wrapper
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index("liberiahealthresponseai")
        
        # Get embeddings
        embeddings = get_embeddings()
        query_embedding = embeddings.embed_query(msg)
        
        # Direct Pinecone query
        results = index.query(
            vector=query_embedding,
            top_k=2,  # Reduced from 3
            include_values=False
        )
        
        # Simple response formatting
        context = "\n".join([r['id'] for r in results['matches']])
        prompt = f"Context: {context}\nQuestion: {msg}\nAnswer:"
        
        response = llm(prompt)
        clear_memory()
        
        return jsonify({"response": response})
        
    except MemoryError:
        clear_memory()
        return jsonify({"response": "Service overloaded. Please try simpler questions."}), 503
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return jsonify({"response": "Technical difficulty. Please try again."}), 500

if __name__ == '_main_':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=False)