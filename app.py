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

        # Import inside function to reduce memory footprint
        from langchain_openai import OpenAI
        from pinecone import Pinecone
        from src.helper import get_embeddings
        
        try:
            llm = OpenAI(
                temperature=0.4,
                max_tokens=200,
                model="gpt-3.5-turbo-instruct"
            )
            
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            index = pc.Index("liberiahealthresponseai")
            
            embeddings = get_embeddings()
            query_embedding = embeddings.embed_query(msg)
            
            results = index.query(
                vector=query_embedding,
                top_k=2,
                include_values=False
            )
            
            context = "\n".join([r['id'] for r in results['matches']])
            response = llm(f"Context: {context}\nQuestion: {msg}\nAnswer:")
            
            return jsonify({"response": response})
            
        except Exception as e:
            print(f"Model error: {str(e)}", file=sys.stderr)
            # Fallback to simple OpenAI response if embeddings fail
            return jsonify({
                "response": llm(f"Question: {msg}\nAnswer:")
            })
            
    except Exception as e:
        print(f"System error: {str(e)}", file=sys.stderr)
        return jsonify({
            "response": "I'm having trouble accessing health information. Please try a more general question or try again later."
        })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)