from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
import sys
from functools import lru_cache

# Initialize Flask app first with minimal imports
app = Flask(_name_)
load_dotenv()

# Configure for low-memory environment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ.update({
    "PINECONE_API_KEY": PINECONE_API_KEY,
    "OPENAI_API_KEY": OPENAI_API_KEY
})

# Global variables for lazy-loaded components
_embeddings = None
_vector_store = None
_llm_chain = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        print("Loading embeddings...", file=sys.stderr)
        from src.helper import download_hugging_face_embeddings
        # Use smallest available model
        _embeddings = download_hugging_face_embeddings(model_name="sentence-transformers/paraphrase-albert-small-v2")
        print("Embeddings loaded", file=sys.stderr)
    return _embeddings

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        print("Loading vector store...", file=sys.stderr)
        from langchain_pinecone import PineconeVectorStore
        _vector_store = PineconeVectorStore.from_existing_index(
            index_name="liberiahealthresponseai",
            embedding=get_embeddings()
        )
        print("Vector store loaded", file=sys.stderr)
    return _vector_store

def get_retriever():
    print("Initializing retriever...", file=sys.stderr)
    return get_vector_store().as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}  # Reduced from 3 to save memory
    )

def get_llm_chain():
    global _llm_chain
    if _llm_chain is None:
        print("Loading LLM chain...", file=sys.stderr)
        from langchain_openai import OpenAI
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate
        from src.prompt import system_prompt
        
        # Lightweight LLM configuration
        llm = OpenAI(
            temperature=0.4,
            max_tokens=300,  # Reduced from 500
            model="gpt-3.5-turbo-instruct"  # More efficient model
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        _llm_chain = create_retrieval_chain(get_retriever(), question_answer_chain)
        print("LLM chain loaded", file=sys.stderr)
    return _llm_chain

@app.route("/")
def index():
    # Minimal pre-loading for the index page
    return render_template('chat.html')

@app.route("/health")
def health_check():
    """Lightweight health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return jsonify({"response": "Please provide a message"})
    
    try:
        # Load components only when first request comes in
        rag_chain = get_llm_chain()
        response = rag_chain.invoke({"input": msg})
        
        # Clear memory-intensive components after response
        if os.environ.get("FLASK_ENV") == "production":
            from gc import collect
            collect()  # Manually trigger garbage collection
            
        return jsonify({"response": response["answer"]})
        
    except MemoryError:
        return jsonify({"response": "The service is currently overloaded. Please try again later."}), 503
    except Exception as e:
        print(f"Error processing request: {str(e)}", file=sys.stderr)
        return jsonify({"response": "Sorry, we're experiencing technical difficulties. Please try again."}), 500

if _name_ == '_main_':
    port = int(os.environ.get("PORT", 8080))
    # Production configuration
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=False
    )