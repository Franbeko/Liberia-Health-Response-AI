from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

_cached_embeddings_model = None

def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def get_embeddings():
    global _cached_embeddings_model
    if _cached_embeddings_model is None:
        _cached_embeddings_model = HuggingFaceEmbeddings(
            model_name='paraphrase-albert-small-v2',
            model_kwargs={'device': 'cpu'}
        )
    return _cached_embeddings_model
