import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "rag-research-kb"

def load_all_documents(text_dir="data/text_files", pdf_dir="data/pdf_files"):
    """Loads all documents from text and PDF directories."""
    all_docs = []
    
    # Load Text files
    print(f"Loading text files from {text_dir}...")
    text_loader = DirectoryLoader(
        text_dir,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},
        show_progress=True
    )
    text_docs = text_loader.load()
    all_docs.extend(text_docs)
    print(f"Loaded {len(text_docs)} text documents.")
    
    # Load PDF files
    print(f"Loading PDF files from {pdf_dir}...")
    pdf_loader = DirectoryLoader(
        pdf_dir,
        glob="*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True
    )
    pdf_docs = pdf_loader.load()
    all_docs.extend(pdf_docs)
    print(f"Loaded {len(pdf_docs)} PDF documents.")
    
    return all_docs

def split_documents(documents):
    """Splits documents into smaller chunks."""
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks.")
    return split_docs

def get_embedding_model():
    """Initializes the HuggingFace embedding model."""
    print("Loading embedding model 'all-MiniLM-L6-v2'...")
    model_name = "all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("Embedding model loaded.")
    return embeddings

def ingest_data(documents, embeddings, index_name):
    """Embeds and uploads document chunks to Pinecone."""
    if not documents:
        print("No documents to ingest.")
        return
        
    print(f"Ingesting {len(documents)} chunks into Pinecone index '{index_name}'...")
    
    # This single command does all the work:
    # 1. Generates embeddings for all chunks (in batches)
    # 2. Uploads the vectors and metadata to Pinecone
    PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=PINECONE_API_KEY
    )
    print("Data ingestion complete.")

def main():
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables.")
        
    all_docs = load_all_documents()
    split_docs = split_documents(all_docs)
    embeddings = get_embedding_model()
    ingest_data(split_docs, embeddings, INDEX_NAME)

if __name__ == "__main__":
    main()