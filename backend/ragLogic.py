import os
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from pinecone import Pinecone
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing import List, Any
import numpy as np # Ensure numpy is imported if it's used implicitly

# --- Environment and Model Setup ---

def get_embedding_model():
    """Loads the sentence-transformer embedding model."""
    print("Loading embedding model 'all-MiniLM-L6-v2'...")
    # Specify device as 'cpu'. Use 'cuda' if you have a GPU, or 'mps' for Apple M1/M2
    model_kwargs = {'device': 'cpu'} 
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs=model_kwargs
    )
    print("Embedding model loaded.")
    return embeddings

def get_groq_llm():
    """Initializes and returns the Groq LLM."""
    print("Initializing Groq LLM...")
    try:
        groq_api_key = os.environ["GROQ_API_KEY"]
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=1024
        )
        print("Groq LLM initialized.")
        return llm
    except KeyError:
        print("Error: GROQ_API_KEY not found in environment.")
        return None
    except Exception as e:
        print(f"Error initializing Groq: {e}")
        return None

def get_pinecone_retriever():
    """Initializes and returns the Pinecone retriever."""
    print("Connecting to Pinecone...")
    try:
        pinecone_api_key = os.environ["PINECONE_API_KEY"]
        index_name = "rag-research-kb"
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Use the serverless index
        index = pc.Index(index_name)
        
        # Get the embedding model
        embeddings = get_embedding_model()
        
        # Create the vector store object
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text" # Use "text" as the key for the document content
        )
        
        # Create the retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5} # Retrieve top 5 most relevant chunks
        )
        print("Pinecone retriever initialized.")
        return retriever
    except KeyError:
        print("Error: PINECONE_API_KEY not found.")
        return None
    except Exception as e:
        print(f"Error connecting to Pinecone: {e}")
        return None

# --- RAG Chain Creation ---

def get_rag_chain():
    """Creates and returns the full RAG chain."""
    llm = get_groq_llm()
    retriever = get_pinecone_retriever()
    
    if not llm or not retriever:
        raise ConnectionError("Failed to initialize LLM or Retriever.")

    # 1. System Prompt
    system_template = """
    You are an expert assistant for Ayush Ranjan, a software engineering intern.
    Your job is to answer questions about his research, projects, and skills based ONLY on the following context.
    Be polite and professional. If the context doesn't contain the answer, say "I'm sorry, that information is not in my knowledge base."
    Do not make up answers.

    Context:
    {context}
    """
    
    # 2. Human Prompt
    human_template = "Question: {question}"
    
    # 3. Create Prompt Templates
    system_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    
    # 4. Context Retriever
    # This chain part will:
    # 1. Take the "question" from the input dict
    # 2. Pass it to the retriever
    # 3. Format the retrieved Document objects into a single string
    context_retriever = itemgetter("question") | retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs]))

    # --- THIS IS THE CRITICAL FIX ---
    # We define the inputs for the final prompt
    # The 'context' is fetched by our context_retriever chain
    # The 'question' is passed through directly from the original input
    rag_chain = (
        {
            "context": context_retriever,
            "question": itemgetter("question"),
        }
        | chat_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- Document Processing Logic ---

def load_document(temp_filepath: str, original_filename: str) -> List[Any]:
    """Loads a document (PDF or TXT) from a temp filepath."""
    if original_filename.endswith(".pdf"):
        loader = PyMuPDFLoader(temp_filepath)
    elif original_filename.endswith(".txt"):
        loader = TextLoader(temp_filepath)
    else:
        raise ValueError("Unsupported file type")
    
    return loader.load()

def split_documents(docs: list) -> List[Any]:
    """Splits loaded documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(docs)

def ingest_chunks_to_pinecone(chunks: list, embedding_model):
    """Embeds chunks and uploads them to Pinecone."""
    index_name = "rag-research-kb"
    print(f"Ingesting {len(chunks)} chunks into Pinecone index '{index_name}'...")
    
    # This will get the text from each chunk, embed it, and upload to Pinecone
    PineconeVectorStore.from_documents(
        chunks,
        embedding=embedding_model,
        index_name=index_name,
        text_key="text" # Ensure this matches the retriever
    )
    print(f"Successfully ingested {len(chunks)} chunks.")
# --- Environment and Model Setup ---

def get_embedding_model():
    """Loads the sentence-transformer embedding model."""
    print("Loading embedding model 'all-MiniLM-L6-v2'...")
    model_name = "all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'} 
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )
    print("Embedding model loaded.")
    return embeddings

def get_groq_llm():
    """Initializes and returns the Groq LLM."""
    print("Initializing Groq LLM...")
    try:
        groq_api_key = os.environ["GROQ_API_KEY"]
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=1024
        )
        print("Groq LLM initialized.")
        return llm
    except KeyError:
        print("Error: GROQ_API_KEY not found in environment.")
        return None
    except Exception as e:
        print(f"Error initializing Groq: {e}")
        return None

def get_pinecone_retriever():
    """Initializes and returns the Pinecone retriever."""
    print("Connecting to Pinecone...")
    try:
        pinecone_api_key = os.environ["PINECONE_API_KEY"]
        index_name = "rag-research-kb"
        
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Use the serverless index
        index = pc.Index(index_name)
        
        embeddings = get_embedding_model()
        
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5} # Retrieve top 5 chunks
        )
        print("Pinecone retriever initialized.")
        return retriever
    except KeyError:
        print("Error: PINECONE_API_KEY not found.")
        return None
    except Exception as e:
        print(f"Error connecting to Pinecone: {e}")
        return None

# --- RAG Chain Creation ---

def get_rag_chain():
    """Creates and returns the full RAG chain."""
    llm = get_groq_llm()
    retriever = get_pinecone_retriever()
    
    if not llm or not retriever:
        raise ConnectionError("Failed to initialize LLM or Retriever.")

    # 1. System Prompt
    system_template = """
    You are an expert assistant for Ayush Ranjan, a software engineering intern.
    Your job is to answer questions about his research, projects, and skills based ONLY on the following context.
    If the context doesn't contain the answer, say "I'm sorry, that information is not in my knowledge base."
    
    Context:
    {context}
    """
    
    # 2. Human Prompt
    human_template = "Question: {question}"
    
    # 3. Create Prompt Templates
    system_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    
    # 4. Context Retriever
    context_retriever = itemgetter("question") | retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs]))

    #
    # --- THIS IS THE FIX ---
    #
    # 'chain_components' is the dictionary of inputs
    chain_components = {
        "context": context_retriever,
        "question": RunnablePassthrough(),
    }
    
    # 'rag_chain' is the FINAL runnable chain
    rag_chain = chain_components | chat_prompt | llm | StrOutputParser()
    
    # Return the runnable chain, not the component dictionary
    return rag_chain

# --- Document Processing Logic ---

def load_document(temp_filepath: str, original_filename: str):
    """Loads a document (PDF or TXT) from a temp filepath."""
    if original_filename.endswith(".pdf"):
        loader = PyMuPDFLoader(temp_filepath)
    elif original_filename.endswith(".txt"):
        loader = TextLoader(temp_filepath)
    else:
        raise ValueError("Unsupported file type")
    
    return loader.load()

def split_documents(docs: list):
    """Splits loaded documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(docs)

def ingest_chunks_to_pinecone(chunks: list, embedding_model):
    """Embeds chunks and uploads them to Pinecone."""
    index_name = "rag-research-kb"
    PineconeVectorStore.from_documents(
        chunks,
        embedding=embedding_model,
        index_name=index_name,
        text_key="text"
    )
    print(f"Successfully ingested {len(chunks)} chunks into Pinecone.")