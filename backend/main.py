import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from ragLogic import get_rag_chain, load_document, split_documents, ingest_chunks_to_pinecone, get_embedding_model
import tempfile
import os
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow your Streamlit frontend
# to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for testing)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Global Variables ---
# Load the RAG chain and embedding model on startup
try:
    rag_chain = get_rag_chain()
    embedding_model = get_embedding_model() # Load embeddings once for uploads
    print("RAG Chain and Embedding Model loaded successfully.")
except Exception as e:
    print(f"FATAL: Error loading RAG chain on startup: {e}")
    rag_chain = None
    embedding_model = None

# --- Pydantic Models ---
class Query(BaseModel):
    query: str

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "RAG Backend is running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a file, process it, and add to the vector DB.
    """
    if not embedding_model:
        raise HTTPException(status_code=500, detail="Embedding model is not loaded.")

    temp_filepath = ""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_f:
            temp_f.write(await file.read())
            temp_filepath = temp_f.name
        
        print(f"File saved temporarily to: {temp_filepath}")

        # 1. Load the document
        docs = load_document(temp_filepath, file.filename)
        if not docs:
            raise HTTPException(status_code=400, detail="Could not load document.")

        # 2. Split the document
        chunks = split_documents(docs)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not split document.")

        # 3. Ingest chunks into Pinecone
        ingest_chunks_to_pinecone(chunks, embedding_model)

        return {
            "status": "success", 
            "filename": file.filename, 
            "chunks_ingested": len(chunks)
        }
    except Exception as e:
        print(f"Error during file upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
    finally:
        # Clean up the temporary file
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            print(f"Cleaned up temp file: {temp_filepath}")

@app.post("/ask")
async def ask_question(query: Query):
    """
    Endpoint to ask a question to the RAG chain.
    """
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG chain is not loaded.")
    
    print(f"Received query: {query.query}")
    try:
        # --- THIS IS THE CRITICAL FIX ---
        # We must pass a dictionary matching the chain's input structure
        result = await rag_chain.ainvoke({"question": query.query})
        
        # The result is now a string from StrOutputParser
        return {"answer": result}
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        # Return a 500 status code
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

if __name__ == "__main__":
    # This allows the app to be run as a script for local testing
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)