import uvicorn
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
# --- MODIFIED ---
# Import new functions and modules
from ragLogic import get_rag_chain, load_document, split_documents, ingest_chunks_to_pinecone, get_embedding_model
import tempfile
import os
# --- END MODIFIED ---
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# ---
# NEW: Add CORS middleware to allow your Streamlit frontend
# to call this API.
# ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the RAG chain and embedding model on startup
try:
    rag_chain = get_rag_chain()
    embedding_model = get_embedding_model() # Load embeddings once
    print("RAG Chain and Embedding Model loaded successfully.")
except Exception as e:
    print(f"Error loading RAG chain: {e}")
    rag_chain = None
    embedding_model = None

class Query(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"status": "RAG Backend is running"}
# --- NEW UPLOAD ENDPOINT ---
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not embedding_model:
        return {"error": "Embedding model is not loaded."}, 500

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_f:
            temp_f.write(await file.read())
            temp_filepath = temp_f.name
        
        print(f"File saved temporarily to: {temp_filepath}")

        # 1. Load the document
        docs = load_document(temp_filepath, file.filename)
        if not docs:
            return {"error": "Could not load document."}, 400

        # 2. Split the document
        chunks = split_documents(docs)
        if not chunks:
            return {"error": "Could not split document."}, 400

        # 3. Ingest chunks into Pinecone
        ingest_chunks_to_pinecone(chunks, embedding_model)

        return {
            "status": "success", 
            "filename": file.filename, 
            "chunks_ingested": len(chunks)
        }
    except Exception as e:
        print(f"Error during file upload: {e}")
        return {"error": f"Error processing file: {e}"}, 500
    finally:
        # Clean up the temporary file
        if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            print(f"Cleaned up temp file: {temp_filepath}")
# --- END NEW ENDPOINT ---

@app.post("/ask")
async def ask_question(query: Query):
    if rag_chain is None:
        return {"error": "RAG chain is not loaded."}, 500
    
    print(f"Received query: {query.query}")
    try:
        # Use the RAG chain to get an answer
        result = await rag_chain.ainvoke(query.query)
        
        # The result is a dictionary with 'context', 'question', and 'answer'
        return {"answer": result['answer']}
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        return {"error": f"Error processing query: {e}"}, 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)