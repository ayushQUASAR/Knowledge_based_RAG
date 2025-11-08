
-----

# Knowledge-Based RAG Chat Application

This project is a full-stack, production-ready Retrieval-Augmented Generation (RAG) application. It allows users to create a dynamic knowledge base by uploading documents and then asking questions about the information contained within them.

The application is built on a scalable, decoupled architecture using a FastAPI backend for the core AI logic, a Streamlit frontend for the user interface, and Pinecone as a managed vector database.

## Features

  * **Dynamic Knowledge Base:** Upload PDF or TXT files directly through the web interface.
  * **Document-Based Q\&A:** Ask questions and get answers sourced directly from your documents.
  * **Retrieval-Augmented Generation (RAG):** Combines document retrieval with powerful Large Language Models (LLMs) for accurate, context-aware answers.
  * **High-Speed Inference:** Powered by Groq using the Llama 3.1 8B model for near-instant responses.
  * **Scalable Architecture:** Decoupled frontend (Streamlit) and backend (FastAPI) services.
  * **Cloud Vector Store:** Uses Pinecone as a production-grade, managed vector database.
  * **Containerized:** Fully containerized with Docker and Docker Compose for easy local setup and deployment.

## Tech Stack

  * **Backend:** Python, FastAPI, LangChain, Groq
  * **Frontend:** Streamlit
  * **Vector DB:** Pinecone
  * **Embeddings:** `all-MiniLM-L6-v2` (via Hugging Face)
  * **Containerization:** Docker, Docker Compose

## Project Structure

```
.
├── backend/            # FastAPI application and RAG logic
│   ├── main.py         # FastAPI server (endpoints: /ask, /upload)
│   ├── rag_logic.py    # Core RAG, Pinecone, and LLM logic
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/           # Streamlit UI application
│   ├── app.py          # Streamlit chat and upload interface
│   ├── Dockerfile
│   └── requirements.txt
├── data/               # Local folder for your documents
│   └── ... (your .pdf and .txt files)
├── .env                # For storing environment variables (API keys)
├── .gitignore          # Specifies files for Git to ignore
├── docker-compose.yml  # Defines and runs the multi-container application
├── ingest.py           # Optional: Script for bulk-uploading data
└── README.md           # This file
```

## Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

  * Git
  * Docker and Docker Compose (Docker Desktop includes both)
  * **Groq API Key:** Get from [GroqCloud](https://console.groq.com/keys)
  * **Pinecone API Key:** Get from [Pinecone](https://www.pinecone.io/)
  * You must also create a serverless index in Pinecone named `rag-research-kb` with `384` dimensions and the `cosine` metric.

### Installation & Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ayushQUASAR/Knowledge_based_RAG.git
    cd Knowledge_based_RAG
    ```

2.  **Create the environment file:**
    Create a file named `.env` in the root of the project directory. This file will store your secret API keys.

    **.env file:**

    ```
    GROQ_API_KEY="gsk_..."
    PINECONE_API_KEY="your-pinecone-api-key"
    ```

3.  **Populate your Knowledge Base (Optional):**
    You can either upload files via the web interface (see *Usage*) or do a one-time bulk ingest. To bulk ingest:

      * Place your PDF and TXT files into the `data/` folder.
      * Create a local virtual environment: `python3 -m venv venv && source venv/bin/activate`
      * Install requirements: `pip install -r backend/requirements.txt`
      * Run the ingest script: `python ingest.py`

4.  **Build and run the application with Docker Compose:**
    This single command will build the Docker images for the frontend and backend services and start them.

    ```bash
    docker compose up --build
    ```

      * `--build`: Forces Docker to rebuild the images. Use this when you change the code.

### Usage

Once the containers are running, you can access the services:

  * **Frontend (Streamlit):** Open your web browser and navigate to `http://localhost:8501`
  * **Backend (FastAPI):** The API is accessible at `http://localhost:8000`. You can view the auto-generated API documentation at `http://localhost:8000/docs`.

You can now use the sidebar in the Streamlit app to upload new documents or chat with the knowledge base.

### Stopping the Application

To stop the running containers, press `Ctrl + C` in the terminal where `docker compose up` is running.

To remove the containers completely, run:

```bash
docker compose down
```
