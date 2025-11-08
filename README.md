Knowledge-Based RAG Chat Application
This project is a full-stack Retrieval-Augmented Generation (RAG) application. It allows users to upload documents and ask questions based on the knowledge contained within them. The application uses a FastAPI backend for the core logic and a Streamlit frontend for a user-friendly chat interface.

Features
Document-Based Q&A: Ask questions and get answers sourced directly from your documents.
Retrieval-Augmented Generation: Combines document retrieval with Large Language Models (LLMs) for accurate, context-aware answers.
Simple Web Interface: Easy-to-use chat interface built with Streamlit.
Containerized: Fully containerized with Docker for easy setup and deployment.
Tech Stack
Backend: Python, FastAPI, LangChain
Frontend: Streamlit
Containerization: Docker, Docker Compose
Project Structure
plaintext
.
├── backend/            # Contains the FastAPI application and RAG logic
│   ├── Dockerfile
│   └── ... (your python files)
├── frontend/           # Contains the Streamlit UI application
│   ├── Dockerfile
│   └── ... (your python files)
├── .env                # For storing environment variables (API keys)
├── .gitignore          # Specifies files for Git to ignore
├── docker-compose.yml  # Defines and runs the multi-container application
└── README.md           # This file
Getting Started
Follow these instructions to get the project up and running on your local machine.

Prerequisites
Git
Docker and Docker Compose (Docker Desktop includes both)
An API key from an LLM provider (e.g., OpenAI, Hugging Face).
Installation & Setup
Clone the repository:

bash
git clone https://github.com/ayushQUASAR/Knowledge_based_RAG.git
cd Knowledge_based_RAG
Create the environment file: Create a file named .env in the root of the project directory. This file will store your secret API keys. Copy the contents from the example below and replace the placeholder with your actual key.

.env file:

plaintext
# Example for OpenAI
OPENAI_API_KEY="sk-..."
Build and run the application with Docker Compose: This single command will build the Docker images for the frontend and backend services and start them.

bash
docker compose up --build
--build: Forces Docker to rebuild the images if you've made changes to the code or Dockerfile.
Usage
Once the containers are running, you can access the services:

Frontend (Streamlit): Open your web browser and navigate to http://localhost:8501
Backend (FastAPI): The API is accessible at http://localhost:8000. You can view the auto-generated API documentation at http://localhost:8000/docs.
You can now interact with the chat interface on the Streamlit app to ask questions about your knowledge base.

Stopping the Application
To stop the running containers, press Ctrl + C in the terminal where docker compose up is running.

To remove the containers completely, run:

docker compose down
