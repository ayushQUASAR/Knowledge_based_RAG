import streamlit as st
import requests
import os


BACKEND_API_URL = os.environ.get("BACKEND_API_URL")

st.set_page_config(layout="wide")
st.title("Internship Research Knowledge Base 🧠")
st.markdown("Ask me anything about my research on MANETs, OLSR, or other topics in my knowledge base.")

# --- NEW: Sidebar for File Upload ---
with st.sidebar:
    st.header("Upload Documents")
    st.markdown("Add new PDF or TXT files to the knowledge base.")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if st.button("Upload and Process"):
            if not BACKEND_API_URL:
                st.error("Backend URL not set. Cannot upload.")
            else:
                with st.spinner(f"Uploading {uploaded_file.name}..."):
                    try:
                        # Send file to backend /upload endpoint
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        response = requests.post(f"{BACKEND_API_URL}/upload", files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"Success! Ingested {result['chunks_ingested']} chunks from {result['filename']}.")
                        else:
                            st.error(f"Error: {response.json().get('error', response.text)}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Failed to connect to backend: {e}")
# --- END NEW ---


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is the OLSR protocol?"):
    if not BACKEND_API_URL:
        st.error("Error: BACKEND_API_URL is not set. The frontend can't find the backend.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Send the query to the FastAPI backend
                    response = requests.post(
                        f"{BACKEND_API_URL}/ask",  # The /ask endpoint
                        json={"query": prompt}
                    )
                    
                    if response.status_code == 200:
                        answer = response.json().get("answer", "No answer found.")
                        st.markdown(answer)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.error(f"Error from backend: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to backend: {e}")