# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the default Streamlit port (8501)
# Note: Hugging Face Spaces will map its internal port (7860) to this.
EXPOSE 8501

# Command to run the Streamlit application
# We use port 8501 as the internal port
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]