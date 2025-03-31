# RAG Chatbot - Retrieval-Augmented Generation (RAG) Powered Chatbot

This repository contains a Retrieval-Augmented Generation (RAG) chatbot that leverages OpenAI's GPT models and Pinecone for semantic search and retrieval. The chatbot is designed to answer user queries by retrieving relevant documents, reranking them, and generating a response using a language model. It is built with a backend powered by FastAPI and a frontend using Streamlit.

---

## Table of Contents

1. [What is RAG?](#what-is-rag)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Setup Instructions](#setup-instructions)
5. [Environment Variables](#environment-variables)
6. [How to Run](#how-to-run)
7. [Usage](#usage)
8. [Model Details](#model-details)
9. [Contributing](#contributing)
10. [License](#license)

---

## What is RAG?

Retrieval-Augmented Generation (RAG) is a framework that combines information retrieval with generative models. Instead of relying solely on a language model's training data, RAG retrieves relevant documents from an external knowledge base (e.g., Pinecone) and uses them as context for generating responses. This approach improves the accuracy and relevance of responses, especially for domain-specific queries.

---

## Features

- **Document Retrieval**: Uses Pinecone to retrieve relevant documents based on user queries.
- **Reranking**: Reranks retrieved documents using a SentenceTransformer model for better relevance.
- **Generative Responses**: Generates responses using OpenAI's GPT-3.5-turbo model.
- **Frontend**: A user-friendly Streamlit interface for interacting with the chatbot.
- **Backend**: A FastAPI-based backend for handling queries and managing retrieval logic.

---

## Project Structure

```
backend/
    lambda_handler.py
    requirements.txt
frontend/
    app.py
    query_handler.py
    data_ingestion.py
    requirements.txt
Dockerfile
env_template
README.md
```

### Key Components

- **Backend**: Handles query processing, document retrieval, and response generation.
- **Frontend**: Provides a web interface for users to interact with the chatbot.
- **Model**: Pretrained SentenceTransformer model for embedding and reranking.

---

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Docker (optional, for containerized deployment)
- Pinecone account and API key
- OpenAI API key

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/ajith_vernekar/rag-chatbot.git
   cd rag-chatbot
   ```

2. Set up the environment variables:
   - Copy the `env_template` file to `.env`:
     ```bash
     cp env_template .env
     ```
   - Fill in the required values in the .env file:
     ```env
     OPENAI_API_KEY=<your_openai_api_key>
     PINECONE_API_KEY=<your_pinecone_api_key>
     PINECONE_ENVIRONMENT=<your_pinecone_environment>
     INDEX_NAME=<your_pinecone_index_name>
     BASE_URL=<backend_base_url>
     ```

3. Install dependencies for the backend:
   ```bash
   cd backend
   python -m venv venv_backend
   source venv_backend/bin/activate  # On Windows: venv_backend\Scripts\activate
   pip install -r requirements.txt
   
   # Download the model
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('./model')"
   ```

4. Install dependencies for the frontend:
   ```bash
   cd ../frontend
   python -m venv venv_frontend
   source venv_frontend/bin/activate  # On Windows: venv_frontend\Scripts\activate
   pip install -r requirements.txt
   ```

---

## Environment Variables

The project requires the following environment variables:

- **OpenAI Configurations**:
  - `OPENAI_API_KEY`: Your OpenAI API key for GPT models.

- **Pinecone Configurations**:
  - `PINECONE_API_KEY`: Your Pinecone API key.
  - `PINECONE_ENVIRONMENT`: The Pinecone environment (e.g., `us-west1-gcp`).
  - `INDEX_NAME`: The name of the Pinecone index.

- **Backend Configuration**:
  - `BASE_URL`: The public URL to access the backend service.

---

## How to Run

### Running Locally

#### Backend

1. Navigate to the backend folder:
   ```bash
   cd backend
   ```

2. Start the backend server:
   ```bash
   uvicorn lambda_handler:app --host 0.0.0.0 --port 8000
   ```

#### Frontend

1. Navigate to the frontend folder:
   ```bash
   cd frontend
   ```

2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your browser and go to `http://localhost:8501`.

---

### Running with Docker

1. Build the Docker image:
   ```bash
   docker build -t rag-chatbot .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8080:8080 rag-chatbot
   ```

3. The backend will be accessible at `http://localhost:8080`.

---

## Usage

### Backend API

- **Test Endpoint**: `GET /`
- **Query Endpoint**: `POST /query`
  - Request Body:
    ```json
    {
      "user_input": "What is RAG?",
      "openai_api_key": "<your_openai_api_key>"
    }
    ```
  - Response:
    ```json
    {
      "response": "RAG stands for Retrieval-Augmented Generation..."
    }
    ```

### Frontend

1. Enter your OpenAI API key in the sidebar.
2. Upload a document for indexing.
3. Ask questions about the uploaded document.

---

## Model Details

The project uses the `all-MiniLM-L6-v2` model from [SentenceTransformers](https://www.sbert.net). This model maps sentences and paragraphs to a 384-dimensional dense vector space, making it suitable for tasks like semantic search and clustering.

### Pretrained Model

- **Source**: [Hugging Face Model Hub](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Usage**:
  - For embedding: `SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')`
  - For fine-tuning: Refer to the training scripts in the model folder.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

