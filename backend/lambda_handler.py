import logging, sys, os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import SentenceTransformer
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone
from mangum import Mangum
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for detailed logs
    format="%(levelname)s: %(asctime)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)  

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware to allow requests from Function URLs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("INDEX_NAME")

# Initialize retriever and memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
reranker = SentenceTransformer("model")  

def rerank_documents(query, retrieved_docs):
    """
    Rerank retrieved documents based on relevance to the query.
    """
    doc_texts = [doc.page_content for doc in retrieved_docs]
    query_embedding = reranker.encode(query, convert_to_tensor=True)
    doc_embeddings = reranker.encode(doc_texts, convert_to_tensor=True)

    # Compute similarity scores
    scores = (query_embedding @ doc_embeddings.T).tolist()
    ranked_docs = [doc for _, doc in sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)]
    return ranked_docs[:5]  # Return top 5 ranked documents

async def handle_query(user_input: str, openai_api_key: str):
    """
    Handles user queries by retrieving documents, reranking, and generating a response.
    """
    try:
        # Initialize the LLM dynamically with the provided OpenAI API key
        llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

        # Initialize Pinecone
        # pinecone, spec = initialize_pinecone()      
        pinecone = Pinecone(PINECONE_API_KEY)
        vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key))
        retriever = vectorstore.as_retriever()

        # Retrieve relevant documents
        retrieved_docs = retriever.get_relevant_documents(user_input)
        if not retrieved_docs:
            return {"response": "Sorry, I couldn't find relevant information."}

        # Rerank documents
        ranked_docs = rerank_documents(user_input, retrieved_docs)
        context = "\n".join([doc.page_content for doc in ranked_docs])

        # Generate response using LLM
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever,memory=memory)
        response = qa_chain.run({"question": user_input, "chat_history": []})

        return {"response": response}
    except Exception as e:
        logging.error(f"Error in handle_query: {str(e)}")
        raise

@app.get("/")
async def test_endpoint():
    return {"message": "Hello to RAG Application!"}

@app.post("/query")
async def query_chatbot(request: Request):
    """
    Endpoint to handle user queries.
    """
    try:
        body = await request.json()
        user_input = body.get("user_input")
        openai_api_key = body.get("openai_api_key")  # Get OpenAI API key from the request body

        if not user_input:
            raise HTTPException(status_code=400, detail="Missing 'user_input' in request body.")
        if not openai_api_key:
            raise HTTPException(status_code=400, detail="Missing 'openai_api_key' in request body.")

        return await handle_query(user_input, openai_api_key)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# AWS Lambda handler for FastAPI using Mangum
lambda_handler = Mangum(app)

# Run the app with uvicorn locally if needed
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)