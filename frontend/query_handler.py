import requests, os, logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
BASE_URL = os.getenv("BASE_URL")

# Function to validate the API Key with OpenAI
def validate_api_key(api_key: str):
    try:
        # Send a request to the OpenAI API to validate the key
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        response = requests.get("https://api.openai.com/v1/models", headers=headers)

        # If the response is successful, the API key is valid
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        return False
    
# Function to call RAG API for querying data
def call_rag_app(question, api_key):
    """ 
    Function to call the RAG API for querying data.
    :param question: The user question to query.
    :param api_key: The OpenAI API key for authorization.
    :return: The response from the API.
    """    
    api_url = f"{BASE_URL}/query"
    headers = {"Content-Type": "application/json"}
    payload = {
        "user_input": question,
        "openai_api_key": api_key 
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        return data
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling RAG API: {str(e)}")
        raise

