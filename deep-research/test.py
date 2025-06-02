import requests
import sseclient
from dotenv import load_dotenv
import os

# Load API keys from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def stream_backend(query: str, google_api_key: str, tavily_api_key: str):
    url = "http://localhost:8000/api/sse"
    payload = {
        "query": query,
        "provider": "google",
        "thinking_model": "gemini-1.5-flash",
        "task_model": "gemini-1.5-flash",
        "search_provider": "tavily",
        "google_api_key": google_api_key,
        "tavily_api_key": tavily_api_key
    }

    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()

        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.data:
                print(f"{event.data}")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to backend: {e}")
    except Exception as e:
        print(f"Error processing response: {e}")

if __name__ == "__main__":
    query = "effectiveness of accelerometry in early detection of Parkinson's disease"
    if not GOOGLE_API_KEY or not TAVILY_API_KEY:
        print("Error: Google and Tavily API keys must be set in .env")
    else:
        print(f"Starting research for: {query}")
        stream_backend(query, GOOGLE_API_KEY, TAVILY_API_KEY)
