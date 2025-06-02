import streamlit as st
import requests
import sseclient

st.title("Deep Research Tool")

# Form for user input
with st.form("research_form"):
    query = st.text_input("Research Query", value="effectiveness of accelerometry in early detection of Parkinson's disease")
    google_api_key = st.text_input("Google API Key", type="password")
    tavily_api_key = st.text_input("Tavily API Key", type="password")
    submit_button = st.form_submit_button("Start Research")

if submit_button:
    if not query or not google_api_key or not tavily_api_key:
        st.error("Please provide a query and both API keys.")
    else:
        # Prepare request payload
        payload = {
            "query": query,
            "provider": "google",
            "thinking_model": "gemini-1.5-flash",
            "task_model": "gemini-1.5-flash",
            "search_provider": "tavily",
            "google_api_key": google_api_key,
            "tavily_api_key": tavily_api_key
        }

        # Placeholder for streamed output
        output_container = st.empty()

        try:
            # Stream response from FastAPI
            response = requests.post("http://localhost:8000/api/sse", json=payload, stream=True)
            response.raise_for_status()

            client = sseclient.SSEClient(response)
            full_output = ""
            for event in client.events():
                if event.data:
                    full_output += event.data + "\n"
                    output_container.markdown(full_output)
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to backend: {e}")
        except Exception as e:
            st.error(f"Error processing response: {e}")
