import streamlit as st
import requests
import json
import sseclient
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(page_title="Deep Research Tool", layout="wide")

# Initialize session state for API keys
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""
if "tavily_api_key" not in st.session_state:
    st.session_state.tavily_api_key = ""

# Title and description
st.title("Deep Research Tool")
st.markdown("Enter your API keys and research topic to generate a detailed research report.")

# API key input section
with st.expander("Configure API Keys", expanded=True):
    gemini_api_key = st.text_input("Gemini API Key", value=st.session_state.gemini_api_key, type="password")
    tavily_api_key = st.text_input("Tavily API Key", value=st.session_state.tavily_api_key, type="password")
    if st.button("Save API Keys"):
        st.session_state.gemini_api_key = gemini_api_key
        st.session_state.tavily_api_key = tavily_api_key
        st.success("API keys saved in session.")

# Research input section
st.subheader("Research Topic")
research_query = st.text_input("Enter your research topic", placeholder="e.g., effectiveness of accelerometry in early detection of Parkinson's disease")
submit_button = st.button("Start Research")

# Placeholder for output
output_container = st.container()

if submit_button:
    if not st.session_state.gemini_api_key or not st.session_state.tavily_api_key:
        st.error("Please provide both Gemini and Tavily API keys.")
    elif not research_query:
        st.error("Please enter a research topic.")
    else:
        with output_container:
            st.subheader("Research Output")
            with st.spinner("Generating research report..."):
                plan_placeholder = st.empty()
                results_placeholder = st.empty()
                report_placeholder = st.empty()

                # Prepare request to backend
                url = "http://localhost:8000/api/sse"
                payload = {
                    "query": research_query,
                    "provider": "google",
                    "thinking_model": "gemini-1.5-flash",
                    "task_model": "gemini-1.5-flash",
                    "search_provider": "tavily"
                }
                headers = {"Content-Type": "application/json"}

                try:
                    # Stream response from backend
                    response = requests.post(url, json=payload, headers=headers, stream=True, timeout=120)
                    client = sseclient.SSEClient(response)

                    plan_text = ""
                    results_text = ""
                    report_buffer = []
                    report_text = ""

                    for event in client.events():
                        if event.data:
                            logger.debug(f"Received event data: {event.data}")
                            data = event.data.strip()
                            try:
                                if data.startswith("Planning:"):
                                    plan_data = data.replace("Planning:", "").strip()
                                    plan_json = json.loads(plan_data)
                                    plan_text = json.dumps(plan_json, indent=2)
                                    plan_placeholder.json(plan_json)
                                elif data.startswith("Search Results for"):
                                    results_text += data + "\n\n"
                                    results_placeholder.text(results_text)
                                elif data.startswith("Quota exceeded") or data.startswith("API error"):
                                    st.error(data)
                                    break
                                else:
                                    # Buffer markdown chunks
                                    report_buffer.append(data)
                                    # Render only when a complete section is likely received
                                    if data.endswith("\n") or len(report_buffer) > 5:
                                        report_text = "\n".join(report_buffer)
                                        report_placeholder.markdown(report_text, unsafe_allow_html=True)
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON parsing error: {e} - Data: {data}")
                                st.warning(f"Skipping malformed data: {data[:50]}...")
                                continue

                    # Final render of any remaining buffered markdown
                    if report_buffer:
                        report_text = "\n".join(report_buffer)
                        report_placeholder.markdown(report_text, unsafe_allow_html=True)

                except requests.exceptions.RequestException as e:
                    logger.error(f"Request exception: {e}")
                    st.error(f"Error connecting to backend: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    st.error(f"Unexpected error: {e}")
