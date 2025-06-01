from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai
from tavily import TavilyClient
import os
from dotenv import load_dotenv
import json
import asyncio

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_GENERATIVE_AI_API_KEY"))
thinking_model = genai.GenerativeModel("gemini-1.5-flash")
task_model = genai.GenerativeModel("gemini-1.5-flash")

# Configure Tavily for web search
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

app = FastAPI()

class ResearchQuery(BaseModel):
    query: str
    provider: str = "google"
    thinking_model: str = "gemini-1.5-flash"
    task_model: str = "gemini-1.5-flash"
    search_provider: str = "tavily"

async def stream_research(query: str, provider: str, thinking_model_name: str, task_model_name: str, search_provider: str):
    # Step 1: Generate search plan with subtasks as search queries
    planning_prompt = f"""
    Create a structured research plan for the topic: {query}. Return the response in JSON format:
    {{
      "plan": "Overall research plan description",
      "subtasks": [
        {{
          "subtask": "Description of subtask 1",
          "search_query": "Specific search query for subtask 1"
        }},
        {{
          "subtask": "Description of subtask 2",
          "search_query": "Specific search query for subtask 2"
        }}
      ]
    }}
    Ensure each subtask has a corresponding search query to be used for web search.
    """
    try:
        planning_response = thinking_model.generate_content(planning_prompt)
        plan = json.loads(planning_response.text)
        yield f"data: Planning: {json.dumps(plan, indent=2)}\n\n"
    except (json.JSONDecodeError, ValueError) as e:
        plan = {
            "plan": f"Default plan for {query}",
            "subtasks": [{"subtask": "Default subtask", "search_query": query}]
        }
        yield f"data: Planning: {json.dumps(plan, indent=2)}\n\n"

    # Step 2: Perform web search for each subtask
    search_results = []
    for subtask in plan["subtasks"]:
        search_query = subtask["search_query"]
        try:
            results = tavily.search(query=search_query, max_results=10)  # Fetch 5–10 results
            search_results.append({
                "search_query": search_query,
                "subtask": subtask["subtask"],
                "results": results["results"]
            })
            yield f"data: Search Results for '{search_query}': {json.dumps(results['results'], indent=2)}\n\n"
        except Exception as e:
            yield f"data: Error searching '{search_query}': {str(e)}\n\n"

    # Step 3: Generate report with task model
    search_summary = "\n".join([
        f"Subtask: {item['subtask']}\nSearch Query: {item['search_query']}\nResults:\n" +
        "\n".join([f"- {result['title']}: {result['content']}" for result in item["results"]])
        for item in search_results
    ])
    report_prompt = f"""
    Generate a comprehensive research report on {query} in markdown format. Use the following data:
    ## Research Plan
    {json.dumps(plan, indent=2)}
    ## Search Results
    {search_summary}
    Structure the report with:
    - An introduction summarizing the topic and plan
    - Sections for each subtask with summarized findings and citations
    - A conclusion synthesizing key insights
    Include citations in the format [Source: Title, URL].
    """
    async for chunk in task_model.generate_content(report_prompt, stream=True):
        yield f"data: {chunk.text}\n\n"

@app.post("/api/sse")
async def research_endpoint(query: ResearchQuery):
    try:
        return StreamingResponse(
            stream_research(
                query.query,
                query.provider,
                query.thinking_model,
                query.task_model,
                query.search_provider
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
import google.generativeai as genai
import google.api_core.exceptions  # Add this import
from tavily import TavilyClient
import os
from dotenv import load_dotenv
import json
import asyncio
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_GENERATIVE_AI_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
thinking_model = genai.GenerativeModel("gemini-1.5-flash")  # Using lighter model
task_model = genai.GenerativeModel("gemini-1.5-flash")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(20),
    retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted)
)
async def generate_plan(query):
    planning_prompt = f"""
    Create a structured research plan for the topic: {query}. Return valid JSON:
    {{
      "plan": "Overall research plan description",
      "subtasks": [
        {{
          "subtask": "Description of subtask 1",
          "search_query": "Specific search query for subtask 1"
        }},
        {{
          "subtask": "Description of subtask 2",
          "search_query": "Specific search query for subtask 2"
        }}
      ]
    }}
    Ensure the response is valid JSON with at least two subtasks, each with a search_query.
    """
    planning_response = thinking_model.generate_content(planning_prompt)
    return json.loads(planning_response.text)

async def test_stream_research(query):
    print("Starting research for:", query)
    
    # Step 1: Generate plan
    try:
        plan = await generate_plan(query)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON parsing error: {e}")
        plan = {
            "plan": f"Default plan for {query}",
            "subtasks": [{"subtask": "Default subtask", "search_query": query}]
        }
    print(f"Planning: {json.dumps(plan, indent=2)}")

    # Step 2: Perform web search
    search_results = []
    for subtask in plan["subtasks"]:
        search_query = subtask["search_query"]
        try:
            results = tavily.search(query=search_query, max_results=5)  # Reduced for testing
            search_results.append({
                "search_query": search_query,
                "subtask": subtask["subtask"],
                "results": results["results"]
            })
            print(f"Search Results for '{search_query}': {json.dumps(results['results'], indent=2)}")
        except Exception as e:
            print(f"Error searching '{search_query}': {str(e)}")

    # Step 3: Generate report
    search_summary = "\n".join([
        f"Subtask: {item['subtask']}\nSearch Query: {item['search_query']}\nResults:\n" +
        "\n".join([f"- {result['title']}: {result['content']}" for result in item["results"]])
        for item in search_results
    ])
    report_prompt = f"""
    Generate a comprehensive research report on {query} in markdown format. Use the following data:
    ## Research Plan
    {json.dumps(plan, indent=2)}
    ## Search Results
    {search_summary}
    Structure the report with:
    - An introduction summarizing the topic and plan
    - Sections for each subtask with summarized findings and citations
    - A conclusion synthesizing key insights
    Include citations in the format [Source: Title, URL].
    """
    try:
        for chunk in task_model.generate_content(report_prompt, stream=True):
            print(chunk.text)
    except google.api_core.exceptions.ResourceExhausted as e:
        print(f"Quota exceeded during report generation: {e}")

if __name__ == "__main__":
    asyncio.run(test_stream_research("effectiveness of accelerometry in early detection of Parkinson's disease"))
