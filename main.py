from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai
import google.api_core.exceptions
from tavily import TavilyClient
import os
from dotenv import load_dotenv
import json
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_GENERATIVE_AI_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
thinking_model = genai.GenerativeModel("gemini-1.5-flash")
task_model = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI()

class ResearchQuery(BaseModel):
    query: str
    provider: str = "google"
    thinking_model: str = "gemini-1.5-flash"
    task_model: str = "gemini-1.5-flash"
    search_provider: str = "tavily"

async def generate_plan(query):
    planning_prompt = f"""
    Return a structured research plan for the topic: "{query}" as valid JSON only, with no additional text, markdown, or code fences. Example:
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
    Ensure at least two subtasks, each with a non-empty search_query relevant to the topic.
    """
    try:
        planning_response = thinking_model.generate_content(planning_prompt)
        plan = json.loads(planning_response.text)
        logger.debug(f"Generated plan: {json.dumps(plan, indent=2)}")
        return plan
    except google.api_core.exceptions.GoogleAPIError as e:
        logger.error(f"API error in generate_plan: {e}")
        raise
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"JSON parsing error: {e}")
        return {
            "plan": f"Default plan for {query}",
            "subtasks": [
                {"subtask": "Default subtask 1", "search_query": query},
                {"subtask": "Default subtask 2", "search_query": f"{query} overview"}
            ]
        }

async def stream_research(query: str, provider: str, thinking_model_name: str, task_model_name: str, search_provider: str):
    # Step 1: Generate plan
    plan = await generate_plan(query)
    logger.debug(f"Sending plan: {json.dumps(plan, indent=2)}")
    yield f"data: Planning: {json.dumps(plan, indent=2)}\n\n"
    await asyncio.sleep(0.1)  # Prevent client overload

    # Step 2: Perform web search
    search_results = []
    for subtask in plan["subtasks"]:
        search_query = subtask["search_query"]
        try:
            results = tavily.search(query=search_query, max_results=5)
            search_results.append({
                "search_query": search_query,
                "subtask": subtask["subtask"],
                "results": results["results"]
            })
            logger.debug(f"Sending search results for '{search_query}': {json.dumps(results['results'], indent=2)}")
            yield f"data: Search Results for '{search_query}': {json.dumps(results['results'], indent=2)}\n\n"
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error searching '{search_query}': {str(e)}")
            search_results.append({
                "search_query": search_query,
                "subtask": subtask["subtask"],
                "results": []
            })
            yield f"data: Error searching '{search_query}': {str(e)}\n\n"
            await asyncio.sleep(0.1)

    # Step 3: Generate report
    search_summary = "\n".join([
        f"Subtask: {item['subtask']}\nSearch Query: {item['search_query']}\nResults:\n" +
        ("\n".join([f"- {result['title']}: {result['content']} [URL: {result.get('url', 'Not Available')}]" for result in item["results"]])
         if item["results"] else "No results found.")
        for item in search_results
    ])
    report_prompt = f"""
    Generate a detailed research report on {query} in markdown format using the following data:
    ## Research Plan
    {json.dumps(plan, indent=2)}
    ## Search Results
    {search_summary}

    Structure the report as follows:
    - **Abstract**: A concise summary (100-150 words) of the research topic, objectives, and key findings.
    - **Introduction**: Introduce the topic, its significance, and the research objectives based on the plan (150-200 words).
    - **Research Findings**: For each subtask, create a dedicated section with:
      - A clear heading matching the subtask description.
      - An objective statement explaining the purpose of this research segment.
      - Detailed findings (200-300 words per section) synthesizing the search results, highlighting key insights, and addressing the objective.
      - Inline citations in the format [Source: Title, URL: <url>] or [Source: Title, URL: Not Available].
    - **Discussion**: Analyze the findings across subtasks, comparing and contrasting results, identifying trends, and noting limitations (200-250 words).
    - **Conclusion**: Summarize key insights, implications, and potential future research directions (150-200 words).
    - **References**: A numbered list of all cited sources in the format:
      1. Title, URL: <url>
      2. Title, URL: Not Available
      Do not omit or replace URLs.

    Ensure the report is comprehensive, concise, and complete, with each section serving its research objective. Use clear, formal language suitable for an academic or professional audience.
    """
    try:
        async def stream_report():
            for chunk in task_model.generate_content(report_prompt, stream=True):
                logger.debug(f"Sending report chunk: {chunk.text[:50]}...")
                yield f"data: {chunk.text}\n\n"
                await asyncio.sleep(0.1)  # Prevent client overload
        async for chunk in stream_report():
            yield chunk
    except google.api_core.exceptions.ResourceExhausted as e:
        logger.error(f"Quota exceeded: {e}")
        yield f"data: Quota exceeded during report generation: {str(e)}\n\n"
    except google.api_core.exceptions.GoogleAPIError as e:
        logger.error(f"API error: {e}")
        yield f"data: API error in report generation: {str(e)}\n\n"

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
        logger.error(f"Endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
