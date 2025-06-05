from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai
import google.api_core.exceptions
from tavily import TavilyClient
import json
import asyncio
import logging

# configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# ---------- schema ----------
class ResearchQuery(BaseModel):
    query: str
    provider: str = "google"
    thinking_model: str = "gemini-1.5-flash"
    task_model: str = "gemini-1.5-flash"
    search_provider: str = "tavily"
    gemini_api_key: str
    tavily_api_key: str


# ---------- planning ----------
async def generate_plan(query, thinking_model):
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
        response = thinking_model.generate_content(planning_prompt)
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Plan generation error: {e}")
        return {
            "plan": f"Default plan for {query}",
            "subtasks": [
                {"subtask": "Default subtask 1", "search_query": query},
                {"subtask": "Default subtask 2", "search_query": f"{query} overview"}
            ]
        }

# ---------- streaming ----------
async def stream_research(query, provider, thinking_model_name, task_model_name, search_provider, tavily_key, google_key):
    genai.configure(api_key=google_key)
    thinking_model = genai.GenerativeModel(thinking_model_name)
    task_model = genai.GenerativeModel(task_model_name)
    tavily = TavilyClient(api_key=tavily_key)

    plan = await generate_plan(query, thinking_model)
    yield f"data: Planning: {json.dumps(plan, indent=2)}\n\n"
    await asyncio.sleep(0.1)

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
            yield f"data: Search Results for '{search_query}': {json.dumps(results['results'], indent=2)}\n\n"
        except Exception as e:
            logger.error(f"Search error for {search_query}: {e}")
            yield f"data: Error searching '{search_query}': {str(e)}\n\n"
            search_results.append({
                "search_query": search_query,
                "subtask": subtask["subtask"],
                "results": []
            })
        await asyncio.sleep(0.1)

    # compile research report
    summary = "\n".join([
        f"Subtask: {s['subtask']}\nSearch Query: {s['search_query']}\nResults:\n" +
        "\n".join([
            f"- {r['title']}: {r['content']} [URL: {r.get('url', 'Not Available')}]" for r in s["results"]
        ]) if s["results"] else "No results found."
        for s in search_results
    ])

    report_prompt = f"""
    Generate a detailed research report on {query} in markdown format using the following data:
    ## Research Plan
    {json.dumps(plan, indent=2)}
    ## Search Results
    {summary}

    Structure the report as follows:
    - **Abstract**: Summary (100-150 words)
    - **Introduction**: Background & objective (150-200 words)
    - **Research Findings**: Each subtask as a section with synthesis (200-300 words each)
    - **Discussion**: Trends, comparison, limitations (200-250 words)
    - **Conclusion**: Key insights & future directions (150-200 words)
    - **References**: Numbered list [Title, URL: <url>]
    """

    try:
        async def stream_report():
            for chunk in task_model.generate_content(report_prompt, stream=True):
                yield f"data: {chunk.text}\n\n"
                await asyncio.sleep(0.1)

        async for chunk in stream_report():
            yield chunk
    except google.api_core.exceptions.ResourceExhausted as e:
        yield f"data: Quota exceeded: {str(e)}\n\n"
    except Exception as e:
        yield f"data: Error generating report: {str(e)}\n\n"

# ---------- endpoint ----------
@app.post("/api/sse")
async def research_endpoint(body: ResearchQuery):
    try:

        # Dynamically configure keys
        genai.configure(api_key=query.gemini_api_key)
        tavily = TavilyClient(api_key=query.tavily_api_key)

        return StreamingResponse(
            stream_research(
                body.query,
                body.provider,
                body.thinking_model,
                body.task_model,
                body.search_provider,
                body.tavily_api_key,
                body.google_api_key
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

