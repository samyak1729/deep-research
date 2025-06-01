import google.generativeai as genai
import google.api_core.exceptions
from tavily import TavilyClient
import os
from dotenv import load_dotenv
import json
import asyncio

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_GENERATIVE_AI_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
thinking_model = genai.GenerativeModel("gemini-1.5-flash")
task_model = genai.GenerativeModel("gemini-1.5-flash")

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
        print(f"Raw LLM response: {planning_response.text}")
        plan = json.loads(planning_response.text)
        print(f"Parsed plan: {json.dumps(plan, indent=2)}")
        return plan
    except google.api_core.exceptions.GoogleAPIError as e:
        print(f"API error: {e}")
        raise
    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON parsing error: {e}")
        return {
            "plan": f"Default plan for {query}",
            "subtasks": [{"subtask": "Default subtask", "search_query": query}]
        }

async def test_stream_research(query):
    print("Starting research for:", query)
    
    # Step 1: Generate plan
    plan = await generate_plan(query)
    print(f"Planning: {json.dumps(plan, indent=2)}")

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
