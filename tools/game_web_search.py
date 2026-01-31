import os
from tavily import TavilyClient
from config.env import load_env
from lib.tooling import tool


@tool
def game_web_search(question: str, max_results: int = 5):
    """
    Searches the web using Tavily to answer game industry questions.

    Args:
        question: A question about the game industry
        max_results: Number of web results to retrieve

    Returns:
        A list of web search results containing:
        - title
        - url
        - snippet
    """

    # Load environment variables
    load_env()

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("TAVILY_API_KEY is missing from environment variables")

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=question,
        max_results=max_results
    )

    results = []

    for item in response.get("results", []):
        results.append({
            "title": item.get("title"),
            "url": item.get("url"),
            "snippet": item.get("content")
        })

    return results
