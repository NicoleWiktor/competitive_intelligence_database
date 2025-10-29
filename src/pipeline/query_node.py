from __future__ import annotations

from typing import Any, Dict, List

from tavily import TavilyClient

from src.config.settings import get_tavily_api_key


def get_tavily_client() -> TavilyClient:
    """Create a Tavily client from settings."""
    return TavilyClient(api_key=get_tavily_api_key())


def tavily_search(query: str, max_results: int = 5, include_raw_content: bool = True) -> List[Dict[str, Any]]:
    """Follow the Tavily tutorial: perform a web search and (optionally) return raw content."""
    client = get_tavily_client()
    resp = client.search(query=query, max_results=max_results, include_raw_content=include_raw_content)
    results: List[Dict[str, Any]] = []
    for r in resp.get("results", []):
        results.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score"),
                "raw_content": r.get("raw_content"),
            }
        )
    return results
# LangGraph node wrapper
def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("query", "")
    max_results = state.get("max_results", 3)
    results = tavily_search(query, max_results=max_results, include_raw_content=True)
    print("[search_node] query=", query, "results=", len(results))
    return {"results": results}

