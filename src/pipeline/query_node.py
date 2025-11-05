"""
Query Node - Performs web search using Tavily API.

This node is the entry point for data collection. It takes a search query
from the state and uses Tavily's web search API to find relevant URLs.

LangGraph Integration:
    Input: state["query"] - search query string
    Output: state["results"] - list of search result dictionaries
"""

from __future__ import annotations

from typing import Any, Dict, List

from tavily import TavilyClient
from src.config.settings import get_tavily_api_key


def get_tavily_client() -> TavilyClient:
    """Create a Tavily client with API key from settings."""
    return TavilyClient(api_key=get_tavily_api_key())


def tavily_search(query: str, max_results: int = 5, include_raw_content: bool = True) -> List[Dict[str, Any]]:
    """
    Perform web search using Tavily API.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        include_raw_content: Whether to include raw HTML content
        
    Returns:
        List of dictionaries with keys: title, url, content, score, raw_content
    """
    client = get_tavily_client()
    resp = client.search(query=query, max_results=max_results, include_raw_content=include_raw_content)
    
    results: List[Dict[str, Any]] = []
    for r in resp.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),  # Short summary from Tavily
            "score": r.get("score"),
            "raw_content": r.get("raw_content"),  # Full HTML content
        })
    return results


def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Execute web search based on current query.
    
    This is called at the start of each iteration with either:
    - The original query (first iteration)
    - A refined query (subsequent iterations)
    """
    query = state.get("query", "")
    max_results = state.get("max_results", 1)  # Process 1 URL per iteration for control
    
    results = tavily_search(query, max_results=max_results, include_raw_content=True)
    print(f"[search_node] query='{query}' | found {len(results)} results")
    
    return {"results": results}

