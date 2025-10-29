from __future__ import annotations

from typing import Any, Dict, List

from tavily import TavilyClient

from src.config.settings import get_tavily_api_key


def get_tavily_client() -> TavilyClient:
    """Create a Tavily client using API key from settings."""
    return TavilyClient(api_key=get_tavily_api_key())


def tavily_extract_from_urls(urls: List[str], extract_depth: str | None = None) -> Dict[str, Any]:
    """Wrapper around client.extract. Returns the raw dict."""
    client = get_tavily_client()
    kwargs: Dict[str, Any] = {"urls": urls}
    if extract_depth:
        kwargs["extract_depth"] = extract_depth
    return client.extract(**kwargs)


 




def extract_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Enriches search results with raw content if not already present.
    Handles timeouts gracefully - continues with basic content if extract fails.
    """
    results: List[Dict[str, Any]] = state.get("results", [])
    missing = [r for r in results if not r.get("raw_content")]
    
    if missing:
        urls = [r["url"] for r in missing][:20]
        try:
            print(f"[extract_node] Attempting to extract raw content from {len(urls)} URLs...")
            resp = tavily_extract_from_urls(urls, extract_depth="advanced")
            url_to_raw = {e.get("url"): e.get("raw_content") for e in resp.get("results", [])}
            for r in results:
                if not r.get("raw_content") and r.get("url") in url_to_raw:
                    r["raw_content"] = url_to_raw[r["url"]]
            print("[extract_node] Successfully extracted raw content")
        except Exception as e:
            print(f"[extract_node] Extract failed ({type(e).__name__}: {e}). Using basic content instead.")
            # Fallback: use the basic content from search results
            for r in results:
                if not r.get("raw_content") and r.get("content"):
                    r["raw_content"] = r["content"]
    
    print("[extract_node] results_with_raw=", sum(1 for r in results if r.get("raw_content")))
    
    # Print a preview of content for visibility
    for i, r in enumerate(results[:3]):
        url = r.get("url", "")
        raw = r.get("raw_content") or r.get("content") or ""
        print(f"[extract_node] [{i}] url={url}")
        if raw:
            snippet = raw[:500]
            if len(raw) > 500:
                snippet += "..."
            print(snippet)
        else:
            print("(no content)")
        print()
    
    return {"results": results}

