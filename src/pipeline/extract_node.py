"""
Extract Node - Enriches search results with full page content and stores in ChromaDB.

This node:
1. Extracts full raw HTML content from URLs (with fallback to basic content)
2. Chunks the content using RecursiveCharacterTextSplitter
3. Stores chunks in ChromaDB with metadata
4. Attaches chunk IDs (evidence_ids) to results for Neo4j linking

LangGraph Integration:
    Input: state["results"] - list of search results (may have partial content)
    Output: state["results"] - enriched with raw_content and chunk_ids fields
"""

from __future__ import annotations

from typing import Any, Dict, List

from tavily import TavilyClient
from src.config.settings import get_tavily_api_key
from src.pipeline.chroma_store import chunk_and_store


def get_tavily_client() -> TavilyClient:
    """Create a Tavily client with API key from settings."""
    return TavilyClient(api_key=get_tavily_api_key())


def tavily_extract_from_urls(urls: List[str], extract_depth: str | None = None) -> Dict[str, Any]:
    """
    Extract full page content from URLs using Tavily's extract API.
    
    Args:
        urls: List of URLs to extract content from
        extract_depth: Extraction depth ("basic" or "advanced")
        
    Returns:
        Dictionary with 'results' key containing extracted content per URL
    """
    client = get_tavily_client()
    kwargs: Dict[str, Any] = {"urls": urls}
    if extract_depth:
        kwargs["extract_depth"] = extract_depth
    return client.extract(**kwargs)


def extract_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Enrich search results with full page content and store in ChromaDB.
    
    Strategy:
    1. Check which results lack raw_content
    2. Attempt to extract full HTML content using Tavily extract API
    3. If extraction fails (timeout, errors), use basic content from search
    4. Chunk content and store in ChromaDB
    5. Attach chunk IDs (evidence_ids) to each result
    6. Print preview of content for debugging
    
    Graceful degradation ensures pipeline continues even if extraction fails.
    """
    results: List[Dict[str, Any]] = state.get("results", [])
    query = state.get("query", "")
    missing = [r for r in results if not r.get("raw_content")]
    
    if missing:
        urls = [r["url"] for r in missing][:20]  # Limit to 20 URLs max
        try:
            print(f"[extract_node] Extracting raw content from {len(urls)} URLs...")
            resp = tavily_extract_from_urls(urls, extract_depth="advanced")
            
            # Map URLs to their extracted content
            url_to_raw = {e.get("url"): e.get("raw_content") for e in resp.get("results", [])}
            
            # Enrich results with extracted content
            for r in results:
                if not r.get("raw_content") and r.get("url") in url_to_raw:
                    r["raw_content"] = url_to_raw[r["url"]]
            
            print("[extract_node] ✓ Successfully extracted raw content")
            
        except Exception as e:
            # Graceful fallback: use basic content from search
            print(f"[extract_node] ✗ Extract failed ({type(e).__name__}: {e})")
            print("[extract_node] Falling back to basic content from search results")
            
            for r in results:
                if not r.get("raw_content") and r.get("content"):
                    r["raw_content"] = r["content"]
    
    # Chunk and store in ChromaDB
    for r in results:
        raw_content = r.get("raw_content") or r.get("content") or ""
        if raw_content:
            url = r.get("url", "")
            title = r.get("title", "")
            
            # Chunk and store, get back chunk IDs
            chunk_ids = chunk_and_store(
                raw_content=raw_content,
                source_url=url,
                query=query,
                page_title=title
            )
            
            # Attach chunk IDs to result
            r["chunk_ids"] = chunk_ids
    
    # Debug: Show how many results have content
    with_content = sum(1 for r in results if r.get("raw_content"))
    print(f"[extract_node] {with_content}/{len(results)} results have content")
    
    # Print preview of first few results for visibility
    for i, r in enumerate(results[:2]):  # Show first 2 results
        url = r.get("url", "")
        raw = r.get("raw_content") or r.get("content") or ""
        chunk_count = len(r.get("chunk_ids", []))
        print(f"\n[extract_node] Result {i+1}: {url}")
        print(f"Chunks stored: {chunk_count}")
        if raw:
            print(f"Content preview ({len(raw)} chars): {raw[:300]}...")
        else:
            print("(no content available)")
    
    return {"results": results}
