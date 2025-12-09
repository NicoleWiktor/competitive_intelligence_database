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

import re
from urllib.parse import urljoin
from src.pipeline.pdf_utils import extract_pdf_text
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
    # ---------------------------------------------------------------------
    # Chunk HTML + PDF Text and Store in ChromaDB
    # ---------------------------------------------------------------------
    for r in results:
        raw_html = r.get("raw_content") or r.get("content") or ""
        url = r.get("url", "")
        title = r.get("title", "")

        # -----------------------------
        # SAVE HTML for LLM extraction
        # -----------------------------
        r["html_text"] = raw_html

        # -----------------------------
        # DETECT PDF SPEC SHEET LINKS
        # -----------------------------
        pdf_urls = []
        if raw_html:
            pdf_urls = re.findall(r'href=[\'"]([^\'"]+\.pdf)[\'"]', raw_html, re.IGNORECASE)

        resolved_pdf_urls = []
        for pdf in pdf_urls:
            if pdf.startswith("/"):
                pdf = urljoin(url, pdf)
            resolved_pdf_urls.append(pdf)

        r["pdf_urls"] = resolved_pdf_urls

        # -----------------------------
        # DOWNLOAD & EXTRACT PDF TEXT
        # -----------------------------
        pdf_texts = []
        for pdf in resolved_pdf_urls[:3]:  # Safety limit
            pdf_text = extract_pdf_text(pdf)
            if pdf_text:
                pdf_texts.append(pdf_text)

        if pdf_texts:
            full_pdf_text = (
                "=== SPEC SHEET PDF TEXT ===\n\n"
                + "\n\n--- NEW PDF ---\n\n".join(pdf_texts)
            )
            r["pdf_text"] = full_pdf_text
        else:
            r["pdf_text"] = ""

        # -----------------------------
        # PREP FOR CHUNKING
        # -----------------------------
        chunk_sources = []

        if raw_html:
            chunk_sources.append(
                ("html", raw_html)
            )

        if r["pdf_text"]:
            chunk_sources.append(
                ("pdf", r["pdf_text"])
            )

        all_chunk_ids = []

        # -----------------------------
        # CHUNK EACH SOURCE INTO ChromaDB
        # -----------------------------
        for source_type, text in chunk_sources:
            chunk_ids = chunk_and_store(
                raw_content=text,
                source_url=url,
                query=query,
                page_title=title,
                source_type=source_type,      # ADDED FIELD: html/pdf
            )
            all_chunk_ids.extend(chunk_ids)

        r["chunk_ids"] = all_chunk_ids

    
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
