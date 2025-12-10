"""
Extract Node - Enriches search results with full page content and stores in ChromaDB.

ENHANCED VERSION - Processes ALL pages aggressively:
1. Extracts full raw HTML content from ALL URLs (not just first few)
2. Follows relevant links recursively (datasheets, spec pages)
3. Chunks the content using RecursiveCharacterTextSplitter
4. Stores chunks in ChromaDB with rich metadata
5. Attaches chunk IDs (evidence_ids) to results for Neo4j linking
6. Extracts structured specs from each page

LangGraph Integration:
    Input: state["results"] - list of search results (may have partial content)
    Output: state["results"] - enriched with raw_content, chunk_ids, and extracted_specs
"""

from __future__ import annotations

import re
import json
import hashlib
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

from tavily import TavilyClient
from langchain_openai import ChatOpenAI

from src.config.settings import get_tavily_api_key, get_openai_api_key
from src.pipeline.chroma_store import chunk_and_store, get_collection
from src.ontology.specifications import (
    PRESSURE_TRANSMITTER_ONTOLOGY,
    get_ontology_for_prompt,
    find_best_ontology_match,
    register_ai_derived_attribute,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Maximum URLs to extract per batch
MAX_URLS_PER_BATCH = 30

# Maximum recursive depth for link following
MAX_RECURSIVE_DEPTH = 2

# Maximum pages to extract recursively
MAX_RECURSIVE_PAGES = 15

# Timeout for each extraction (seconds)
EXTRACTION_TIMEOUT = 30

# Keywords indicating relevant pages to follow
RELEVANT_LINK_KEYWORDS = [
    "specification", "datasheet", "technical", "data sheet", "data-sheet",
    "product", "pressure", "transmitter", "catalog", "brochure",
    "price", "order", "quote", "detail", "features", "model",
    "series", "range", "overview", "pdf", "download"
]

# Keywords to exclude (avoid irrelevant pages)
EXCLUDE_KEYWORDS = [
    "login", "signin", "signup", "register", "cart", "checkout",
    "privacy", "terms", "cookie", "legal", "careers", "jobs",
    "news", "blog", "press", "media", "contact", "about-us",
    "support-ticket", "warranty-claim"
]


# =============================================================================
# TAVILY CLIENT
# =============================================================================

def get_tavily_client() -> TavilyClient:
    """Create a Tavily client with API key from settings."""
    return TavilyClient(api_key=get_tavily_api_key())


def tavily_extract_from_urls(urls: List[str], extract_depth: str = "advanced") -> Dict[str, Any]:
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


# =============================================================================
# LINK EXTRACTION AND FILTERING
# =============================================================================

def extract_links_from_content(content: str, base_url: str) -> List[str]:
    """
    Extract and filter relevant links from page content.
    
    Args:
        content: Raw page content
        base_url: Base URL for resolving relative links
        
    Returns:
        List of relevant absolute URLs
    """
    # Extract href links
    link_pattern = r'href=["\']([^"\']+)["\']'
    raw_links = re.findall(link_pattern, content, re.IGNORECASE)
    
    # Also look for PDF links
    pdf_pattern = r'["\']([^"\']+\.pdf)["\']'
    pdf_links = re.findall(pdf_pattern, content, re.IGNORECASE)
    raw_links.extend(pdf_links)
    
    base_domain = urlparse(base_url).netloc
    relevant_links = []
    seen = set()
    
    for link in raw_links:
        # Skip empty, anchors, javascript
        if not link or link.startswith('#') or link.startswith('javascript:'):
            continue
        
        # Resolve relative URLs
        full_link = urljoin(base_url, link)
        
        # Parse and validate
        parsed = urlparse(full_link)
        if not parsed.scheme or not parsed.netloc:
            continue
        
        # Normalize
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if normalized in seen:
            continue
        seen.add(normalized)
        
        # Check domain - prefer same domain or known manufacturer domains
        if parsed.netloc != base_domain:
            # Allow cross-domain only for PDFs or known domains
            if not link.endswith('.pdf'):
                continue
        
        link_lower = full_link.lower()
        
        # Exclude irrelevant pages
        if any(kw in link_lower for kw in EXCLUDE_KEYWORDS):
            continue
        
        # Check for relevant keywords
        if any(kw in link_lower for kw in RELEVANT_LINK_KEYWORDS):
            relevant_links.append(full_link)
    
    return relevant_links[:20]  # Limit to 20 links per page


# =============================================================================
# STRUCTURED SPEC EXTRACTION FROM PAGE
# =============================================================================

def extract_specs_from_content(
    content: str,
    source_url: str,
    llm: Optional[ChatOpenAI] = None
) -> Dict[str, Any]:
    """
    Extract structured specifications from page content.
    
    Uses LLM to identify and structure technical specifications.
    
    Args:
        content: Raw page content
        source_url: Source URL for evidence
        llm: Optional LLM instance (will create if not provided)
        
    Returns:
        Dict with extracted specs, products, and companies
    """
    if not content or len(content) < 100:
        return {"specs": {}, "products": [], "companies": []}
    
    if llm is None:
        llm = ChatOpenAI(
            api_key=get_openai_api_key(),
            model="gpt-4o-mini",
            temperature=0,
        )
    
    # Truncate content for LLM
    content_truncated = content[:8000]
    
    ontology_prompt = get_ontology_for_prompt()
    
    prompt = f"""Extract technical specifications from this page.

{ontology_prompt}

=== PAGE CONTENT ===
{content_truncated}

Return JSON:
{{
    "products_found": ["product model names"],
    "companies_found": ["company names"],
    "specs": {{
        "ontology_key": {{
            "value": "extracted value",
            "unit": "unit if applicable",
            "raw_text": "source quote (max 100 chars)",
            "applies_to": "product name if specific"
        }}
    }},
    "other_specs": {{
        "spec_name": {{
            "value": "value",
            "raw_text": "source quote"
        }}
    }},
    "prices": [
        {{"product": "name", "price": "value", "raw_text": "quote"}}
    ]
}}

Be thorough but only extract what you actually see in the text."""

    try:
        response = llm.invoke(prompt)
        content_resp = getattr(response, "content", str(response))
        
        # Clean response
        content_resp = content_resp.strip()
        if content_resp.startswith("```"):
            content_resp = re.sub(r"```json?\s*", "", content_resp)
            content_resp = content_resp.replace("```", "")
        
        parsed = json.loads(content_resp)
        
        # Process specs through fuzzy matching
        processed_specs = {}
        for key, spec_data in parsed.get("specs", {}).items():
            if key in PRESSURE_TRANSMITTER_ONTOLOGY:
                processed_specs[key] = spec_data
            else:
                match_key, score = find_best_ontology_match(key)
                if match_key and score >= 0.6:
                    spec_data["fuzzy_match_score"] = score
                    spec_data["original_key"] = key
                    processed_specs[match_key] = spec_data
                else:
                    # Register as AI-derived
                    register_ai_derived_attribute(
                        name=key,
                        value=str(spec_data.get("value", "")),
                        source_url=source_url
                    )
                    if "other_specs" not in parsed:
                        parsed["other_specs"] = {}
                    parsed["other_specs"][key] = spec_data
        
        # Process other_specs
        for name, data in parsed.get("other_specs", {}).items():
            register_ai_derived_attribute(
                name=name,
                value=str(data.get("value", "")),
                source_url=source_url
            )
        
        return {
            "specs": processed_specs,
            "other_specs": parsed.get("other_specs", {}),
            "products": parsed.get("products_found", []),
            "companies": parsed.get("companies_found", []),
            "prices": parsed.get("prices", []),
            "source_url": source_url
        }
        
    except Exception as e:
        print(f"[extract_node] Spec extraction error: {e}")
        return {"specs": {}, "products": [], "companies": [], "error": str(e)}


# =============================================================================
# RECURSIVE PAGE EXTRACTION
# =============================================================================

def extract_recursive(
    starting_urls: List[str],
    max_depth: int = MAX_RECURSIVE_DEPTH,
    max_pages: int = MAX_RECURSIVE_PAGES,
    query: str = ""
) -> List[Dict[str, Any]]:
    """
    Recursively extract content from URLs and their relevant links.
    
    Args:
        starting_urls: Initial URLs to extract
        max_depth: Maximum link-following depth
        max_pages: Maximum total pages to extract
        query: Original query for metadata
        
    Returns:
        List of extraction results with content and metadata
    """
    client = get_tavily_client()
    results = []
    visited: Set[str] = set()
    to_visit: List[tuple] = [(url, 0) for url in starting_urls]
    
    while to_visit and len(results) < max_pages:
        current_url, depth = to_visit.pop(0)
        
        # Skip if visited or at max depth for following
        if current_url in visited:
            continue
        visited.add(current_url)
        
        try:
            print(f"[extract_node] Extracting (depth {depth}): {current_url[:80]}...")
            response = client.extract(urls=[current_url], extract_depth="advanced")
            
            for result in response.get("results", []):
                content = result.get("raw_content", "")
                
                if not content:
                    continue
                
                # Store in ChromaDB
                chunk_ids = chunk_and_store(
                    raw_content=content,
                    source_url=current_url,
                    query=query,
                    page_title=result.get("title", "")
                )
                
                results.append({
                    "url": current_url,
                    "title": result.get("title", ""),
                    "raw_content": content,
                    "content_length": len(content),
                    "chunk_ids": chunk_ids,
                    "depth": depth,
                })
                
                # Extract and queue relevant links (if not at max depth)
                if depth < max_depth and len(results) + len(to_visit) < max_pages:
                    links = extract_links_from_content(content, current_url)
                    for link in links:
                        if link not in visited and (link, depth + 1) not in to_visit:
                            to_visit.append((link, depth + 1))
                            
        except Exception as e:
            print(f"[extract_node] Failed to extract {current_url[:50]}: {e}")
            continue
    
    return results


# =============================================================================
# BATCH EXTRACTION WITH PARALLELIZATION
# =============================================================================

def extract_urls_parallel(
    urls: List[str],
    max_workers: int = 5,
    timeout: int = EXTRACTION_TIMEOUT
) -> Dict[str, str]:
    """
    Extract content from multiple URLs in parallel.
    
    Args:
        urls: List of URLs to extract
        max_workers: Number of parallel workers
        timeout: Timeout per extraction
        
    Returns:
        Dict mapping URL to raw content
    """
    client = get_tavily_client()
    results = {}
    
    def extract_single(url: str) -> tuple:
        try:
            response = client.extract(urls=[url], extract_depth="advanced")
            for result in response.get("results", []):
                if result.get("url") == url:
                    return (url, result.get("raw_content", ""))
            return (url, "")
        except Exception as e:
            return (url, "")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_single, url): url for url in urls}
        
        for future in as_completed(futures, timeout=timeout * len(urls)):
            try:
                url, content = future.result(timeout=timeout)
                results[url] = content
            except Exception:
                continue
    
    return results


# =============================================================================
# MAIN EXTRACT NODE
# =============================================================================

def extract_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Enrich search results with full page content and store in ChromaDB.
    
    ENHANCED VERSION - Aggressive extraction:
    1. Extract content from ALL results (not just first few)
    2. Follow relevant links recursively (datasheets, spec pages)
    3. Extract structured specs from each page
    4. Store everything in ChromaDB with rich metadata
    5. Track all evidence IDs for Neo4j linking
    
    Strategy:
    1. Batch extract all URLs that lack raw_content
    2. For each page, follow relevant links (depth 1-2)
    3. Extract specs from all pages
    4. Store chunks in ChromaDB
    5. Aggregate and deduplicate results
    """
    results: List[Dict[str, Any]] = state.get("results", [])
    query = state.get("query", "")
    
    # Initialize LLM for spec extraction
    llm = ChatOpenAI(
        api_key=get_openai_api_key(),
        model="gpt-4o-mini",
        temperature=0,
    )
    
    # Collect URLs needing extraction
    urls_to_extract = []
    for r in results:
        if not r.get("raw_content"):
            url = r.get("url", "")
            if url:
                urls_to_extract.append(url)
    
    # Limit to MAX_URLS_PER_BATCH
    urls_to_extract = urls_to_extract[:MAX_URLS_PER_BATCH]
    
    if urls_to_extract:
        print(f"[extract_node] Extracting content from {len(urls_to_extract)} URLs...")
        
        try:
            # Batch extraction with Tavily
            resp = tavily_extract_from_urls(urls_to_extract, extract_depth="advanced")
            
            # Map URLs to their extracted content
            url_to_raw = {}
            url_to_title = {}
            for e in resp.get("results", []):
                url_to_raw[e.get("url")] = e.get("raw_content", "")
                url_to_title[e.get("url")] = e.get("title", "")
            
            # Enrich results with extracted content
            for r in results:
                url = r.get("url", "")
                if not r.get("raw_content") and url in url_to_raw:
                    r["raw_content"] = url_to_raw[url]
                    if not r.get("title"):
                        r["title"] = url_to_title.get(url, "")
            
            print(f"[extract_node] ✓ Successfully extracted {len(url_to_raw)} pages")
            
        except Exception as e:
            print(f"[extract_node] ✗ Batch extract failed: {e}")
            print("[extract_node] Falling back to basic content from search results")
            
            for r in results:
                if not r.get("raw_content") and r.get("content"):
                    r["raw_content"] = r["content"]
    
    # Process all results: chunk, store, and extract specs
    all_extracted_specs = []
    all_products_found = set()
    all_companies_found = set()
    all_prices_found = []
    
    for i, r in enumerate(results):
        raw_content = r.get("raw_content") or r.get("content") or ""
        url = r.get("url", "")
        title = r.get("title", "")
        
        if not raw_content:
            continue
        
        # Chunk and store in ChromaDB
        chunk_ids = chunk_and_store(
            raw_content=raw_content,
            source_url=url,
            query=query,
            page_title=title
        )
        r["chunk_ids"] = chunk_ids
        
        # Extract structured specs from the page
        print(f"[extract_node] Extracting specs from result {i+1}/{len(results)}: {url[:60]}...")
        
        try:
            extracted = extract_specs_from_content(raw_content, url, llm)
            
            r["extracted_specs"] = extracted.get("specs", {})
            r["other_specs"] = extracted.get("other_specs", {})
            r["prices_found"] = extracted.get("prices", [])
            
            # Aggregate findings
            all_extracted_specs.append({
                "url": url,
                "specs": extracted.get("specs", {}),
                "other_specs": extracted.get("other_specs", {})
            })
            
            for prod in extracted.get("products", []):
                all_products_found.add(prod)
            
            for comp in extracted.get("companies", []):
                all_companies_found.add(comp)
            
            all_prices_found.extend(extracted.get("prices", []))
            
        except Exception as e:
            print(f"[extract_node] Spec extraction failed for {url[:40]}: {e}")
            r["extracted_specs"] = {}
    
    # Follow links from pages with good content (recursive extraction)
    pages_with_content = [r for r in results if len(r.get("raw_content", "") or "") > 1000]
    
    if pages_with_content and len(results) < 20:  # Only if we have room for more
        print(f"[extract_node] Following links from {len(pages_with_content)} pages...")
        
        for r in pages_with_content[:5]:  # Limit to 5 source pages
            content = r.get("raw_content", "")
            base_url = r.get("url", "")
            
            if not content or not base_url:
                continue
            
            # Extract relevant links
            links = extract_links_from_content(content, base_url)
            
            if links:
                print(f"[extract_node] Found {len(links)} relevant links from {base_url[:50]}")
                
                # Extract first few links
                for link in links[:3]:
                    if link in [res.get("url") for res in results]:
                        continue  # Skip if already in results
                    
                    try:
                        resp = tavily_extract_from_urls([link], extract_depth="advanced")
                        
                        for ext_result in resp.get("results", []):
                            link_content = ext_result.get("raw_content", "")
                            
                            if link_content:
                                # Store in ChromaDB
                                link_chunk_ids = chunk_and_store(
                                    raw_content=link_content,
                                    source_url=link,
                                    query=f"{query} (followed link)",
                                    page_title=ext_result.get("title", "")
                                )
                                
                                # Extract specs
                                link_extracted = extract_specs_from_content(link_content, link, llm)
                                
                                # Add to results
                                results.append({
                                    "url": link,
                                    "title": ext_result.get("title", ""),
                                    "raw_content": link_content,
                                    "chunk_ids": link_chunk_ids,
                                    "extracted_specs": link_extracted.get("specs", {}),
                                    "other_specs": link_extracted.get("other_specs", {}),
                                    "prices_found": link_extracted.get("prices", []),
                                    "from_link_following": True,
                                    "parent_url": base_url,
                                })
                                
                                # Aggregate
                                all_extracted_specs.append({
                                    "url": link,
                                    "specs": link_extracted.get("specs", {}),
                                    "other_specs": link_extracted.get("other_specs", {})
                                })
                                
                                for prod in link_extracted.get("products", []):
                                    all_products_found.add(prod)
                                
                                for comp in link_extracted.get("companies", []):
                                    all_companies_found.add(comp)
                                
                                all_prices_found.extend(link_extracted.get("prices", []))
                                
                                print(f"[extract_node] ✓ Extracted specs from followed link: {link[:60]}")
                                
                    except Exception as e:
                        print(f"[extract_node] Failed to follow link {link[:40]}: {e}")
    
    # Summary statistics
    with_content = sum(1 for r in results if r.get("raw_content"))
    with_specs = sum(1 for r in results if r.get("extracted_specs"))
    total_specs = sum(len(r.get("extracted_specs", {})) for r in results)
    total_other_specs = sum(len(r.get("other_specs", {})) for r in results)
    
    print(f"\n[extract_node] === EXTRACTION SUMMARY ===")
    print(f"[extract_node] Results with content: {with_content}/{len(results)}")
    print(f"[extract_node] Results with specs: {with_specs}/{len(results)}")
    print(f"[extract_node] Total ontology specs extracted: {total_specs}")
    print(f"[extract_node] Total other specs extracted: {total_other_specs}")
    print(f"[extract_node] Products found: {len(all_products_found)}")
    print(f"[extract_node] Companies found: {len(all_companies_found)}")
    print(f"[extract_node] Prices found: {len(all_prices_found)}")
    
    # Print preview of first few results
    for i, r in enumerate(results[:3]):
        url = r.get("url", "")
        raw = r.get("raw_content") or r.get("content") or ""
        chunk_count = len(r.get("chunk_ids", []))
        spec_count = len(r.get("extracted_specs", {}))
        print(f"\n[extract_node] Result {i+1}: {url[:60]}")
        print(f"  Content: {len(raw)} chars, Chunks: {chunk_count}, Specs: {spec_count}")
        if r.get("extracted_specs"):
            for spec_name in list(r["extracted_specs"].keys())[:3]:
                print(f"  - {spec_name}: {r['extracted_specs'][spec_name].get('value', '')}")
    
    # Return enriched state
    return {
        "results": results,
        "all_extracted_specs": all_extracted_specs,
        "products_found": list(all_products_found),
        "companies_found": list(all_companies_found),
        "prices_found": all_prices_found,
    }


# =============================================================================
# STANDALONE EXTRACTION FUNCTIONS (for use outside LangGraph)
# =============================================================================

def extract_and_store_url(url: str, query: str = "") -> Dict[str, Any]:
    """
    Extract content from a single URL, store in ChromaDB, and return results.
    
    Standalone function for use outside the LangGraph pipeline.
    """
    client = get_tavily_client()
    
    try:
        response = client.extract(urls=[url], extract_depth="advanced")
        
        for result in response.get("results", []):
            content = result.get("raw_content", "")
            
            if content:
                # Store in ChromaDB
                chunk_ids = chunk_and_store(
                    raw_content=content,
                    source_url=url,
                    query=query,
                    page_title=result.get("title", "")
                )
                
                # Extract specs
                llm = ChatOpenAI(
                    api_key=get_openai_api_key(),
                    model="gpt-4o-mini",
                    temperature=0,
                )
                extracted = extract_specs_from_content(content, url, llm)
                
                return {
                    "url": url,
                    "title": result.get("title", ""),
                    "content_length": len(content),
                    "chunk_ids": chunk_ids,
                    "extracted_specs": extracted.get("specs", {}),
                    "other_specs": extracted.get("other_specs", {}),
                    "products": extracted.get("products", []),
                    "companies": extracted.get("companies", []),
                    "prices": extracted.get("prices", []),
                    "success": True
                }
        
        return {"url": url, "success": False, "error": "No content extracted"}
        
    except Exception as e:
        return {"url": url, "success": False, "error": str(e)}


def batch_extract_urls(urls: List[str], query: str = "") -> List[Dict[str, Any]]:
    """
    Extract and process multiple URLs.
    
    Standalone function for batch processing.
    """
    results = []
    for url in urls:
        result = extract_and_store_url(url, query)
        results.append(result)
        print(f"[extract] {'✓' if result.get('success') else '✗'} {url[:60]}")
    
    return results
