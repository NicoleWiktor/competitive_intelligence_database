"""
Agent Tools - The capabilities available to the Competitive Intelligence Agent.

These tools follow the LangChain Tool pattern and can be used by the ReAct agent
to dynamically search, extract, verify, and analyze competitive intelligence.

AGENTIC PRINCIPLE: The AI decides WHEN and HOW to use these tools based on its
reasoning about what data is missing and what would be most valuable to find.

ENHANCED FEATURES:
- Multi-step search with query variations and synonyms
- Recursive page extraction (follows links)
- Aggressive spec extraction with fuzzy ontology matching
- Unit normalization
- ChromaDB evidence storage
"""

from __future__ import annotations

import json
import re
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from src.config.settings import get_tavily_api_key, get_openai_api_key
from src.ontology.specifications import (
    PRESSURE_TRANSMITTER_ONTOLOGY,
    get_ontology_for_prompt,
    get_aggressive_extraction_prompt,
    find_best_ontology_match,
    normalize_spec_value,
    register_ai_derived_attribute,
    normalize_pressure,
    PRESSURE_TO_PSI,
)
from src.pipeline.chroma_store import chunk_and_store, query_evidence


# =============================================================================
# QUERY EXPANSION - SYNONYMS AND VARIATIONS
# =============================================================================

# Industry synonyms for pressure transmitters
PRODUCT_SYNONYMS = {
    "pressure transmitter": [
        "pressure transducer",
        "pressure sensor",
        "pressure gauge transmitter",
        "pressure measurement device",
        "process pressure transmitter",
        "industrial pressure transmitter",
        "pressure instrument",
        "pressure measuring instrument",
    ],
    "specifications": [
        "specs", "technical specifications", "datasheet",
        "technical data", "product specifications", "spec sheet",
        "product data", "technical specs"
    ],
    "price": [
        "pricing", "cost", "quote", "buy", "purchase",
        "order", "list price", "msrp", "price list"
    ],
    "review": [
        "reviews", "feedback", "customer review", "user review",
        "testimonial", "rating", "experience", "opinion"
    ],
}

# Major competitors to search for
KNOWN_COMPETITORS = [
    "Emerson Rosemount",
    "Siemens",
    "ABB",
    "Endress+Hauser",
    "Yokogawa",
    "WIKA",
    "Danfoss",
    "Schneider Electric",
    "Fuji Electric",
    "Dwyer Instruments",
    "Ashcroft",
    "VEGA",
    "Krohne",
    "Sensata",
    "TE Connectivity",
]

# URLs we've already visited to avoid duplicates
VISITED_URLS: Set[str] = set()


def generate_search_variations(base_query: str, num_variations: int = 8) -> List[str]:
    """
    Generate multiple search query variations from a base query.
    
    Uses synonym expansion and reformulation to create diverse queries.
    
    Args:
        base_query: The original search query
        num_variations: Number of variations to generate (default 8)
    
    Returns:
        List of query variations
    """
    variations = [base_query]
    base_lower = base_query.lower()
    
    # Apply synonym substitutions
    for term, synonyms in PRODUCT_SYNONYMS.items():
        if term in base_lower:
            for syn in synonyms[:3]:  # Limit to 3 synonyms per term
                variation = base_lower.replace(term, syn)
                if variation not in [v.lower() for v in variations]:
                    variations.append(variation)
    
    # Add industry-specific variations
    if "pressure" in base_lower:
        variations.extend([
            f"{base_query} industrial",
            f"{base_query} process automation",
            f"{base_query} 4-20mA HART",
        ])
    
    # Add datasheet variations
    if "spec" in base_lower or "data" in base_lower:
        variations.extend([
            base_query.replace("specifications", "datasheet"),
            base_query.replace("specs", "technical data sheet"),
            f"{base_query} PDF",
        ])
    
    # Add comparison variations
    variations.extend([
        f"{base_query} vs",
        f"{base_query} comparison",
        f"{base_query} alternative",
    ])
    
    # Deduplicate and limit
    seen = set()
    unique = []
    for v in variations:
        v_lower = v.lower().strip()
        if v_lower not in seen:
            seen.add(v_lower)
            unique.append(v)
    
    return unique[:num_variations]


def get_competitor_search_queries(competitor: str) -> List[str]:
    """Generate comprehensive search queries for a competitor."""
    queries = [
        f"{competitor} pressure transmitter",
        f"{competitor} pressure transducer",
        f"{competitor} pressure transmitter specifications",
        f"{competitor} pressure transmitter price",
        f"{competitor} pressure transmitter datasheet",
        f"{competitor} industrial pressure sensor",
        f"{competitor} process pressure transmitter 4-20mA",
        f"{competitor} pressure transmitter vs Honeywell",
    ]
    return queries


def get_product_search_queries(product: str, company: str) -> List[str]:
    """Generate comprehensive search queries for a specific product."""
    queries = [
        f"{product} {company} specifications",
        f"{product} datasheet",
        f"{product} technical data",
        f"{product} price",
        f"{product} accuracy range",
        f"{product} pressure transmitter specs",
        f"{company} {product} buy",
        f"{product} review industrial",
    ]
    return queries


# =============================================================================
# TOOL: ENHANCED WEB SEARCH WITH VARIATIONS
# =============================================================================

@tool
def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web for competitive intelligence information.
    
    This enhanced version:
    - Uses advanced search depth
    - Returns more results (up to 10)
    - Includes raw content length for prioritization
    - Tracks visited URLs to avoid duplicates
    
    Args:
        query: A specific search query. Be precise and include company/product names.
        max_results: Number of results to return (1-10)
    
    Returns:
        Search results with titles, URLs, and content snippets.
    """
    global VISITED_URLS
    client = TavilyClient(api_key=get_tavily_api_key())
    
    try:
        response = client.search(
            query=query,
            max_results=min(max_results, 10),
            include_raw_content=True,
            search_depth="advanced"
        )
        
        results = []
        for r in response.get("results", []):
            url = r.get("url", "")
            
            # Track URL but don't skip (agent may want to re-visit)
            is_new = url not in VISITED_URLS
            VISITED_URLS.add(url)
            
            results.append({
                "title": r.get("title", ""),
                "url": url,
                "content": r.get("content", "")[:800],  # More content
                "raw_content_length": len(r.get("raw_content", "") or ""),
                "is_new_url": is_new,
            })
        
        return json.dumps(results, indent=2)
    
    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def search_with_variations(base_query: str, num_variations: int = 5) -> str:
    """
    Search the web using multiple query variations to find more results.
    
    This tool automatically generates synonyms and reformulations of your
    query to capture more diverse results.
    
    Args:
        base_query: The base search query to expand
        num_variations: Number of query variations to try (1-8)
    
    Returns:
        Combined results from all query variations with deduplication.
    """
    global VISITED_URLS
    client = TavilyClient(api_key=get_tavily_api_key())
    
    variations = generate_search_variations(base_query, min(num_variations, 8))
    all_results = []
    seen_urls = set()
    
    for query in variations:
        try:
            response = client.search(
                query=query,
                max_results=5,
                include_raw_content=True,
                search_depth="advanced"
            )
            
            for r in response.get("results", []):
                url = r.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    VISITED_URLS.add(url)
                    all_results.append({
                        "title": r.get("title", ""),
                        "url": url,
                        "content": r.get("content", "")[:600],
                        "raw_content_length": len(r.get("raw_content", "") or ""),
                        "query_used": query,
                    })
        except Exception as e:
            continue  # Skip failed queries, continue with others
    
    # Add guidance for the agent on what to do next
    result = {
        "queries_used": variations,
        "total_results": len(all_results),
        "results": all_results[:15],  # Limit to top 15
        "NEXT_STEPS": "Use extract_page_content on the most relevant URLs above to get full content, then save_competitor, save_product, and save_multiple_specs with what you find."
    }
    return json.dumps(result, indent=2)


@tool
def search_competitor_products(competitor_name: str) -> str:
    """
    Comprehensive search for a competitor's pressure transmitter products.
    
    Generates multiple targeted queries to find:
    - Product names and models
    - Technical specifications
    - Pricing information
    - Datasheets
    
    Args:
        competitor_name: The competitor company name (e.g., "Emerson", "WIKA")
    
    Returns:
        Combined search results for the competitor's products.
    """
    client = TavilyClient(api_key=get_tavily_api_key())
    queries = get_competitor_search_queries(competitor_name)
    
    all_results = []
    seen_urls = set()
    
    for query in queries[:6]:  # Limit to 6 queries per competitor
        try:
            response = client.search(
                query=query,
                max_results=5,
                include_raw_content=True,
                search_depth="advanced"
            )
            
            for r in response.get("results", []):
                url = r.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append({
                        "title": r.get("title", ""),
                        "url": url,
                        "content": r.get("content", "")[:500],
                        "query": query,
                    })
        except:
            continue
    
    return json.dumps({
        "competitor": competitor_name,
        "queries_executed": len(queries[:6]),
        "unique_results": len(all_results),
        "results": all_results
    }, indent=2)


# =============================================================================
# TOOL: ENHANCED PAGE EXTRACTION WITH LINK FOLLOWING
# =============================================================================

@tool
def extract_page_content(url: str) -> str:
    """
    Extract full content from a specific URL for detailed analysis.
    
    Also stores the content in ChromaDB for evidence tracking.
    
    Args:
        url: The URL to extract content from.
    
    Returns:
        The full text content of the page (up to 15000 chars).
    """
    client = TavilyClient(api_key=get_tavily_api_key())
    
    try:
        response = client.extract(urls=[url], extract_depth="advanced")
        
        for result in response.get("results", []):
            if result.get("url") == url:
                content = result.get("raw_content", "")
                
                # Store in ChromaDB for evidence
                if content:
                    chunk_ids = chunk_and_store(
                        raw_content=content,
                        source_url=url,
                        query="page extraction",
                        page_title=result.get("title", "")
                    )
                    print(f"[tools] Stored {len(chunk_ids)} chunks from {url}")
                
                if len(content) > 15000:
                    content = content[:15000] + "\n\n[Content truncated...]"
                return content
        
        return "No content extracted from URL"
    
    except Exception as e:
        return f"Extraction failed: {str(e)}"


@tool
def extract_multiple_pages(urls: str) -> str:
    """
    Extract content from multiple URLs in batch.
    
    Use this when you have several promising URLs from search results
    and want to extract them all at once.
    
    Args:
        urls: JSON array of URLs to extract (max 10)
    
    Returns:
        Extracted content from all URLs with metadata.
    """
    try:
        url_list = json.loads(urls)
        if not isinstance(url_list, list):
            url_list = [url_list]
    except:
        url_list = [urls]
    
    url_list = url_list[:10]  # Limit to 10 URLs
    client = TavilyClient(api_key=get_tavily_api_key())
    
    results = []
    
    for url in url_list:
        try:
            response = client.extract(urls=[url], extract_depth="advanced")
            
            for result in response.get("results", []):
                content = result.get("raw_content", "")
                
                # Store in ChromaDB
                chunk_ids = []
                if content:
                    chunk_ids = chunk_and_store(
                        raw_content=content,
                        source_url=url,
                        query="batch extraction"
                    )
                
                results.append({
                    "url": url,
                    "content_length": len(content),
                    "content_preview": content[:2000] if content else "",
                    "chunks_stored": len(chunk_ids),
                    "success": bool(content)
                })
        except Exception as e:
            results.append({
                "url": url,
                "success": False,
                "error": str(e)
            })
    
    return json.dumps({
        "urls_processed": len(url_list),
        "successful": sum(1 for r in results if r.get("success")),
        "results": results
    }, indent=2)


@tool
def extract_with_link_following(url: str, max_depth: int = 1) -> str:
    """
    Extract content from a URL and follow relevant links on the page.
    
    This recursive extraction helps capture related product pages,
    datasheets, and specification pages linked from the main page.
    
    Args:
        url: The starting URL
        max_depth: How many levels of links to follow (1-2)
    
    Returns:
        Content from the main page and any relevant linked pages.
    """
    client = TavilyClient(api_key=get_tavily_api_key())
    max_depth = min(max_depth, 2)  # Safety limit
    
    results = []
    visited = set()
    to_visit = [(url, 0)]  # (url, depth)
    
    # Keywords that indicate relevant pages to follow
    relevant_keywords = [
        "specification", "datasheet", "technical", "data sheet",
        "product", "pressure", "transmitter", "catalog",
        "price", "order", "quote", "detail"
    ]
    
    while to_visit and len(results) < 8:  # Limit total pages
        current_url, depth = to_visit.pop(0)
        
        if current_url in visited:
            continue
        visited.add(current_url)
        
        try:
            response = client.extract(urls=[current_url], extract_depth="advanced")
            
            for result in response.get("results", []):
                content = result.get("raw_content", "")
                
                if content:
                    # Store in ChromaDB
                    chunk_ids = chunk_and_store(
                        raw_content=content,
                        source_url=current_url,
                        query="recursive extraction"
                    )
                    
                    results.append({
                        "url": current_url,
                        "depth": depth,
                        "content_length": len(content),
                        "content": content[:3000],  # Include more content
                        "chunks_stored": len(chunk_ids),
                    })
                    
                    # Extract and queue relevant links (if not at max depth)
                    if depth < max_depth:
                        link_pattern = r'href=["\']([^"\']+)["\']'
                        links = re.findall(link_pattern, content, re.IGNORECASE)
                        
                        base_domain = urlparse(current_url).netloc
                        
                        for link in links[:20]:  # Check up to 20 links
                            # Resolve relative URLs
                            full_link = urljoin(current_url, link)
                            link_domain = urlparse(full_link).netloc
                            
                            # Only follow same-domain links with relevant keywords
                            if link_domain == base_domain:
                                link_lower = full_link.lower()
                                if any(kw in link_lower for kw in relevant_keywords):
                                    if full_link not in visited:
                                        to_visit.append((full_link, depth + 1))
        except Exception as e:
            results.append({
                "url": current_url,
                "depth": depth,
                "error": str(e)
            })
    
    return json.dumps({
        "starting_url": url,
        "pages_extracted": len(results),
        "results": results
    }, indent=2)


# =============================================================================
# TOOL: ENHANCED SPECIFICATION EXTRACTION
# =============================================================================

@tool
def extract_product_specs(product_name: str, company_name: str, page_content: str) -> str:
    """
    Extract structured specifications from page content using the enhanced ontology.
    
    This tool:
    1. Uses aggressive extraction to capture ALL specs
    2. Maps specs to ontology using fuzzy matching
    3. Normalizes units (psi/bar/kPa, °C/°F, etc.)
    4. Tags unmapped specs as AI_DERIVED_ATTRIBUTE
    
    Args:
        product_name: The exact product model name (e.g., "ST800", "A-10")
        company_name: The manufacturer (e.g., "Honeywell", "Wika")
        page_content: The raw text content containing specifications
    
    Returns:
        JSON with extracted, normalized, and categorized specifications.
    """
    llm = ChatOpenAI(
        api_key=get_openai_api_key(),
        model="gpt-4o-mini",
        temperature=0,
    )
    
    ontology_prompt = get_ontology_for_prompt()
    aggressive_prompt = get_aggressive_extraction_prompt()
    
    prompt = f"""You are a technical specification extraction expert.

PRODUCT: {product_name} by {company_name}

{ontology_prompt}

{aggressive_prompt}

=== PAGE CONTENT ===
{page_content[:12000]}

=== YOUR TASK ===
Extract ALL specifications you can find for {product_name} from the content above.

Return a JSON object with TWO sections:

1. "ontology_specs" - specs that match the known ontology (use exact ontology keys):
{{
    "pressure_range": {{"raw_value": "0 to 6000 psi", "normalized_value": "0-6000", "unit": "psi", "raw_text": "...", "confidence": 0.95}},
    "accuracy": {{"raw_value": "±0.075%", "normalized_value": 0.075, "unit": "percent_fs", "raw_text": "...", "confidence": 0.9}},
    ...
}}

2. "other_specifications" - specs NOT in the ontology but still valuable:
{{
    "display_type": {{"value": "LCD", "raw_text": "Features LCD display", "confidence": 0.8}},
    "update_rate": {{"value": "20 Hz", "raw_text": "Update rate: 20 Hz", "confidence": 0.9}},
    ...
}}

IMPORTANT:
- Extract EVERY spec you can find, even partial ones
- Include raw_text snippets as evidence
- Convert units when possible (psi to bar, etc.)
- Estimate confidence (0-1) based on clarity of the source
- If a spec appears multiple times with different values, include the most reliable one

Return valid JSON only, no markdown code blocks."""

    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        
        # Clean up response
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"```json?\s*", "", content)
            content = content.replace("```", "")
        
        # Try to parse as JSON
        try:
            parsed = json.loads(content)
            
            # Process ontology specs - normalize and validate
            ontology_specs = parsed.get("ontology_specs", {})
            processed_specs = {}
            
            for key, spec_data in ontology_specs.items():
                # Validate against ontology
                if key in PRESSURE_TRANSMITTER_ONTOLOGY:
                    processed_specs[key] = spec_data
                else:
                    # Try fuzzy matching
                    match_key, score = find_best_ontology_match(key)
                    if match_key and score >= 0.6:
                        spec_data["fuzzy_match_score"] = score
                        spec_data["original_key"] = key
                        processed_specs[match_key] = spec_data
                    else:
                        # Add to other specs
                        if "other_specifications" not in parsed:
                            parsed["other_specifications"] = {}
                        parsed["other_specifications"][key] = spec_data
            
            # Process other specs - register as AI-derived
            other_specs = parsed.get("other_specifications", {})
            for name, data in other_specs.items():
                value = data.get("value", data.get("raw_value", ""))
                register_ai_derived_attribute(
                    name=name,
                    value=str(value),
                    source_url=f"extraction:{product_name}"
                )
            
            return json.dumps({
                "product": product_name,
                "company": company_name,
                "ontology_specs": processed_specs,
                "other_specifications": other_specs,
                "total_specs_found": len(processed_specs) + len(other_specs),
                "extraction_status": "success"
            }, indent=2)
            
        except json.JSONDecodeError:
            return json.dumps({
                "product": product_name,
                "company": company_name,
                "ontology_specs": {},
                "other_specifications": {},
                "extraction_status": "failed_to_parse",
                "raw_response": content[:1000]
            }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "product": product_name,
            "company": company_name,
            "ontology_specs": {},
            "other_specifications": {},
            "extraction_status": f"error: {str(e)}"
        }, indent=2)


@tool
def extract_all_specs_from_page(page_content: str, source_url: str = "") -> str:
    """
    Extract ALL possible specifications from a page without knowing the product.
    
    Use this for pages where you want to capture everything before
    knowing which product the specs belong to.
    
    Args:
        page_content: The raw text content to analyze
        source_url: The URL for evidence tracking
    
    Returns:
        JSON with all extracted specifications and potential product names.
    """
    llm = ChatOpenAI(
        api_key=get_openai_api_key(),
        model="gpt-4o-mini",
        temperature=0,
    )
    
    prompt = f"""You are a technical specification extraction expert.

Analyze this page content and extract:
1. Any product names/models mentioned
2. Any company names mentioned
3. ALL technical specifications you can find

=== PAGE CONTENT ===
{page_content[:10000]}

Return JSON:
{{
    "products_found": ["product1", "product2"],
    "companies_found": ["company1", "company2"],
    "specifications": {{
        "spec_name": {{
            "value": "value",
            "unit": "unit if applicable",
            "applies_to": "product name if specific",
            "raw_text": "source quote",
            "confidence": 0.0-1.0
        }}
    }},
    "prices_found": [
        {{"product": "name", "price": "$X", "raw_text": "quote"}}
    ]
}}

Be thorough - extract EVERY technical value you see."""

    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"```json?\s*", "", content)
            content = content.replace("```", "")
        
        parsed = json.loads(content)
        parsed["source_url"] = source_url
        parsed["extraction_status"] = "success"
        
        return json.dumps(parsed, indent=2)
    
    except Exception as e:
        return json.dumps({
            "extraction_status": f"error: {str(e)}",
            "source_url": source_url
        }, indent=2)


# =============================================================================
# TOOL: UNIT CONVERSION
# =============================================================================

@tool
def convert_pressure_unit(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert pressure values between units.
    
    Supported units: psi, bar, mbar, kPa, MPa, Pa, atm, mmHg, inHg
    
    Args:
        value: The numeric value to convert
        from_unit: Source unit (e.g., "bar")
        to_unit: Target unit (e.g., "psi")
    
    Returns:
        JSON with original and converted values.
    """
    from_lower = from_unit.lower().replace(" ", "")
    to_lower = to_unit.lower().replace(" ", "")
    
    # Convert to PSI first (canonical)
    if from_lower not in PRESSURE_TO_PSI:
        return json.dumps({"error": f"Unknown unit: {from_unit}"})
    
    psi_value = value * PRESSURE_TO_PSI[from_lower]
    
    # Convert from PSI to target
    if to_lower not in PRESSURE_TO_PSI:
        return json.dumps({"error": f"Unknown unit: {to_unit}"})
    
    result = psi_value / PRESSURE_TO_PSI[to_lower]
    
    return json.dumps({
        "original": {"value": value, "unit": from_unit},
        "converted": {"value": round(result, 4), "unit": to_unit},
        "psi_equivalent": round(psi_value, 4)
    }, indent=2)


# =============================================================================
# TOOL: VERIFY INFORMATION
# =============================================================================

@tool
def verify_claim(claim: str, evidence_text: str) -> str:
    """
    Verify if a claim is supported by the evidence text.
    
    Use this tool to validate extracted information before adding it
    to the knowledge base. This prevents hallucinations and ensures
    data quality.
    
    Args:
        claim: The claim to verify (e.g., "Wika A-10 costs $161.09")
        evidence_text: The source text that should support this claim
    
    Returns:
        Verification result with confidence score.
    """
    llm = ChatOpenAI(
        api_key=get_openai_api_key(),
        model="gpt-4o-mini",
        temperature=0,
    )
    
    prompt = f"""You are a fact-checker. Verify if the claim is supported by the evidence.

CLAIM: {claim}

EVIDENCE TEXT:
{evidence_text[:3000]}

Analyze if the evidence DIRECTLY supports the claim.
Return JSON:
{{
    "supported": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "exact_quote": "the exact text that supports/contradicts" (if found)
}}"""

    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        
        try:
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r"```json?\s*", "", content)
                content = content.replace("```", "")
            return content
        except:
            return json.dumps({"supported": False, "confidence": 0, "reasoning": "Failed to parse"})
    
    except Exception as e:
        return json.dumps({"supported": False, "confidence": 0, "reasoning": str(e)})


# =============================================================================
# TOOL: ANALYZE COMPETITORS
# =============================================================================

@tool  
def analyze_competitive_landscape(current_competitors: str) -> str:
    """
    Analyze the current competitive landscape and suggest next steps.
    
    Use this tool to get strategic guidance on what data to collect next.
    The tool considers what competitors, products, and specs we already
    have and suggests the most valuable next actions.
    
    Args:
        current_competitors: JSON string of current competitor data
    
    Returns:
        Analysis with recommended next actions.
    """
    llm = ChatOpenAI(
        api_key=get_openai_api_key(),
        model="gpt-4o-mini",
        temperature=0.3,
    )
    
    prompt = f"""You are a competitive intelligence analyst for Honeywell pressure transmitters.

CURRENT DATA:
{current_competitors}

Analyze what we have and what's missing. Consider:
1. Do we have enough competitors? (Target: 5+ major competitors)
2. Do we have products for each competitor?
3. Do we have prices for comparison?
4. Do we have specifications for head-to-head analysis?
5. Do we have at least 20 specifications across products?
6. Do we have reviews for customer insight?

Major pressure transmitter competitors include:
{json.dumps(KNOWN_COMPETITORS[:10], indent=2)}

Return JSON:
{{
    "analysis": "What we have and what's missing",
    "completeness_score": 0-100,
    "competitors_missing": ["list of competitors not yet covered"],
    "data_gaps": ["what specific data is missing"],
    "priority_actions": [
        {{"action": "description", "target": "specific target", "reason": "why this matters"}}
    ],
    "recommended_search_queries": ["specific searches to run next"]
}}"""

    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        
        try:
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r"```json?\s*", "", content)
                content = content.replace("```", "")
            return content
        except:
            return json.dumps({"analysis": "Failed to analyze", "priority_actions": []})
    
    except Exception as e:
        return json.dumps({"analysis": str(e), "priority_actions": []})


# =============================================================================
# TOOL: EVIDENCE SEARCH
# =============================================================================

@tool
def search_evidence_store(query: str, n_results: int = 5) -> str:
    """
    Search ChromaDB for previously stored evidence.
    
    Use this to find supporting evidence for claims or to
    check if we already have information on a topic.
    
    Args:
        query: What to search for
        n_results: Number of results to return
    
    Returns:
        Matching evidence chunks with metadata.
    """
    results = query_evidence(query, n_results)
    
    return json.dumps({
        "query": query,
        "results_found": len(results),
        "results": [
            {
                "id": r["id"],
                "content": r["document"][:500],
                "source_url": r["metadata"].get("source_url", ""),
                "relevance": 1 - (r["distance"] or 0) if r["distance"] else None
            }
            for r in results
        ]
    }, indent=2)


# =============================================================================
# TOOL: SAVE TO KNOWLEDGE BASE
# =============================================================================

@tool
def save_to_knowledge_base(entity_type: str, entity_data: str, source_url: str) -> str:
    """
    Save verified information to the knowledge base.
    
    Use this tool ONLY after you have:
    1. Found information through search
    2. Extracted structured data
    3. Verified the data is accurate
    
    Args:
        entity_type: One of: "competitor", "product", "price", "specification"
        entity_data: JSON string with the entity details
        source_url: The URL where this information was found
    
    Returns:
        Confirmation of what was saved.
    """
    try:
        data = json.loads(entity_data)
        data["source_url"] = source_url
        data["entity_type"] = entity_type
        
        # This will be processed by the pipeline state
        return json.dumps({
            "status": "queued_for_save",
            "entity_type": entity_type,
            "data": data
        }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


# =============================================================================
# ALL TOOLS COLLECTION
# =============================================================================

ALL_TOOLS = [
    search_web,
    search_with_variations,
    search_competitor_products,
    extract_page_content,
    extract_multiple_pages,
    extract_with_link_following,
    extract_product_specs,
    extract_all_specs_from_page,
    convert_pressure_unit,
    verify_claim,
    analyze_competitive_landscape,
    search_evidence_store,
    save_to_knowledge_base,
]


def get_tools_description() -> str:
    """Get a formatted description of all available tools."""
    descriptions = []
    for tool in ALL_TOOLS:
        descriptions.append(f"- {tool.name}: {tool.description.split(chr(10))[0]}")
    return "\n".join(descriptions)


def reset_visited_urls():
    """Reset the visited URLs tracker (for new agent sessions)."""
    global VISITED_URLS
    VISITED_URLS = set()
