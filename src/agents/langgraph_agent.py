"""
LangGraph Agentic Pipeline - ENHANCED VERSION with aggressive data extraction.

Architecture:
    __start__ → agent → tools → agent (loop) → __end__
    
    The 'tools' node fans out to execute multiple tools in parallel,
    with individual tool nodes shown in the visualization.

ENHANCEMENTS:
- Higher extraction thresholds (5+ competitors, 8+ products, 20+ specs)
- Multi-step search with query variations
- Aggressive extraction prompts
- Unit normalization
- ChromaDB evidence storage
- Fallback loops to ensure requirements are met
- AI-derived attributes support
"""

from __future__ import annotations

import json
import operator
import time
import re
from typing import Annotated, Any, Dict, List, Sequence, TypedDict, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient

from src.config.settings import get_openai_api_key, get_tavily_api_key
from src.ontology.specifications import (
    find_best_ontology_match,
    normalize_spec_value,
    register_ai_derived_attribute,
    get_ai_derived_attributes,
    PRESSURE_TRANSMITTER_ONTOLOGY,
)
from src.pipeline.chroma_store import chunk_and_store


# =============================================================================
# CONFIGURATION - EXTRACTION THRESHOLDS WITH LIMITS
# =============================================================================

# Target requirements (ideal goals)
MIN_COMPETITORS = 5
MIN_PRODUCTS = 8
MIN_SPECIFICATIONS = 20
MIN_PRICES = 5
MIN_REVIEWS = 3

# HARD MINIMUM (can complete even if targets not met after max attempts)
HARD_MIN_COMPETITORS = 3
HARD_MIN_PRODUCTS = 4
HARD_MIN_SPECIFICATIONS = 10
HARD_MIN_PRICES = 2  # Prices are often not public
HARD_MIN_REVIEWS = 1  # Reviews are hard to find

# Maximum iterations and search limits
DEFAULT_MAX_ITERATIONS = 50
MAX_SEARCH_RESULTS = 10
SEARCH_VARIATIONS_PER_QUERY = 6

# Per-competitor/per-product attempt limits (prevent infinite loops)
MAX_SEARCHES_PER_COMPETITOR = 4  # Max search attempts per competitor
MAX_PRICE_SEARCH_ATTEMPTS = 3   # Max attempts to find price for a product
MAX_REVIEW_SEARCH_ATTEMPTS = 2  # Max attempts to find reviews for a product
MAX_SPEC_SEARCH_ATTEMPTS = 3    # Max attempts to get specs for a product

# Known competitors for targeted search
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
    "Dwyer",
    "Ashcroft",
    "VEGA",
]


# =============================================================================
# STATE DEFINITION - ENHANCED WITH ATTEMPT TRACKING
# =============================================================================

class AgentState(TypedDict):
    """State that flows through the LangGraph agent."""
    messages: Annotated[List[BaseMessage], operator.add]
    competitors: Dict[str, Dict]
    products: Dict[str, Dict]
    specifications: Dict[str, Dict]  # product -> {spec_name: {value, normalized_value, snippet, evidence_id, source_url}}
    prices: Dict[str, Dict]          # product -> {value, snippet, evidence_id, source_url}
    reviews: Dict[str, List[Dict]]   # product_name -> list of reviews
    ai_derived_specs: Dict[str, Dict]  # AI-discovered specs not in ontology
    sources: List[str]
    visited_urls: List[str]
    iteration: int
    max_iterations: int
    is_complete: bool
    search_queries_used: List[str]
    # Attempt tracking to prevent infinite loops
    competitor_search_attempts: Dict[str, int]  # competitor -> search count
    price_search_attempts: Dict[str, int]       # product -> price search count
    review_search_attempts: Dict[str, int]      # product -> review search count
    spec_search_attempts: Dict[str, int]        # product -> spec search count
    failed_searches: List[str]                  # queries that yielded no results


# =============================================================================
# TOOLS - ENHANCED
# =============================================================================

@tool
def search_web(query: str, max_results: int = 8) -> str:
    """Search the web for competitive intelligence. Returns up to 10 results with full content."""
    client = TavilyClient(api_key=get_tavily_api_key())
    try:
        response = client.search(
            query=query, 
            max_results=min(max_results, MAX_SEARCH_RESULTS), 
            search_depth="advanced",
            include_raw_content=True
        )
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:800],  # More content
                "raw_content_available": bool(r.get("raw_content")),
            })
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def search_with_variations(base_query: str) -> str:
    """
    Search using multiple query variations to find more diverse results.
    Automatically generates synonyms and reformulations.
    """
    client = TavilyClient(api_key=get_tavily_api_key())
    
    # Generate variations
    variations = [base_query]
    base_lower = base_query.lower()
    
    # Synonym substitutions
    synonyms = {
        "pressure transmitter": ["pressure transducer", "pressure sensor", "pressure gauge"],
        "specifications": ["specs", "datasheet", "technical data"],
        "price": ["pricing", "cost", "quote", "buy"],
    }
    
    for term, syns in synonyms.items():
        if term in base_lower:
            for syn in syns[:2]:
                variations.append(base_lower.replace(term, syn))
    
    # Add targeted variations
    variations.extend([
        f"{base_query} industrial",
        f"{base_query} datasheet PDF",
        f"{base_query} 4-20mA HART",
    ])
    
    # Deduplicate
    variations = list(dict.fromkeys(variations))[:SEARCH_VARIATIONS_PER_QUERY]
    
    all_results = []
    seen_urls = set()
    
    for query in variations:
        try:
            response = client.search(
                query=query,
                max_results=5,
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
                        "query_used": query,
                    })
        except:
            continue
    
    return json.dumps({
        "queries_used": variations,
        "total_results": len(all_results),
        "results": all_results[:15]
    }, indent=2)


@tool
def extract_page_content(url: str) -> str:
    """Extract full content from a URL and store in ChromaDB for evidence tracking."""
    client = TavilyClient(api_key=get_tavily_api_key())
    try:
        response = client.extract(urls=[url], extract_depth="advanced")
        results = response.get("results", [])
        if results:
            content = results[0].get("raw_content", "")
            title = results[0].get("title", "")
            
            # Store FULL content in ChromaDB for evidence
            if content:
                chunk_ids = chunk_and_store(
                    raw_content=content,
                    source_url=url,
                    query="agent extraction",
                    page_title=title
                )
                print(f"  [ChromaDB] Stored {len(chunk_ids)} chunks from {url[:50]}")
            
            # Return TRUNCATED content to agent to avoid context overflow
            # Full content is safely stored in ChromaDB
            if len(content) > 8000:
                content = content[:8000] + "\n\n[Content truncated - full version stored in ChromaDB]"
            return content
        return "No content found"
    except Exception as e:
        return f"Extraction error: {str(e)}"


@tool
def extract_multiple_pages(urls_json: str) -> str:
    """Extract content from multiple URLs in batch. Provide a JSON array of URLs."""
    try:
        urls = json.loads(urls_json)
        if not isinstance(urls, list):
            urls = [urls]
    except:
        urls = [urls_json]
    
    urls = urls[:8]  # Limit
    client = TavilyClient(api_key=get_tavily_api_key())
    results = []
    
    for url in urls:
        try:
            response = client.extract(urls=[url], extract_depth="advanced")
            for result in response.get("results", []):
                content = result.get("raw_content", "")
                if content:
                    chunk_ids = chunk_and_store(
                        raw_content=content,
                        source_url=url,
                        query="batch extraction"
                    )
                    results.append({
                        "url": url,
                        "content_preview": content[:2000],
                        "chunks_stored": len(chunk_ids)
                    })
        except Exception as e:
            results.append({"url": url, "error": str(e)})
    
    return json.dumps({"extracted": len(results), "results": results}, indent=2)


@tool
def save_competitor(name: str, source_url: str = "") -> str:
    """Save a competitor company to the knowledge base."""
    return json.dumps({"action": "save_competitor", "name": name, "source_url": source_url})


@tool
def save_product(product_name: str, company_name: str, source_url: str = "") -> str:
    """Save a product to the knowledge base."""
    return json.dumps({
        "action": "save_product", 
        "product_name": product_name, 
        "company_name": company_name, 
        "source_url": source_url
    })


@tool
def save_specification(
    product_name: str, 
    spec_name: str, 
    spec_value: str, 
    spec_unit: str = "",
    snippet: str = "", 
    evidence_id: str = "", 
    source_url: str = ""
) -> str:
    """
    Save a specification with unit normalization and evidence tracking.
    
    For pressure specs, the value will be normalized to PSI.
    For temperature specs, the value will be normalized to Celsius.
    """
    # Try to normalize the value
    normalized_value = spec_value
    try:
        # Check if this maps to an ontology spec
        match_key, score = find_best_ontology_match(spec_name)
        
        if match_key and score >= 0.6:
            spec_name = match_key  # Use canonical name
            
            # Try to parse and normalize numeric values
            num_match = re.search(r'([\d.]+)', spec_value)
            if num_match and spec_unit:
                try:
                    num_val = float(num_match.group(1))
                    normalized = normalize_spec_value(match_key, num_val, spec_unit)
                    normalized_value = str(normalized.normalized_value)
                except:
                    pass
        else:
            # Register as AI-derived attribute
            register_ai_derived_attribute(spec_name, spec_value, source_url)
            
    except Exception:
        pass
    
    return json.dumps({
        "action": "save_specification",
        "product_name": product_name,
        "spec_name": spec_name,
        "spec_value": spec_value,
        "normalized_value": normalized_value,
        "spec_unit": spec_unit,
        "snippet": snippet,
        "evidence_id": evidence_id,
        "source_url": source_url,
    })


@tool
def save_multiple_specs(product_name: str, specs_json: str, source_url: str = "") -> str:
    """
    Save multiple specifications at once for efficiency.
    
    specs_json should be a JSON object like:
    {"pressure_range": "0-6000 psi", "accuracy": "0.075%", "output": "4-20mA"}
    """
    try:
        specs = json.loads(specs_json)
        saved = []
        
        for spec_name, spec_value in specs.items():
            # Try to match to ontology
            match_key, score = find_best_ontology_match(spec_name)
            
            if match_key and score >= 0.6:
                saved.append({
                    "spec_name": match_key,
                    "spec_value": spec_value,
                    "matched": True,
                    "score": score
                })
            else:
                register_ai_derived_attribute(spec_name, str(spec_value), source_url)
                saved.append({
                    "spec_name": spec_name,
                    "spec_value": spec_value,
                    "matched": False,
                    "ai_derived": True
                })
        
        return json.dumps({
            "action": "save_multiple_specs",
            "product_name": product_name,
            "specs_saved": saved,
            "source_url": source_url,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def save_price(product_name: str, price: str, snippet: str = "", evidence_id: str = "", source_url: str = "") -> str:
    """Save a price with evidence tracking."""
    return json.dumps({
        "action": "save_price",
        "product_name": product_name,
        "price": price,
        "snippet": snippet,
        "evidence_id": evidence_id,
        "source_url": source_url,
    })


@tool
def save_review(
    product_name: str, 
    review_text: str, 
    rating: str = "", 
    source: str = "", 
    snippet: str = "", 
    evidence_id: str = "", 
    source_url: str = ""
) -> str:
    """Save a customer review for a product."""
    return json.dumps({
        "action": "save_review",
        "product_name": product_name,
        "review_text": review_text,
        "rating": rating,
        "source": source,
        "source_url": source_url or source,
        "snippet": snippet or review_text,
        "evidence_id": evidence_id,
    })


@tool
def get_search_suggestions(current_data: str) -> str:
    """
    Get intelligent search suggestions based on current data gaps.
    
    Analyzes what's missing and suggests specific search queries.
    """
    try:
        data = json.loads(current_data)
    except:
        data = {}
    
    competitors = data.get("competitors", {})
    products = data.get("products", {})
    
    suggestions = []
    
    # Check for missing competitors
    covered_competitors = set(c.lower() for c in competitors.keys())
    for comp in KNOWN_COMPETITORS:
        if comp.lower() not in covered_competitors:
            suggestions.append({
                "query": f"{comp} pressure transmitter specifications",
                "reason": f"Missing competitor: {comp}"
            })
    
    # Check for products needing more specs
    specs = data.get("specifications", {})
    for product in products:
        product_specs = specs.get(product, {})
        if len(product_specs) < 4:
            suggestions.append({
                "query": f"{product} technical specifications datasheet",
                "reason": f"Product {product} has only {len(product_specs)} specs"
            })
    
    # Check for missing prices
    prices = data.get("prices", {})
    for product in products:
        if product not in prices:
            company = products[product].get("company", "")
            suggestions.append({
                "query": f"{product} {company} price buy",
                "reason": f"Missing price for {product}"
            })
    
    return json.dumps({
        "suggestions": suggestions[:10],
        "missing_competitors": [c for c in KNOWN_COMPETITORS if c.lower() not in covered_competitors][:5]
    }, indent=2)


@tool
def mark_complete(summary: str) -> str:
    """Mark the mission as complete. Only call when ALL thresholds are met."""
    return json.dumps({"action": "complete", "summary": summary})


# All tools available to the agent
TOOLS = [
    search_web,
    search_with_variations,
    extract_page_content,
    extract_multiple_pages,
    save_competitor,
    save_product,
    save_specification,
    save_multiple_specs,
    save_price,
    save_review,
    get_search_suggestions,
    mark_complete,
]

TOOL_MAP = {t.name: t for t in TOOLS}


# =============================================================================
# AGENT NODE - ENHANCED PROMPTS
# =============================================================================

def create_agent_node(llm_with_tools):
    """Create the agent node with enhanced extraction prompts."""
    
    system_prompt = f"""You are an AGGRESSIVE Competitive Intelligence Agent for Honeywell pressure transmitters.

YOUR MISSION: Build a COMPREHENSIVE competitive analysis database.

=== MINIMUM REQUIREMENTS (MUST achieve before mark_complete) ===
- Competitors: {MIN_COMPETITORS}+ (not counting Honeywell)
- Products: {MIN_PRODUCTS}+ (with company associations)
- Specifications: {MIN_SPECIFICATIONS}+ total across all products
- Prices: {MIN_PRICES}+ (or "Contact for quote" entries)
- Reviews: {MIN_REVIEWS}+ customer reviews

=== BASELINE FIRST ===
1. Save Honeywell as the baseline company
2. Save SmartLine ST700 as the baseline product
3. Extract ALL specifications for ST700 (search datasheets)
4. Find pricing and reviews for ST700

=== COMPETITOR STRATEGY ===
Known competitors to investigate:
{json.dumps(KNOWN_COMPETITORS[:8], indent=2)}

For EACH competitor:
1. save_competitor - save the company
2. Search for their pressure transmitter products
3. save_product - save each product found
4. extract_page_content - get full datasheets/spec pages
5. save_specification - save EVERY spec you find (use save_multiple_specs for efficiency)
6. save_price - get pricing or "Contact for quote"
7. save_review - search for reviews

=== SEARCH STRATEGY ===
- Use search_with_variations for broader results
- Extract content from EVERY promising URL
- Follow datasheet links
- Search multiple variations: "[product] datasheet", "[product] specifications PDF", "[company] pressure transmitter price"

=== SPECIFICATION EXTRACTION RULES ===
Extract EVERY technical specification you find:
- Pressure range (e.g., "0-6000 psi")
- Accuracy (e.g., "±0.075%")
- Output signal (e.g., "4-20mA HART")
- Process connection (e.g., "1/2 NPT")
- Operating temperature
- IP rating
- Certifications (ATEX, IECEx, SIL)
- Materials (wetted parts, housing)
- Response time
- Supply voltage
- ANY other technical specifications

Use save_multiple_specs to save multiple specs at once for efficiency.
Include snippet (quote from source) and source_url for each spec.

=== UNIT HANDLING ===
The system automatically normalizes units:
- Pressure: converts to PSI (bar, kPa, MPa accepted)
- Temperature: converts to Celsius (°F accepted)
Include the original unit when saving specs.

=== EVIDENCE RULES ===
For EVERY saved item include:
- snippet: exact quote from source (max 100 chars)
- source_url: the URL where found
- evidence_id: chunk ID if available

=== ITERATION STRATEGY ===
- Keep searching until targets are met OR you've exhausted reasonable attempts
- If stuck, use get_search_suggestions for ideas
- Try different search variations
- Don't repeat the same search twice
- Extract content from all promising URLs
- If a price/review cannot be found after 2-3 attempts, move on
- Maximum {MAX_SEARCHES_PER_COMPETITOR} search attempts per competitor

=== GRACEFUL COMPLETION ===
Target thresholds: {MIN_COMPETITORS} competitors, {MIN_PRODUCTS} products, {MIN_SPECIFICATIONS} specs, {MIN_PRICES} prices, {MIN_REVIEWS} reviews
Hard minimums (can complete even if targets not met): {HARD_MIN_COMPETITORS} competitors, {HARD_MIN_PRODUCTS} products, {HARD_MIN_SPECIFICATIONS} specs, {HARD_MIN_PRICES} prices, {HARD_MIN_REVIEWS} reviews

If you've tried multiple times and can't find prices/reviews, save "Contact for quote" or "No public reviews found" and move on.
Call mark_complete when targets are met OR when hard minimums are met and you've exhausted search attempts."""

    def agent_node(state: AgentState) -> Dict:
        messages = list(state.get("messages", []))
        
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        
        # CONTEXT MANAGEMENT: Keep only recent messages to avoid token limit
        # Keep system prompt + last 20 messages (10 agent/tool pairs)
        MAX_MESSAGES = 20
        if len(messages) > MAX_MESSAGES + 1:  # +1 for system prompt
            system_msg = messages[0] if isinstance(messages[0], SystemMessage) else None
            recent_messages = messages[-(MAX_MESSAGES):]
            if system_msg:
                messages = [system_msg] + recent_messages
            else:
                messages = recent_messages
        
        competitors = state.get("competitors", {})
        products = state.get("products", {})
        specs = state.get("specifications", {})
        total_specs = sum(len(s) for s in specs.values())
        prices = state.get("prices", {})
        reviews = state.get("reviews", {})
        total_reviews = sum(len(r) for r in reviews.values())
        ai_derived = state.get("ai_derived_specs", {})
        
        # Get attempt tracking
        price_attempts = state.get("price_search_attempts", {})
        review_attempts = state.get("review_search_attempts", {})
        failed_searches = state.get("failed_searches", [])
        
        iteration = state.get('iteration', 0)
        max_iters = state.get('max_iterations', DEFAULT_MAX_ITERATIONS)
        remaining_iters = max_iters - iteration
        
        # Check if all TARGET thresholds are met
        targets_met = (
            len(competitors) >= MIN_COMPETITORS and 
            len(products) >= MIN_PRODUCTS and 
            total_specs >= MIN_SPECIFICATIONS and 
            len(prices) >= MIN_PRICES and 
            total_reviews >= MIN_REVIEWS
        )
        
        # Check if HARD MINIMUM thresholds are met (graceful completion)
        hard_mins_met = (
            len(competitors) >= HARD_MIN_COMPETITORS and 
            len(products) >= HARD_MIN_PRODUCTS and 
            total_specs >= HARD_MIN_SPECIFICATIONS and 
            len(prices) >= HARD_MIN_PRICES and 
            total_reviews >= HARD_MIN_REVIEWS
        )
        
        # Calculate products without prices/reviews that haven't been tried too many times
        products_needing_price = [
            p for p in products 
            if p not in prices and price_attempts.get(p, 0) < MAX_PRICE_SEARCH_ATTEMPTS
        ]
        products_needing_review = [
            p for p in products 
            if p not in [r for r in reviews if reviews.get(r)] and review_attempts.get(p, 0) < MAX_REVIEW_SEARCH_ATTEMPTS
        ]
        
        # Check if we've exhausted search attempts (allow graceful completion)
        exhausted_attempts = (
            hard_mins_met and 
            len(products_needing_price) == 0 and 
            len(products_needing_review) == 0 and
            remaining_iters < 10
        )
        
        can_complete = targets_met or exhausted_attempts
        
        # Calculate what's still needed
        needs = []
        if len(competitors) < MIN_COMPETITORS:
            needs.append(f"competitors: need {MIN_COMPETITORS - len(competitors)} more (hard min: {HARD_MIN_COMPETITORS})")
        if len(products) < MIN_PRODUCTS:
            needs.append(f"products: need {MIN_PRODUCTS - len(products)} more (hard min: {HARD_MIN_PRODUCTS})")
        if total_specs < MIN_SPECIFICATIONS:
            needs.append(f"specs: need {MIN_SPECIFICATIONS - total_specs} more (hard min: {HARD_MIN_SPECIFICATIONS})")
        if len(prices) < MIN_PRICES:
            needs.append(f"prices: need {MIN_PRICES - len(prices)} more (hard min: {HARD_MIN_PRICES})")
        if total_reviews < MIN_REVIEWS:
            needs.append(f"reviews: need {MIN_REVIEWS - total_reviews} more (hard min: {HARD_MIN_REVIEWS})")
        
        # Suggest next actions if not complete
        suggestions = ""
        if not targets_met:
            missing_competitors = [c for c in KNOWN_COMPETITORS if c.lower() not in [comp.lower() for comp in competitors.keys()]]
            if missing_competitors and len(competitors) < MIN_COMPETITORS:
                suggestions = f"\n\nSUGGESTED SEARCHES:\n- Try searching for: {', '.join(missing_competitors[:3])}"
            if total_specs < MIN_SPECIFICATIONS:
                products_needing_specs = [p for p in products if len(specs.get(p, {})) < 4]
                if products_needing_specs:
                    suggestions += f"\n- Products needing more specs: {', '.join(products_needing_specs[:3])}"
            if products_needing_price:
                suggestions += f"\n- Products needing price (attempts left): {', '.join(products_needing_price[:3])}"
            if products_needing_review:
                suggestions += f"\n- Products needing review (attempts left): {', '.join(products_needing_review[:3])}"
        
        # Status message
        if targets_met:
            status = "✅ ALL TARGET THRESHOLDS MET - You may call mark_complete"
        elif exhausted_attempts:
            status = "⚠️ HARD MINIMUMS MET & SEARCH ATTEMPTS EXHAUSTED - You may call mark_complete with what you have"
        else:
            status = f"❌ NOT COMPLETE - Still need: {', '.join(needs)}"
        
        context = f"""
=== CURRENT PROGRESS ===
Competitors: {len(competitors)}/{MIN_COMPETITORS} (hard min: {HARD_MIN_COMPETITORS}) {'✓' if len(competitors) >= MIN_COMPETITORS else ''}
  - Saved: {list(competitors.keys())[:8]}

Products: {len(products)}/{MIN_PRODUCTS} (hard min: {HARD_MIN_PRODUCTS}) {'✓' if len(products) >= MIN_PRODUCTS else ''}
  - Saved: {list(products.keys())[:8]}

Specifications: {total_specs}/{MIN_SPECIFICATIONS} (hard min: {HARD_MIN_SPECIFICATIONS}) {'✓' if total_specs >= MIN_SPECIFICATIONS else ''}
  - By product: {json.dumps({k: len(v) for k, v in list(specs.items())[:5]}, indent=2)}

Prices: {len(prices)}/{MIN_PRICES} (hard min: {HARD_MIN_PRICES}) {'✓' if len(prices) >= MIN_PRICES else ''}
  - Products with price: {list(prices.keys())[:5]}
  - Products needing price (attempts remaining): {products_needing_price[:3] if products_needing_price else 'None'}

Reviews: {total_reviews}/{MIN_REVIEWS} (hard min: {HARD_MIN_REVIEWS}) {'✓' if total_reviews >= MIN_REVIEWS else ''}
  - Products needing review (attempts remaining): {products_needing_review[:3] if products_needing_review else 'None'}

AI-Derived Specs: {len(ai_derived)} (specs not in standard ontology)

Iteration: {iteration}/{max_iters} ({remaining_iters} remaining)

=== STATUS ===
{status}
{suggestions}

{'TIP: If prices/reviews cannot be found, save "Contact for quote" or "No public reviews found" and continue.' if not targets_met and hard_mins_met else ''}"""
        
        messages.append(HumanMessage(content=context))
        
        # Rate limit protection - significant delay to avoid 429 errors
        time.sleep(4.0)  # 4 second delay between API calls
        
        # Try with exponential backoff on rate limit errors
        max_retries = 3
        response = None
        for retry in range(max_retries):
            try:
                response = llm_with_tools.invoke(messages)
                break  # Success, exit retry loop
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate limit" in error_str.lower():
                    wait_time = (retry + 1) * 30  # 30s, 60s, 90s
                    print(f"\n⏳ Rate limit hit. Waiting {wait_time}s before retry {retry + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    if retry == max_retries - 1:
                        print("❌ Rate limit retries exhausted. Stopping gracefully.")
                        # Return what we have so far
                        return {
                            "messages": messages,
                            "competitors": state.get("competitors", {}),
                            "products": state.get("products", {}),
                            "specifications": state.get("specifications", {}),
                            "prices": state.get("prices", {}),
                            "reviews": state.get("reviews", {}),
                            "iteration": iteration,
                            "should_continue": False,
                        }
                else:
                    raise e  # Re-raise non-rate-limit errors
        
        if response is None:
            # Fallback if all retries failed
            return {
                "messages": messages,
                "competitors": state.get("competitors", {}),
                "products": state.get("products", {}),
                "specifications": state.get("specifications", {}),
                "prices": state.get("prices", {}),
                "reviews": state.get("reviews", {}),
                "iteration": iteration,
                "should_continue": False,
            }
        
        return {
            "messages": [response],
            "iteration": iteration + 1,
        }
    
    return agent_node


# =============================================================================
# TOOL NODES - ENHANCED STATE HANDLING
# =============================================================================

def create_tool_node(tool_name: str):
    """Create a node for a specific tool with enhanced state handling."""
    tool_fn = TOOL_MAP[tool_name]
    
    def node(state: AgentState) -> Dict:
        messages = state.get("messages", [])
        if not messages:
            return {"messages": [], "competitors": {}, "products": {}, "specifications": {}, "prices": {}, "reviews": {}, "ai_derived_specs": {}}
        
        last_msg = messages[-1]
        if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
            return {"messages": [], "competitors": {}, "products": {}, "specifications": {}, "prices": {}, "reviews": {}, "ai_derived_specs": {}}
        
        # Find calls for this specific tool
        calls = [tc for tc in last_msg.tool_calls if tc["name"] == tool_name]
        if not calls:
            return {"messages": [], "competitors": {}, "products": {}, "specifications": {}, "prices": {}, "reviews": {}, "ai_derived_specs": {}}
        
        results = []
        new_competitors = {}
        new_products = {}
        new_specs = {}
        new_prices = {}
        new_reviews = {}
        new_ai_derived = {}
        is_complete = state.get("is_complete", False)
        
        for tc in calls:
            try:
                result = tool_fn.invoke(tc["args"])
                results.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                
                # Parse and update state
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, dict):
                        action = parsed.get("action", "")
                        
                        if action == "save_competitor":
                            name = parsed.get("name", "")
                            if name:
                                new_competitors[name] = {
                                    "name": name, 
                                    "source_url": parsed.get("source_url", "")
                                }
                                print(f"  ✓ Saved competitor: {name}")
                        
                        elif action == "save_product":
                            name = parsed.get("product_name", "")
                            company = parsed.get("company_name", "")
                            if name:
                                new_products[name] = {
                                    "name": name, 
                                    "company": company,
                                    "source_url": parsed.get("source_url", "")
                                }
                                print(f"  ✓ Saved product: {name} by {company}")
                        
                        elif action == "save_specification":
                            product = parsed.get("product_name", "")
                            spec_name = parsed.get("spec_name", "")
                            spec_value = parsed.get("spec_value", "")
                            normalized_value = parsed.get("normalized_value", spec_value)
                            spec_unit = parsed.get("spec_unit", "")
                            snippet = parsed.get("snippet", "")
                            evidence_id = parsed.get("evidence_id", "")
                            source_url = parsed.get("source_url", "")
                            
                            if product and spec_name:
                                if product not in new_specs:
                                    new_specs[product] = {}
                                new_specs[product][spec_name] = {
                                    "value": spec_value,
                                    "normalized_value": normalized_value,
                                    "unit": spec_unit,
                                    "snippet": snippet,
                                    "evidence_id": evidence_id,
                                    "source_url": source_url,
                                }
                                print(f"  ✓ Saved spec: {product} → {spec_name} = {spec_value}")
                        
                        elif action == "save_multiple_specs":
                            product = parsed.get("product_name", "")
                            specs_saved = parsed.get("specs_saved", [])
                            source_url = parsed.get("source_url", "")
                            
                            if product and specs_saved:
                                if product not in new_specs:
                                    new_specs[product] = {}
                                
                                for spec in specs_saved:
                                    spec_name = spec.get("spec_name", "")
                                    spec_value = spec.get("spec_value", "")
                                    
                                    if spec_name:
                                        new_specs[product][spec_name] = {
                                            "value": spec_value,
                                            "source_url": source_url,
                                            "matched": spec.get("matched", False),
                                            "ai_derived": spec.get("ai_derived", False),
                                        }
                                        
                                        if spec.get("ai_derived"):
                                            new_ai_derived[spec_name] = {
                                                "value": spec_value,
                                                "source_url": source_url
                                            }
                                
                                print(f"  ✓ Saved {len(specs_saved)} specs for {product}")
                        
                        elif action == "save_price":
                            product = parsed.get("product_name", "")
                            price = parsed.get("price", "")
                            snippet = parsed.get("snippet", "")
                            evidence_id = parsed.get("evidence_id", "")
                            source_url = parsed.get("source_url", "")
                            
                            if product and price:
                                new_prices[product] = {
                                    "value": price,
                                    "snippet": snippet,
                                    "evidence_id": evidence_id,
                                    "source_url": source_url,
                                }
                                print(f"  ✓ Saved price: {product} = {price}")
                        
                        elif action == "save_review":
                            product = parsed.get("product_name", "")
                            review_text = parsed.get("review_text", "")
                            rating = parsed.get("rating", "")
                            source = parsed.get("source", "")
                            snippet = parsed.get("snippet", "")
                            evidence_id = parsed.get("evidence_id", "")
                            source_url = parsed.get("source_url", "")
                            
                            if product and review_text:
                                if product not in new_reviews:
                                    new_reviews[product] = []
                                new_reviews[product].append({
                                    "text": review_text,
                                    "rating": rating,
                                    "source": source,
                                    "source_url": source_url,
                                    "snippet": snippet or review_text[:100],
                                    "evidence_id": evidence_id,
                                })
                                print(f"  ✓ Saved review: {product} - {rating}")
                        
                        elif action == "complete":
                            is_complete = True
                            print(f"  🏁 Mission complete!")
                except:
                    pass
                    
            except Exception as e:
                results.append(ToolMessage(content=f"Error: {str(e)}", tool_call_id=tc["id"]))
        
        return {
            "messages": results,
            "competitors": new_competitors,
            "products": new_products,
            "specifications": new_specs,
            "prices": new_prices,
            "reviews": new_reviews,
            "ai_derived_specs": new_ai_derived,
            "is_complete": is_complete,
        }
    
    return node


# =============================================================================
# ROUTER
# =============================================================================

def route_after_agent(state: AgentState) -> str:
    """Route from agent to appropriate tool or end."""
    if state.get("is_complete", False):
        return "end"
    
    if state.get("iteration", 0) >= state.get("max_iterations", DEFAULT_MAX_ITERATIONS):
        return "end"
    
    messages = state.get("messages", [])
    if not messages:
        return "end"
    
    last_msg = messages[-1]
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return "end"
    
    # Route to the first tool called
    tool_name = last_msg.tool_calls[0]["name"]
    return tool_name


# =============================================================================
# BUILD GRAPH
# =============================================================================

def build_agentic_graph(max_iterations: int = DEFAULT_MAX_ITERATIONS):
    """Build the enhanced LangGraph with separate tool nodes."""
    
    llm = ChatOpenAI(
        api_key=get_openai_api_key(),
        model="gpt-4o",
        temperature=0.1,
    )
    llm_with_tools = llm.bind_tools(TOOLS, parallel_tool_calls=False)
    
    graph = StateGraph(AgentState)
    
    # Add agent node
    graph.add_node("agent", create_agent_node(llm_with_tools))
    
    # Add individual tool nodes
    graph.add_node("search_web", create_tool_node("search_web"))
    graph.add_node("search_variations", create_tool_node("search_with_variations"))
    graph.add_node("extract_page", create_tool_node("extract_page_content"))
    graph.add_node("extract_multiple", create_tool_node("extract_multiple_pages"))
    graph.add_node("save_competitor", create_tool_node("save_competitor"))
    graph.add_node("save_product", create_tool_node("save_product"))
    graph.add_node("save_spec", create_tool_node("save_specification"))
    graph.add_node("save_multi_specs", create_tool_node("save_multiple_specs"))
    graph.add_node("save_price", create_tool_node("save_price"))
    graph.add_node("save_review", create_tool_node("save_review"))
    graph.add_node("get_suggestions", create_tool_node("get_search_suggestions"))
    graph.add_node("mark_complete", create_tool_node("mark_complete"))
    
    # Edges
    graph.add_edge(START, "agent")
    
    # Agent routes to specific tool
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "search_web": "search_web",
            "search_with_variations": "search_variations",
            "extract_page_content": "extract_page",
            "extract_multiple_pages": "extract_multiple",
            "save_competitor": "save_competitor",
            "save_product": "save_product",
            "save_specification": "save_spec",
            "save_multiple_specs": "save_multi_specs",
            "save_price": "save_price",
            "save_review": "save_review",
            "get_search_suggestions": "get_suggestions",
            "mark_complete": "mark_complete",
            "end": END,
        }
    )
    
    # All tools go back to agent
    for tool_node in [
        "search_web", "search_variations", "extract_page", "extract_multiple",
        "save_competitor", "save_product", "save_spec", "save_multi_specs",
        "save_price", "save_review", "get_suggestions", "mark_complete"
    ]:
        graph.add_edge(tool_node, "agent")
    
    return graph.compile()


# =============================================================================
# RUN AGENT
# =============================================================================

def run_langgraph_agent(max_iterations: int = DEFAULT_MAX_ITERATIONS, max_competitors: int = 5) -> Dict[str, Any]:
    """Run the enhanced LangGraph agent with aggressive extraction."""
    
    print("="*70)
    print("🤖 LANGGRAPH AGENTIC PIPELINE - ENHANCED EXTRACTION")
    print(f"Max iterations: {max_iterations}")
    print(f"Target thresholds: {MIN_COMPETITORS} competitors, {MIN_PRODUCTS} products, {MIN_SPECIFICATIONS} specs, {MIN_PRICES} prices, {MIN_REVIEWS} reviews")
    print(f"Hard minimums: {HARD_MIN_COMPETITORS} competitors, {HARD_MIN_PRODUCTS} products, {HARD_MIN_SPECIFICATIONS} specs, {HARD_MIN_PRICES} prices, {HARD_MIN_REVIEWS} reviews")
    print(f"Per-item limits: {MAX_SEARCHES_PER_COMPETITOR} searches/competitor, {MAX_PRICE_SEARCH_ATTEMPTS} price attempts, {MAX_REVIEW_SEARCH_ATTEMPTS} review attempts")
    print("="*70)
    
    app = build_agentic_graph(max_iterations)
    
    initial_state = {
        "messages": [],
        "competitors": {},
        "products": {},
        "specifications": {},
        "prices": {},
        "reviews": {},
        "ai_derived_specs": {},
        "sources": [],
        "visited_urls": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "is_complete": False,
        "search_queries_used": [],
        # Attempt tracking to prevent infinite loops
        "competitor_search_attempts": {},
        "price_search_attempts": {},
        "review_search_attempts": {},
        "spec_search_attempts": {},
        "failed_searches": [],
    }
    
    print("\n[Agent Starting...]\n")
    
    final_state = None
    step = 0
    
    try:
        for event in app.stream(initial_state, {"recursion_limit": 300}):
            node_name = list(event.keys())[0]
            node_output = event[node_name]
            
            if node_name == "agent":
                step += 1
                msgs = node_output.get("messages", [])
                if msgs and hasattr(msgs[-1], "tool_calls") and msgs[-1].tool_calls:
                    print(f"\n[Step {step}] Agent calling:")
                    for tc in msgs[-1].tool_calls:
                        print(f"  🔧 {tc['name']}")
            
            if node_output:
                if final_state is None:
                    final_state = dict(initial_state)
                
                for key, value in node_output.items():
                    if key == "messages" and isinstance(value, list):
                        final_state["messages"] = final_state.get("messages", []) + value
                    elif key == "specifications" and isinstance(value, dict):
                        if value:
                            existing = final_state.get(key, {})
                            for product_name, specs_dict in value.items():
                                if product_name not in existing:
                                    existing[product_name] = {}
                                for spec_name, spec_payload in specs_dict.items():
                                    if spec_name not in existing[product_name] or isinstance(existing[product_name].get(spec_name), str):
                                        existing[product_name][spec_name] = spec_payload
                                    else:
                                        existing[product_name][spec_name].update(spec_payload)
                            final_state[key] = existing
                    elif key == "reviews" and isinstance(value, dict):
                        if value:
                            existing = final_state.get(key, {})
                            for product_name, review_list in value.items():
                                if product_name not in existing:
                                    existing[product_name] = []
                                existing[product_name].extend(review_list)
                            final_state[key] = existing
                    elif key in ["competitors", "products", "prices", "ai_derived_specs"] and isinstance(value, dict):
                        if value:
                            existing = final_state.get(key, {})
                            existing.update(value)
                            final_state[key] = existing
                    elif value is not None:
                        final_state[key] = value
                        
    except Exception as e:
        print(f"\n⚠️ Error: {str(e)[:100]}")
    
    if final_state is None:
        final_state = initial_state
    
    # Extract results
    competitors = final_state.get("competitors", {})
    products = final_state.get("products", {})
    specifications = final_state.get("specifications", {})
    prices = final_state.get("prices", {})
    reviews = final_state.get("reviews", {})
    ai_derived = final_state.get("ai_derived_specs", {})
    
    # Also get globally registered AI-derived attributes
    global_ai_derived = get_ai_derived_attributes()
    
    total_specs = sum(len(s) for s in specifications.values())
    total_reviews = sum(len(r) for r in reviews.values())
    
    print("\n" + "="*70)
    print("🏁 AGENT COMPLETE - FINAL RESULTS")
    print("="*70)
    print(f"✓ Competitors: {len(competitors)} - {list(competitors.keys())}")
    print(f"✓ Products: {len(products)} - {list(products.keys())}")
    print(f"✓ Specifications: {total_specs} total")
    for prod, specs in list(specifications.items())[:3]:
        print(f"    {prod}: {len(specs)} specs")
    print(f"✓ Prices: {len(prices)} - {list(prices.keys())}")
    print(f"✓ Reviews: {total_reviews} total")
    print(f"✓ AI-Derived Specs: {len(global_ai_derived)}")
    print("="*70)
    
    return convert_to_neo4j_format(competitors, products, specifications, prices, reviews, ai_derived)


def convert_to_neo4j_format(competitors, products, specifications, prices, reviews, ai_derived=None) -> Dict[str, Any]:
    """Convert to Neo4j format with enhanced metadata."""
    relationships = []
    baseline_company = "Honeywell"
    
    for name, data in competitors.items():
        if name.strip().lower() == baseline_company.lower():
            continue
        relationships.append({
            "source": baseline_company, "source_type": "Company",
            "relationship": "COMPETES_WITH",
            "target": name, "target_type": "Company",
            "source_url": data.get("source_url", ""),
        })
    
    for name, data in products.items():
        company = data.get("company", "")
        if not company:
            lower_name = name.lower()
            if "honeywell" in lower_name or "smartline" in lower_name or lower_name.startswith("st7"):
                company = baseline_company
        if company:
            relationships.append({
                "source": company, "source_type": "Company",
                "relationship": "OFFERS_PRODUCT",
                "target": name, "target_type": "Product",
            })
    
    for product_name, specs in specifications.items():
        for spec_name, spec_payload in specs.items():
            if isinstance(spec_payload, dict):
                spec_value = spec_payload.get("value", "")
                normalized_value = spec_payload.get("normalized_value", spec_value)
                unit = spec_payload.get("unit", "")
                snippet = spec_payload.get("snippet", "")
                evidence_id = spec_payload.get("evidence_id", "")
                source_url = spec_payload.get("source_url", "")
                ai_derived_flag = spec_payload.get("ai_derived", False)
            else:
                spec_value = spec_payload
                normalized_value = spec_value
                unit = ""
                snippet = ""
                evidence_id = ""
                source_url = ""
                ai_derived_flag = False
            
            relationships.append({
                "source": product_name, "source_type": "Product",
                "relationship": "HAS_SPEC",
                "target": f"{spec_name}: {spec_value}",
                "target_type": "Specification",
                "spec_type": spec_name,
                "spec_value": spec_value,
                "normalized_value": normalized_value,
                "unit": unit,
                "snippet": snippet,
                "evidence_ids": [evidence_id] if evidence_id else [],
                "source_url": source_url,
                "ai_derived": ai_derived_flag,
            })
    
    for product_name, price_payload in prices.items():
        if isinstance(price_payload, dict):
            price_value = price_payload.get("value", "")
            snippet = price_payload.get("snippet", "")
            evidence_id = price_payload.get("evidence_id", "")
            source_url = price_payload.get("source_url", "")
        else:
            price_value = price_payload
            snippet = ""
            evidence_id = ""
            source_url = ""
        relationships.append({
            "source": product_name, "source_type": "Product",
            "relationship": "HAS_PRICE",
            "target": f"{product_name} | {price_value}", "target_type": "Price",
            "snippet": snippet,
            "evidence_ids": [evidence_id] if evidence_id else [],
            "source_url": source_url,
        })
    
    for product_name, review_list in reviews.items():
        for i, review in enumerate(review_list):
            review_text = review.get("text", "")[:100]
            rating = review.get("rating", "")
            source = review.get("source", "")
            snippet = review.get("snippet", review_text)
            evidence_id = review.get("evidence_id", "")
            relationships.append({
                "source": product_name, "source_type": "Product",
                "relationship": "HAS_REVIEW",
                "target": f"{rating}: {review_text}..." if rating else review_text[:50] + "...",
                "target_type": "Review",
                "review_text": review.get("text", ""),
                "rating": rating,
                "review_source": source,
                "source_url": source,
                "snippet": snippet,
                "evidence_ids": [evidence_id] if evidence_id else [],
            })
    
    return {
        "relationships": relationships,
        "competitors": competitors,
        "products": products,
        "specifications": specifications,
        "prices": prices,
        "reviews": reviews,
        "ai_derived_specs": ai_derived or {},
    }
