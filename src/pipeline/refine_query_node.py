from __future__ import annotations

import json
from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from src.config.settings import get_openai_api_key


# Constants (configurable values)
MAX_SEARCH_ATTEMPTS_PER_COMPETITOR = 3  # Give up after 3 tries per competitor
TAVILY_QUERY_MAX_LENGTH = 400  # Tavily API limit
GENERIC_PRODUCT_TERMS = [  # Terms that indicate generic products (not specific models)
    "pressure transmitter",
    "pressure sensor",
    "sensor",
    "transmitter"
]


def _make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=get_openai_api_key(),
        model="gpt-4o-mini",
        temperature=0.3,
        response_format={"type": "json_object"},
    )


def refine_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes current extracted data and decides what to search for next.
    
    LOGIC FLOW (simplified):
    1. Count what we have: X competitors, Y products per competitor
    2. Decide what's missing:
       - If X < max_competitors → Need more competitors → Phase 1
       - If X >= max_competitors → Need products for competitors → Phase 2
    3. Generate a search query to fill the gap
    4. Return the query (or signal we're done)
    
    TWO-PHASE APPROACH:
    
    PHASE 1: Find Competitors (when we have < max_competitors)
    - Goal: Get up to max_competitors companies that compete with target_company
    - Strategy: Generate broad queries like "companies that compete with Honeywell in pressure transmitters"
    - Continues until we have max_competitors (e.g., 5)
    
    PHASE 2: Find Specific Product Models (once we have enough competitors)
    - Goal: For each competitor, find up to max_products specific model names
    - Strategy: Generate focused queries like "Wika pressure transmitter model numbers list"
    - One competitor at a time (more focused = better results)
    - Tracks attempts per competitor (MAX_SEARCH_ATTEMPTS_PER_COMPETITOR) to avoid infinite loops
    - If a competitor has 0 products after max attempts → skip to next competitor
    
    CONFIGURABLE VALUES (not hardcoded):
    - max_competitors: From state (default 5, set in graph_builder.py)
    - max_products: From state (default 3, set in graph_builder.py)
    - max_iterations: From state (default 12, set in graph_builder.py)
    - MAX_SEARCH_ATTEMPTS_PER_COMPETITOR: Constant at top of file (3)
    - TAVILY_QUERY_MAX_LENGTH: Constant at top of file (400, API limit)
    - GENERIC_PRODUCT_TERMS: Constant at top of file (list of terms to filter)
    
    Returns:
        - If more searching needed: {"query": "...", "needs_refinement": True, ...} → loops back to search
        - If complete: {"needs_refinement": False, ...} → goes to write_node (saves to Neo4j)
    """
    
    # Step 1: Get current state and check if we've hit the iteration limit
    data = state.get("data", {})
    original_query = state.get("original_query", state.get("query", ""))
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 12)

    # Safety check: Stop if we've run too many iterations
    if iteration >= max_iterations:
        print(f"[refine_query] Max iterations ({max_iterations}) reached. Stopping.")
        return {"needs_refinement": False, "iteration": iteration}

    # Step 2: Get configuration from state (these come from graph_builder.py, so they're configurable)
    relationships = data.get("Relationships", [])
    max_products = state.get("max_products_per_company", 3)  # Configurable: how many products per competitor
    max_competitors = state.get("max_competitors", 5)         # Configurable: how many competitors to find
    target_company = "Honeywell"  # The company we're building the competitive landscape for (could be made configurable)

    # Step 3: Build data structures to track what we've found
    competitors_list: List[str] = []                      # List of competitor names (max 5)
    products_by_competitor: Dict[str, List[str]] = {}      # Map: competitor -> list of their product models

    # Step 4: Loop through all relationships and organize them into data structures
    # This is where we COUNT what we have so far
    
    for r in relationships:
        # Count COMPETES_WITH relationships (target_company -> Competitor)
        # We only care about relationships where our target company (e.g., Honeywell) is the source
        if r.get("relationship") == "COMPETES_WITH" and r.get("source") == target_company:
            comp = r.get("target")
            # Only track up to max_competitors (maintains the "top N" list)
            if comp and comp not in competitors_list and len(competitors_list) < max_competitors:
                competitors_list.append(comp)
                products_by_competitor[comp] = []  # Initialize empty product list for this competitor
        
        # Count OFFERS_PRODUCT relationships (Competitor -> Product)
        if r.get("relationship") == "OFFERS_PRODUCT":
            comp = r.get("source")
            product = (r.get("target") or "").strip()
            
            # Filter out generic terms (we only want specific model names like "Rosemount 3051")
            # Generic terms like "pressure transmitter" are not useful - we want actual model numbers
            low = product.lower()
            is_generic = any(term in low for term in GENERIC_PRODUCT_TERMS)
            
            # Only track products for competitors we're monitoring (in our top N list)
            # And skip generic product names (they're not specific models)
            if comp in competitors_list and not is_generic:
                products_by_competitor.setdefault(comp, [])
                # Only keep up to max_products per competitor (enforces limit)
                if len(products_by_competitor[comp]) < max_products:
                    products_by_competitor[comp].append(product)

    # Step 5: Calculate current status
    competitors_count = len(competitors_list)
    total_products = sum(len(v) for v in products_by_competitor.values())
    print(f"[refine_query] Tracking {competitors_count}/{max_competitors} competitors, {total_products} total products (max {max_products} per company)")

    # Step 6: Get search attempt tracker (how many times we've searched for each competitor's products)
    search_attempts: Dict[str, int] = state.get("competitor_search_attempts", {})

    # Step 7: Initialize LLM (used in both phases)
    llm = _make_llm()

    # ========================================================================
    # PHASE 1: COMPETITORS MISSING (if we don't have enough yet)
    # ========================================================================
    if competitors_count < max_competitors:
        # We need more COMPETES_WITH relationships in our schema
        # Let LLM analyze what's missing and propose a query
        # This query will be sent to Tavily API, which has a character limit
        # Show sample of relationships for context (first 10)
        relationships_sample = relationships[:10]
        prompt = f"""Analyze the current data and propose the next search query.

Original query: "{original_query}"

Current Relationships (sample):
{json.dumps(relationships_sample, indent=2)}

Analysis:
- Target company: {target_company}
- Found: {competitors_count} COMPETES_WITH relationships
- Goal: {max_competitors} COMPETES_WITH relationships
- Missing: {max_competitors - competitors_count} more company relationships

Propose a query that will help extract the missing COMPETES_WITH relationships.
Keep query under {TAVILY_QUERY_MAX_LENGTH} chars.

Return JSON ONLY:
{{
  "is_complete": false,
  "reason": "Need more COMPETES_WITH relationships",
  "new_query": "concise query under {TAVILY_QUERY_MAX_LENGTH} chars"
}}"""
    
    # ========================================================================
    # PHASE 2: FIND SPECIFIC PRODUCT MODELS (once we have enough competitors)
    # ========================================================================
    else:
        # We have enough competitors (5), now find their specific product models
        # Build list of competitors that still need products
        competitors_needing_products: List[str] = []
        
        for comp, products in products_by_competitor.items():
            # Check if this competitor needs more products
            # Example: they have 1 product but we want 3 total
            if len(products) < max_products:
                attempts = search_attempts.get(comp, 0)
                # Only search if we haven't exceeded max attempts per competitor
                # This prevents infinite loops if a competitor just doesn't have product info available
                if attempts < MAX_SEARCH_ATTEMPTS_PER_COMPETITOR:
                    competitors_needing_products.append(comp)
                else:
                    # We've tried max times for this competitor and still don't have enough products
                    # Give up on this competitor and move to the next one (don't waste more iterations)
                    print(f"[refine_query] Skipping {comp} (attempts={attempts}, max={MAX_SEARCH_ATTEMPTS_PER_COMPETITOR})")

        # If no competitors need more products, we're done!
        if not competitors_needing_products:
            print("[refine_query] All competitors satisfied or max attempts reached.")
            return {"needs_refinement": False, "iteration": iteration}  # Signal to stop searching

        # Pick the first competitor that needs products
        target_competitor = competitors_needing_products[0]
        
        # Increment attempt counter for this competitor
        attempts = search_attempts.get(target_competitor, 0)
        search_attempts[target_competitor] = attempts + 1

        # Ask LLM to analyze what's missing and propose a query for OFFERS_PRODUCT relationships
        # The query will be sent to Tavily API, which has a character limit
        current_products_for_target = products_by_competitor.get(target_competitor, [])
        relationships_for_target = [r for r in relationships if r.get("source") == target_competitor and r.get("relationship") == "OFFERS_PRODUCT"]
        
        prompt = f"""Analyze the current data and propose the next search query.

Original query: "{original_query}"

Current Relationships for {target_competitor}:
{json.dumps(relationships_for_target, indent=2)}

Analysis:
- Company: {target_competitor}
- Found: {len(current_products_for_target)} OFFERS_PRODUCT relationships
- Goal: {max_products} OFFERS_PRODUCT relationships
- Missing: {max_products - len(current_products_for_target)} more product relationships
- Attempt: {attempts + 1} of {MAX_SEARCH_ATTEMPTS_PER_COMPETITOR}

Requirements:
- OFFERS_PRODUCT relationships need SPECIFIC model names/numbers (e.g., "Rosemount 3051", "U5300", "PMP21")
- Avoid generic terms like "pressure transmitter" or "sensor"
- Keep query under {TAVILY_QUERY_MAX_LENGTH} chars

Propose a query that will help extract the missing OFFERS_PRODUCT relationships for {target_competitor}.

Return JSON ONLY:
{{
  "is_complete": false,
  "reason": "Need more OFFERS_PRODUCT relationships for {target_competitor}",
  "new_query": "focused query under {TAVILY_QUERY_MAX_LENGTH} chars"
}}"""

    # ========================================================================
    # STEP 8: ASK LLM TO GENERATE THE NEXT SEARCH QUERY
    # ========================================================================
    prompt += "\n\nReturn ONLY valid JSON."
    response = llm.invoke(prompt)
    content = getattr(response, "content", str(response))

    # Step 9: Parse LLM response and return updated state
    try:
        decision = json.loads(content)
        new_query = decision.get("new_query", "")
        
        # Safety check: Tavily API has a character limit, so truncate if too long
        # We truncate to (max - 3) to add "..." so the truncated query is still valid
        if len(new_query) > TAVILY_QUERY_MAX_LENGTH:
            new_query = new_query[:TAVILY_QUERY_MAX_LENGTH - 3] + "..."
        
        print(f"[refine_query] Generated new query: {new_query}")
        
        # Return the new query so the pipeline can search again
        return {
            "query": new_query,                    # Next query to search
            "needs_refinement": True,              # Signal: keep looping (don't go to write_node yet)
            "iteration": iteration + 1,            # Increment iteration counter
            "competitor_search_attempts": search_attempts,  # Save attempt counters for next iteration
        }
    except Exception as e:
        # If LLM response is malformed, stop searching (safety fallback)
        print(f"[refine_query] Error parsing LLM response: {e}. Stopping refinement.")
        return {"needs_refinement": False, "iteration": iteration}  # Signal to stop and write what we have

