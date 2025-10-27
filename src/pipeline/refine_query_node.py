from __future__ import annotations

import json
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from src.config.settings import get_openai_api_key


def _make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=get_openai_api_key(),
        model="gpt-4o-mini",
        temperature=0.3,
        response_format={"type": "json_object"},
    )


def refine_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Uses LLM to analyze extracted data and decide if refinement is needed.
    Returns: dict updating 'query', 'needs_refinement', and 'iteration' state keys.
    """
    data = state.get("data", {})
    schema = state.get("schema", {})
    original_query = state.get("original_query", state.get("query", ""))
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    
    # Check if we've hit max iterations
    if iteration >= max_iterations:
        print(f"[refine_query] Max iterations ({max_iterations}) reached. Stopping.")
        return {
            "needs_refinement": False,
            "iteration": iteration
        }
    
    llm = _make_llm()

    
    # Analyze current state
    relationships = data.get("Relationships", [])
    max_products = state.get("max_products_per_company", 3)
    max_competitors = state.get("max_competitors", 5)
    
    # Track competitors in order they appear (to get "top" ones)
    competitors_list = []
    products_by_competitor = {}  # competitor -> list of products
    
    for r in relationships:
        if r.get("relationship") == "COMPETES_WITH" and r.get("source") == "Honeywell":
            comp = r.get("target")
            if comp and comp not in competitors_list:
                # Only track up to max_competitors
                if len(competitors_list) < max_competitors:
                    competitors_list.append(comp)
                    products_by_competitor[comp] = []
        
        if r.get("relationship") == "OFFERS_PRODUCT":
            comp = r.get("source")
            product = r.get("target", "").lower().strip()
            # Check if product is generic
            is_generic = any(term in product for term in ["pressure transmitter", "pressure sensor", "transmitter", "sensor"])
            if comp and comp in competitors_list and not is_generic:
                if comp not in products_by_competitor:
                    products_by_competitor[comp] = []
                # Only add if under the limit
                if len(products_by_competitor[comp]) < max_products:
                    products_by_competitor[comp].append(r.get("target"))
    
    competitors_count = len(competitors_list)
    competitors_with_products = sum(1 for products in products_by_competitor.values() if len(products) > 0)
    total_products = sum(len(products) for products in products_by_competitor.values())
    
    print(f"[refine_query] Tracking {competitors_count}/{max_competitors} competitors, {competitors_with_products} with products, {total_products} total products (max {max_products} per company)")
    
    # Get search attempts tracker (used in both phases)
    search_attempts = state.get("competitor_search_attempts", {})
    
    # PHASE 1: Need at least max_competitors
    if competitors_count < max_competitors:
        prompt = f"""PHASE 1: Find more Honeywell competitors

Original query: "{original_query}"

Current competitors found: {competitors_list}

Generate a search query to find MORE competitors of Honeywell in this market.
Focus on finding competitor company names.

Return JSON:
{{
  "is_complete": false,
  "reason": "Need more competitors (found {competitors_count}, need {max_competitors})",
  "new_query": "search query under 400 characters"
}}"""
    
    # PHASE 2: Need specific product names
    else:
        # Find competitors that don't have enough products yet AND haven't been searched too many times
        competitors_needing_products = []
        for c, products in products_by_competitor.items():
            if len(products) < max_products:
                attempts = search_attempts.get(c, 0)
                if attempts < 3:  # Max 3 attempts per competitor
                    competitors_needing_products.append(c)
                else:
                    print(f"[refine_query] Skipping {c} (tried {attempts} times, still only has {len(products)} products)")
        
        if not competitors_needing_products:
            print(f"[refine_query] All competitors either have {max_products} products or max attempts reached. Done!")
            return {
                "needs_refinement": False,
                "iteration": iteration
            }
        
        # Pick ONE competitor to search for (more focused = better results)
        target_competitor = competitors_needing_products[0]
        current_products = len(products_by_competitor.get(target_competitor, []))
        needed = max_products - current_products
        attempts = search_attempts.get(target_competitor, 0)
        
        print(f"[refine_query] Searching products for: {target_competitor} (has {current_products}/{max_products}, attempt {attempts + 1}/3)")
        
        # Increment attempt counter
        search_attempts[target_competitor] = attempts + 1
        
        prompt = f"""PHASE 2: Find SPECIFIC product model names

Original query: "{original_query}"

Target company: {target_competitor}

Generate a FOCUSED search query to find SPECIFIC product model names/numbers that {target_competitor} offers.

Examples of good queries:
- "{target_competitor} pressure transmitter models specifications"
- "{target_competitor} industrial pressure sensor product line"
- "what pressure transmitter models does {target_competitor} make"

REQUIREMENTS:
- Focus ONLY on {target_competitor}
- Find actual model names/numbers (e.g., "3051", "ABB 266", "SITRANS P")
- NOT generic terms like "pressure transmitter"
- Query must be under 400 characters

Return JSON:
{{
  "is_complete": false,
  "reason": "Need {needed} more products for {target_competitor} (attempt {attempts + 1}/3)",
  "new_query": "focused search query for {target_competitor} specific models"
}}"""
    
    llm = _make_llm()
    prompt += "\n\nReturn ONLY valid JSON."

    response = llm.invoke(prompt)
    content = getattr(response, "content", str(response))
    
    try:
        decision = json.loads(content)
        is_complete = decision.get("is_complete", False)
        reason = decision.get("reason", "")
        new_query = decision.get("new_query", "")
        
        print(f"[refine_query] Iteration {iteration}: {reason}")
        
        if is_complete:
            print("[refine_query] Data is complete. Proceeding to write.")
            return {
                "needs_refinement": False,
                "iteration": iteration
            }
        else:
            # Safety check: Tavily has 400 char limit
            if len(new_query) > 400:
                print(f"[refine_query] WARNING: Query too long ({len(new_query)} chars), truncating to 400")
                new_query = new_query[:397] + "..."
            
            print(f"[refine_query] Generated new query ({len(new_query)} chars): {new_query}")
            return {
                "query": new_query,
                "needs_refinement": True,
                "iteration": iteration + 1,
                "competitor_search_attempts": search_attempts,  # Persist the attempts tracker
            }
    except Exception as e:
        print(f"[refine_query] Error parsing LLM response: {e}. Stopping refinement.")
        return {
            "needs_refinement": False,
            "iteration": iteration
        }

