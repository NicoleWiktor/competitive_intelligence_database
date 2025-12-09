"""
Refine Query Node - Intelligently generates next search query based on missing data.

This node implements a 4-phase approach:
Phase 1: Find 5 competitors (max 8 attempts)
Phase 2: Find 1 product per competitor (max 7 attempts each)
Phase 3: Find price for each product (max 7 attempts each)
Phase 4: Find specifications for each product (max 5 attempts each)

The node analyzes what data is missing and generates targeted queries to fill gaps.
"""

from __future__ import annotations

import json
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from src.config.settings import get_openai_api_key


def refine_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Analyze extracted data and generate next search query.
    
    Decision logic:
    1. If we have < 5 competitors → Phase 1: search for more competitors
    2. Else if any competitor needs products → Phase 2: search for that product
    3. Else if any product needs price → Phase 3: search for that price
    4. Else → All done, stop iterating
    
    Returns:
        Dictionary with:
        - query: Next search query string
        - needs_refinement: True to continue, False to stop
        - iteration: Incremented counter
        - phase_attempts: Updated attempt tracking
    """
    # Get current state
    data = state.get("data", {})
    original_query = state.get("original_query", "")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 100)  # Increased for Phase 4
    max_competitors = state.get("max_competitors", 5)
    max_products_per_company = state.get("max_products_per_company", 1)
    phase_attempts = state.get("phase_attempts", {"competitors": 0, "products": {}, "prices": {}, "specs": {}, "reviews": {}})
    
    # Safety: Stop if max iterations reached
    if iteration >= max_iterations:
        print(f"[refine] Max iterations ({max_iterations}) reached - stopping")
        return {"needs_refinement": False, "iteration": iteration}
    
    # Analyze what we have
    relationships = data.get("Relationships", [])
    
    competitors = set()  # Companies that compete with Honeywell
    products_by_competitor = {}  # {"Wika": ["A-10"], ...}
    prices_by_product = {}  # {"A-10": "$161.09", ...}
    specs_by_product = {}  # {"A-10": "specifications string", ...}
    reviews_by_product = {}  # {"A-10": 3, ...}
    
    for rel in relationships:
        rel_type = rel.get("relationship", "")
        source = rel.get("source", "")
        target = rel.get("target", "")
        
        if rel_type == "COMPETES_WITH" and source == "Honeywell":
            competitors.add(target)
        elif rel_type == "OFFERS_PRODUCT":
            if source not in products_by_competitor:
                products_by_competitor[source] = []
            if target and target not in products_by_competitor[source]:
                products_by_competitor[source].append(target)
        elif rel_type == "HAS_PRICE":
            prices_by_product[source] = target
        elif rel_type == "HAS_SPECIFICATION":
            specs_by_product[source] = target
        elif rel_type == "HAS_REVIEW":
            reviews_by_product[source] = reviews_by_product.get(source, 0) + 1

    
    # Print progress
    print(f"[refine] Iteration {iteration}: {len(competitors)} competitors, "
          f"{sum(len(p) for p in products_by_competitor.values())} products, "
          f"{len(prices_by_product)} prices, {len(specs_by_product)} specs")
    print(f"[refine] Competitors: {list(competitors)}")
    print(f"[refine] Products: {dict(products_by_competitor)}")
    print(f"[refine] Prices: {dict(prices_by_product)}")
    print(f"[refine] Specifications: {dict(specs_by_product)}")
    
    llm = ChatOpenAI(
        api_key=get_openai_api_key(),
        model="gpt-4o-mini",
        temperature=0.3,  # Slight creativity for query variation
        response_format={"type": "json_object"},
    )
    
    # PHASE 1: Find competitors (try up to 8 times)
    if len(competitors) < max_competitors and phase_attempts["competitors"] < 8:
        phase_attempts["competitors"] += 1
        
        prompt = f"""Find competitors for Honeywell in the pressure transmitter industry.

Current competitors: {list(competitors)}
Need: {max_competitors - len(competitors)} more
Attempt: {phase_attempts["competitors"]}/8

Generate a search query to find competitor companies.
Return JSON: {{"query": "your search query here"}}"""
        
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        try:
            result = json.loads(content)
            new_query = result.get("query", "")
            print(f"[refine] Phase 1 (Competitors, attempt {phase_attempts['competitors']}/8): {new_query}")
            return {
                "query": new_query,
                "needs_refinement": True,
                "iteration": iteration + 1,
                "phase_attempts": phase_attempts,
            }
        except:
            pass
    
    # PHASE 2: Find products for each competitor (try up to 7 times per competitor)
    competitors_needing_products = [
        comp for comp in competitors
        if len(products_by_competitor.get(comp, [])) < max_products_per_company
        and phase_attempts["products"].get(comp, 0) < 7
    ]
    
    if competitors_needing_products:
        target_competitor = competitors_needing_products[0]
        phase_attempts["products"][target_competitor] = phase_attempts["products"].get(target_competitor, 0) + 1
        
        prompt = f"""Find the specific product model for {target_competitor} pressure transmitter.

Need: ONE specific model name/number (e.g., "A-10", "PMP21", "MPM281 Series")
NOT generic terms like "pressure transmitter"

Attempt: {phase_attempts['products'][target_competitor]}/7

Generate a search query.
Return JSON: {{"query": "your search query here"}}"""
        
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        try:
            result = json.loads(content)
            new_query = result.get("query", "")
            print(f"[refine] Phase 2 (Product for {target_competitor}, attempt {phase_attempts['products'][target_competitor]}/7): {new_query}")
            return {
                "query": new_query,
                "needs_refinement": True,
                "iteration": iteration + 1,
                "phase_attempts": phase_attempts,
            }
        except:
            pass
    
    # PHASE 3: Find prices for products (try up to 7 times per product)
    products_needing_prices = [
        prod for comp_prods in products_by_competitor.values()
        for prod in comp_prods
        if prod not in prices_by_product
        and phase_attempts["prices"].get(prod, 0) < 7
    ]
    
    if products_needing_prices:
        target_product = products_needing_prices[0]
        phase_attempts["prices"][target_product] = phase_attempts["prices"].get(target_product, 0) + 1
        
        # Find which company makes this product
        product_company = None
        for comp, prods in products_by_competitor.items():
            if target_product in prods:
                product_company = comp
                break
        
        prompt = f"""Find the price for "{target_product}" made by {product_company}.

Attempt: {phase_attempts['prices'][target_product]}/7

Generate a search query to find the price.
Return JSON: {{"query": "your search query here"}}"""
        
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        try:
            result = json.loads(content)
            new_query = result.get("query", "")
            print(f"[refine] Phase 3 (Price for {target_product}, attempt {phase_attempts['prices'][target_product]}/7): {new_query}")
            return {
                "query": new_query,
                "needs_refinement": True,
                "iteration": iteration + 1,
                "phase_attempts": phase_attempts,
            }
        except:
            pass
    
    # PHASE 4: Find specifications for products (try up to 5 times per product)
    products_needing_specs = [
        prod for comp_prods in products_by_competitor.values()
        for prod in comp_prods
        if prod not in specs_by_product
        and phase_attempts["specs"].get(prod, 0) < 5
    ]
    
    if products_needing_specs:
        target_product = products_needing_specs[0]
        phase_attempts["specs"][target_product] = phase_attempts["specs"].get(target_product, 0) + 1
        
        # Find which company makes this product
        product_company = None
        for comp, prods in products_by_competitor.items():
            if target_product in prods:
                product_company = comp
                break
        
        prompt = f"""Find detailed specifications for the "{target_product}" pressure transmitter made by {product_company}.

Look for technical specifications like:
- Pressure range (e.g., "0-6000 psi")
- Accuracy (e.g., "±0.075%")
- Output signal (e.g., "4-20 mA")
- Material (e.g., "316 stainless steel")
- Temperature range (e.g., "-40 to 85°C")
- Process connection (e.g., "1/4 NPT")

Attempt: {phase_attempts['specs'][target_product]}/5

Generate a search query to find technical specifications.
Return JSON: {{"query": "your search query here"}}"""
        
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        try:
            result = json.loads(content)
            new_query = result.get("query", "")
            print(f"[refine] Phase 4 (Specs for {target_product}, attempt {phase_attempts['specs'][target_product]}/5): {new_query}")
            return {
                "query": new_query,
                "needs_refinement": True,
                "iteration": iteration + 1,
                "phase_attempts": phase_attempts,
            }
        except:
            pass
    

    # PHASE 5: Find customer reviews for products (max 5 attempts per product)
    products_needing_reviews = [
        prod for comp_prods in products_by_competitor.values()
        for prod in comp_prods
        if reviews_by_product.get(prod, 0) == 0
        and phase_attempts["reviews"].get(prod, 0) < 5
    ]

    if products_needing_reviews:
        target_product = products_needing_reviews[0]
        phase_attempts["reviews"][target_product] = phase_attempts["reviews"].get(target_product, 0) + 1

        # Identify the company that makes the product
        product_company = None
        for comp, prods in products_by_competitor.items():
            if target_product in prods:
                product_company = comp
                break

        prompt = f"""Find customer reviews for the product "{target_product}" made by {product_company}.

Look specifically for:
- Customer review sections
- Ratings (stars or numeric)
- Reviewer names
- Review dates
- Long-form review text

These sites often contain reviews:
- Grainger
- AutomationDirect
- Amazon Business
- Industrial distributor websites

Attempt: {phase_attempts['reviews'][target_product]}/5

Generate a search query that is very likely to yield real customer reviews.
Return JSON: {{"query": "your search query here"}}"""

        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))

        try:
            result = json.loads(content)
            new_query = result.get("query", "")
            print(f"[refine] Phase 5 (Reviews for {target_product}, attempt {phase_attempts['reviews'][target_product]}/5): {new_query}")
            return {
                "query": new_query,
                "needs_refinement": True,
                "iteration": iteration + 1,
                "phase_attempts": phase_attempts,
            }
        except:
            pass

    # All phases complete - done!
    print("[refine] All phases complete - stopping")
    return {"needs_refinement": False, "iteration": iteration}
