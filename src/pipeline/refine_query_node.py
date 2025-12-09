"""
Refine Query Node - Intelligently generates next search query based on missing data.

ENHANCED FOR SPECIFICATION EXTRACTION:
- Phase 4 now specifically targets technical specifications
- Uses ontology-aware search queries
- Prioritizes high-importance specs from the ontology

Phases:
Phase 1: Find 5 competitors (max 8 attempts)
Phase 2: Find 1 product per competitor (max 7 attempts each)
Phase 3: Find price for each product (max 7 attempts each)
Phase 4: Find specifications for each product (max 10 attempts each) <- ENHANCED
"""

from __future__ import annotations

import json
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from src.config.settings import get_openai_api_key
from src.ontology.specifications import PRESSURE_TRANSMITTER_ONTOLOGY


def refine_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Analyze extracted data and generate next search query.
    
    ENHANCED: Now tracks individual specification types per product
    to ensure comprehensive spec coverage.
    """
    data = state.get("data", {})
    original_query = state.get("original_query", "")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 100)
    max_competitors = state.get("max_competitors", 5)
    max_products_per_company = state.get("max_products_per_company", 1)
    phase_attempts = state.get("phase_attempts", {
        "competitors": 0, 
        "products": {}, 
        "prices": {}, 
        "specs": {},
        "spec_types": {}  # NEW: Track which spec types we've searched for
    })
    
    # Ensure spec_types exists
    if "spec_types" not in phase_attempts:
        phase_attempts["spec_types"] = {}
    
    if iteration >= max_iterations:
        print(f"[refine] Max iterations ({max_iterations}) reached - stopping")
        return {"needs_refinement": False, "iteration": iteration}
    
    # Analyze current data
    relationships = data.get("Relationships", [])
    
    competitors = set()
    products_by_competitor = {}
    prices_by_product = {}
    specs_by_product = {}  # {product: {spec_name: value}}
    
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
        elif rel_type == "HAS_SPEC":
            spec_name = rel.get("spec_name", "")
            if source not in specs_by_product:
                specs_by_product[source] = {}
            if spec_name:
                specs_by_product[source][spec_name] = target
    
    # Count total specs
    total_specs = sum(len(specs) for specs in specs_by_product.values())
    
    print(f"[refine] Iteration {iteration}: {len(competitors)} competitors, "
          f"{sum(len(p) for p in products_by_competitor.values())} products, "
          f"{len(prices_by_product)} prices, {total_specs} specs")
    print(f"[refine] Competitors: {list(competitors)}")
    print(f"[refine] Products: {dict(products_by_competitor)}")
    print(f"[refine] Specs per product: {dict((k, len(v)) for k, v in specs_by_product.items())}")
    
    llm = ChatOpenAI(
        api_key=get_openai_api_key(),
        model="gpt-4o-mini",
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    
    # PHASE 1: Find competitors
    if len(competitors) < max_competitors and phase_attempts["competitors"] < 8:
        phase_attempts["competitors"] += 1
        
        prompt = f"""Find competitors for Honeywell in the pressure transmitter industry.

Current competitors: {list(competitors)}
Need: {max_competitors - len(competitors)} more

Major competitors to look for:
- Emerson (Rosemount)
- Siemens
- ABB  
- Endress+Hauser
- Yokogawa
- WIKA
- Danfoss

Attempt: {phase_attempts["competitors"]}/8

Generate a search query to find competitor companies.
Return JSON: {{"query": "your search query here"}}"""
        
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        try:
            result = json.loads(content)
            new_query = result.get("query", "")
            print(f"[refine] Phase 1 (Competitors): {new_query}")
            return {
                "query": new_query,
                "needs_refinement": True,
                "iteration": iteration + 1,
                "phase_attempts": phase_attempts,
            }
        except:
            pass
    
    # PHASE 2: Find products
    competitors_needing_products = [
        comp for comp in competitors
        if len(products_by_competitor.get(comp, [])) < max_products_per_company
        and phase_attempts["products"].get(comp, 0) < 7
    ]
    
    if competitors_needing_products:
        target_competitor = competitors_needing_products[0]
        phase_attempts["products"][target_competitor] = phase_attempts["products"].get(target_competitor, 0) + 1
        
        prompt = f"""Find the specific pressure transmitter product model for {target_competitor}.

Need: A specific model name/number like "A-10", "PMP21", "Rosemount 3051", "ST800"
NOT generic terms like "pressure transmitter"

Attempt: {phase_attempts['products'][target_competitor]}/7

Generate a search query.
Return JSON: {{"query": "your search query here"}}"""
        
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        try:
            result = json.loads(content)
            new_query = result.get("query", "")
            print(f"[refine] Phase 2 (Product for {target_competitor}): {new_query}")
            return {
                "query": new_query,
                "needs_refinement": True,
                "iteration": iteration + 1,
                "phase_attempts": phase_attempts,
            }
        except:
            pass
    
    # PHASE 3: Find prices
    products_needing_prices = [
        prod for comp_prods in products_by_competitor.values()
        for prod in comp_prods
        if prod not in prices_by_product
        and phase_attempts["prices"].get(prod, 0) < 7
    ]
    
    if products_needing_prices:
        target_product = products_needing_prices[0]
        phase_attempts["prices"][target_product] = phase_attempts["prices"].get(target_product, 0) + 1
        
        product_company = None
        for comp, prods in products_by_competitor.items():
            if target_product in prods:
                product_company = comp
                break
        
        prompt = f"""Find the price for "{target_product}" pressure transmitter by {product_company}.

Look for:
- List price
- MSRP
- Buy now price
- Distributor price

Attempt: {phase_attempts['prices'][target_product]}/7

Generate a search query.
Return JSON: {{"query": "your search query here"}}"""
        
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        try:
            result = json.loads(content)
            new_query = result.get("query", "")
            print(f"[refine] Phase 3 (Price for {target_product}): {new_query}")
            return {
                "query": new_query,
                "needs_refinement": True,
                "iteration": iteration + 1,
                "phase_attempts": phase_attempts,
            }
        except:
            pass
    
    # PHASE 4: Find specifications (ENHANCED!)
    # Get list of important spec types we should look for
    priority_specs = [
        (name, defn) for name, defn in PRESSURE_TRANSMITTER_ONTOLOGY.items()
        if defn.importance >= 4
    ]
    priority_spec_names = [name for name, _ in priority_specs]
    
    # Find products that need more specs
    products_needing_specs = []
    for comp_prods in products_by_competitor.values():
        for prod in comp_prods:
            current_specs = set(specs_by_product.get(prod, {}).keys())
            missing_priority_specs = set(priority_spec_names) - current_specs
            attempts = phase_attempts["specs"].get(prod, 0)
            
            if missing_priority_specs and attempts < 10:
                products_needing_specs.append({
                    "product": prod,
                    "missing": list(missing_priority_specs)[:3],  # Top 3 missing
                    "attempts": attempts
                })
    
    if products_needing_specs:
        target = products_needing_specs[0]
        target_product = target["product"]
        missing_specs = target["missing"]
        
        phase_attempts["specs"][target_product] = phase_attempts["specs"].get(target_product, 0) + 1
        
        # Find company
        product_company = None
        for comp, prods in products_by_competitor.items():
            if target_product in prods:
                product_company = comp
                break
        
        # Build spec-specific query
        spec_hints = []
        for spec_name in missing_specs:
            if spec_name in PRESSURE_TRANSMITTER_ONTOLOGY:
                spec_def = PRESSURE_TRANSMITTER_ONTOLOGY[spec_name]
                spec_hints.append(spec_def.display_name)
        
        prompt = f"""Find DETAILED technical specifications for "{target_product}" pressure transmitter by {product_company}.

SPECIFICALLY looking for these specs:
{chr(10).join(f"- {hint}" for hint in spec_hints)}

The search should find a datasheet, spec sheet, or product page with technical details.

Attempt: {phase_attempts['specs'][target_product]}/10

Generate a search query that will find a page with detailed specifications.
Return JSON: {{"query": "your search query here"}}"""
        
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        try:
            result = json.loads(content)
            new_query = result.get("query", "")
            print(f"[refine] Phase 4 (Specs for {target_product}, looking for: {spec_hints}): {new_query}")
            return {
                "query": new_query,
                "needs_refinement": True,
                "iteration": iteration + 1,
                "phase_attempts": phase_attempts,
            }
        except:
            pass
    
    # All phases complete
    print("[refine] All phases complete - stopping")
    print(f"[refine] Final counts: {len(competitors)} competitors, "
          f"{sum(len(p) for p in products_by_competitor.values())} products, "
          f"{len(prices_by_product)} prices, {total_specs} specifications")
    
    return {"needs_refinement": False, "iteration": iteration}
