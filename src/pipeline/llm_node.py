"""
LLM Node - Extracts structured competitive intelligence from web content.

ENHANCED WITH ONTOLOGY-AWARE SPECIFICATION EXTRACTION:
- Uses the specification ontology for consistent extraction
- Extracts individual specs as separate relationships
- Validates and normalizes values according to ontology rules

This node uses GPT-4o-mini to analyze web pages and extract:
- Competitor companies (COMPETES_WITH relationships)
- Products offered by competitors (OFFERS_PRODUCT relationships)  
- Prices for products (HAS_PRICE relationships)
- STRUCTURED specifications (HAS_SPEC relationships) <- NEW!
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from src.config.settings import get_openai_api_key
from src.ontology.specifications import (
    PRESSURE_TRANSMITTER_ONTOLOGY,
    get_ontology_for_prompt,
)


def _normalize_company_name(name: str) -> str:
    """Standardize company names for consistency."""
    if not name:
        return name
    
    name = name.strip()
    name_lower = name.lower()
    
    # Common normalizations
    normalizations = {
        "sick": "Sick",
        "wika": "Wika",
        "emerson": "Emerson",
        "siemens": "Siemens",
        "abb": "ABB",
        "yokogawa": "Yokogawa",
        "danfoss": "Danfoss",
    }
    
    if name_lower in normalizations:
        return normalizations[name_lower]
    if "endress" in name_lower and "hauser" in name_lower:
        return "Endress+Hauser"
    if "micro" in name_lower and "sensor" in name_lower:
        return "Micro Sensor"
    if "te connectivity" in name_lower:
        return "TE Connectivity"
    if "honeywell" in name_lower:
        return "Honeywell"
    if "rosemount" in name_lower:
        return "Emerson (Rosemount)"
    
    return name


def _build_enhanced_prompt(text: str, schema: Dict[str, Any], existing_products: dict = None) -> str:
    """
    Build an enhanced extraction prompt with full ontology context.
    """
    # Get ontology for specs
    ontology_prompt = get_ontology_for_prompt()
    
    # Build context of existing products
    product_context = ""
    if existing_products:
        product_context = "\n=== EXISTING PRODUCTS (USE THESE EXACT NAMES) ===\n"
        for company, products in existing_products.items():
            for product in products:
                product_context += f"- {company} → '{product}'\n"
        product_context += "\n"
    
    return f"""You are extracting competitive intelligence for Honeywell pressure transmitters.

{product_context}

=== EXTRACTION SCHEMA ===

You must extract the following relationship types:

1. COMPETES_WITH: Honeywell → Competitor Company
   - Find companies that compete with Honeywell in pressure transmitters
   - Use full company names: 'Endress+Hauser' not 'E+H'

2. OFFERS_PRODUCT: Company → Product Model
   - Extract specific model names/numbers (e.g., "A-10", "PMP21", "Rosemount 3051")
   - NOT generic terms like "pressure transmitter"

3. HAS_PRICE: Product → Price
   - Include currency symbol: "$161.09", "€299.00"
   - ONLY extract prices you SEE in the text

4. HAS_SPEC: Product → Specification Value
   *** THIS IS THE MOST IMPORTANT PART ***
   
   For EACH product, extract as many specifications as you can find:
   
{ontology_prompt}

=== OUTPUT FORMAT ===

Return JSON with this structure:
{{
    "Relationships": [
        {{"source": "Honeywell", "source_type": "Company", "relationship": "COMPETES_WITH", "target": "Wika", "target_type": "Company"}},
        {{"source": "Wika", "source_type": "Company", "relationship": "OFFERS_PRODUCT", "target": "A-10", "target_type": "Product"}},
        {{"source": "A-10", "source_type": "Product", "relationship": "HAS_PRICE", "target": "$161.09", "target_type": "Price"}},
        
        // SPECIFICATIONS - Create one relationship per spec found:
        {{"source": "A-10", "source_type": "Product", "relationship": "HAS_SPEC", "target": "0-6000 psi", "target_type": "Specification", "spec_name": "pressure_range", "spec_value": "0-6000 psi"}},
        {{"source": "A-10", "source_type": "Product", "relationship": "HAS_SPEC", "target": "±0.075%", "target_type": "Specification", "spec_name": "accuracy", "spec_value": "±0.075%"}},
        {{"source": "A-10", "source_type": "Product", "relationship": "HAS_SPEC", "target": "4-20mA", "target_type": "Specification", "spec_name": "output_signal", "spec_value": "4-20mA"}},
        {{"source": "A-10", "source_type": "Product", "relationship": "HAS_SPEC", "target": "IP67", "target_type": "Specification", "spec_name": "ip_rating", "spec_value": "IP67"}}
    ]
}}

=== SOURCE TEXT ===
{text[:6000]}

=== CRITICAL RULES ===
1. Extract EVERY specification you can find - each as a SEPARATE HAS_SPEC relationship
2. Use spec_name from the ontology (e.g., "pressure_range", "accuracy", "output_signal")
3. Only extract data you can SEE in the text - no guessing
4. Be thorough - specifications are the most valuable data!

Return valid JSON only."""


def _coerce_to_json(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response."""
    if not text:
        return {"Relationships": []}
    
    # Try direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed[0] if parsed else {"Relationships": []}
        return parsed
    except Exception:
        pass
    
    # Try extracting from code block
    match = re.search(r"```json?\s*([\s\S]*?)```", text)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, list):
                return parsed[0] if parsed else {"Relationships": []}
            return parsed
        except Exception:
            pass
    
    # Try finding JSON between first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, list):
                return parsed[0] if parsed else {"Relationships": []}
            return parsed
        except Exception:
            pass
    
    return {"Relationships": []}


def extract_with_schema(
    schema: Dict[str, Any], 
    raw_content: str, 
    source_url: str, 
    chunk_ids: List[str] = None, 
    existing_products: dict = None
) -> Dict[str, Any]:
    """
    Extract structured data using the enhanced ontology-aware prompt.
    """
    llm = ChatOpenAI(
        api_key=get_openai_api_key(),
        model="gpt-4o-mini",
        temperature=0,
    )
    
    prompt = _build_enhanced_prompt(raw_content, schema, existing_products)
    
    print(f"[llm] Analyzing: {source_url}")
    print(f"[llm] Content: {len(raw_content)} chars")
    print(f"[llm] Evidence chunks: {len(chunk_ids or [])}")
    
    response = llm.invoke(prompt)
    response_text = getattr(response, "content", str(response))
    print(f"[llm] Response preview: {response_text[:150]}...")
    
    data = _coerce_to_json(response_text)
    
    # Validate and tag relationships
    validated_rels = []
    spec_count = 0
    
    for rel in data.get("Relationships", []):
        # Normalize company names
        if rel.get("source_type") == "Company":
            rel["source"] = _normalize_company_name(rel.get("source", ""))
        if rel.get("target_type") == "Company":
            rel["target"] = _normalize_company_name(rel.get("target", ""))
        
        # Validate prices
        if rel.get("relationship") == "HAS_PRICE":
            price = rel.get("target", "")
            price_numbers = re.sub(r'[^\d\.]', '', price)
            
            if price_numbers and price_numbers in raw_content:
                print(f"[llm] ✅ Validated price '{price}'")
                rel["source_url"] = source_url
                rel["evidence_ids"] = chunk_ids or []
                validated_rels.append(rel)
            else:
                print(f"[llm] ❌ Rejected price '{price}' - not in source")
                continue
        
        # Count specifications
        elif rel.get("relationship") == "HAS_SPEC":
            spec_count += 1
            rel["source_url"] = source_url
            rel["evidence_ids"] = chunk_ids or []
            validated_rels.append(rel)
        
        else:
            rel["source_url"] = source_url
            rel["evidence_ids"] = chunk_ids or []
            validated_rels.append(rel)
    
    print(f"[llm] Extracted {spec_count} specifications, {len(validated_rels)} total relationships")
    
    return {
        "Industry": "",
        "CustomerSegment": "",
        "CustomerNeed": [],
        "HoneywellProduct": "",
        "Competitor": "",
        "CompetitiveProduct": "",
        "Relationships": validated_rels,
        "Doc": {"source_url": source_url},
        "Flags": {}
    }


def llm_state_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Extract data from search results and merge with existing data.
    """
    schema = state.get("schema")
    results = state.get("results", []) or []
    iteration = state.get("iteration", 0)
    existing_data = state.get("data", {})
    max_competitors = state.get("max_competitors", 5)
    max_products = state.get("max_products_per_company", 1)
    
    if not results:
        return {"data": schema}
    
    # Build map of existing products
    existing_products = {}
    if existing_data and existing_data.get("Relationships"):
        for rel in existing_data.get("Relationships", []):
            if rel.get("relationship") == "OFFERS_PRODUCT":
                company = rel.get("source", "")
                product = rel.get("target", "")
                if company and product:
                    if company not in existing_products:
                        existing_products[company] = []
                    if product not in existing_products[company]:
                        existing_products[company].append(product)
    
    result = results[0]
    text = result.get("raw_content") or result.get("content") or ""
    url = result.get("url", "")
    chunk_ids = result.get("chunk_ids", [])
    
    if len(text) > 8000:
        text = text[:8000]
    
    text_with_url = f"SOURCE: {url}\n{text}"
    
    print(f"[llm] Iteration {iteration}")
    if existing_products:
        print(f"[llm] Existing products: {existing_products}")
    
    new_data = extract_with_schema(schema, text_with_url, url, chunk_ids, existing_products)
    
    # Merge with existing data
    if iteration > 0 and existing_data and existing_data.get("Relationships"):
        merged = _merge_data(existing_data, new_data, max_competitors, max_products)
    else:
        merged = new_data
    
    # Accumulate source URLs
    existing_urls = existing_data.get("Doc", {}).get("source_url", []) if existing_data else []
    if isinstance(existing_urls, str):
        existing_urls = [existing_urls]
    all_urls = list(dict.fromkeys(list(existing_urls) + [url]))
    merged["Doc"] = {"source_url": all_urls}
    
    return {"data": merged}


def _merge_data(existing: Dict[str, Any], new: Dict[str, Any], max_competitors: int, max_products: int) -> Dict[str, Any]:
    """
    Merge new extractions with existing data.
    
    ENHANCED: Now properly merges HAS_SPEC relationships
    - Allows multiple specs per product
    - Deduplicates by (product, spec_name) pair
    """
    existing_rels = list(existing.get("Relationships", []))
    new_rels = new.get("Relationships", [])
    
    # Track what we have
    competitors = set()
    products_by_company = {}
    prices_by_product = {}
    specs_by_product = {}  # {product: {spec_name: True}}
    
    for rel in existing_rels:
        rel_type = rel.get("relationship")
        source = rel.get("source", "")
        target = rel.get("target", "")
        
        if rel_type == "COMPETES_WITH" and target:
            competitors.add(target)
        elif rel_type == "OFFERS_PRODUCT" and source:
            if source not in products_by_company:
                products_by_company[source] = set()
            if target:
                products_by_company[source].add(target)
        elif rel_type == "HAS_PRICE" and source and target:
            prices_by_product[source] = target
        elif rel_type == "HAS_SPEC" and source:
            spec_name = rel.get("spec_name", "")
            if source not in specs_by_product:
                specs_by_product[source] = set()
            if spec_name:
                specs_by_product[source].add(spec_name)
    
    # Add new relationships
    for rel in new_rels:
        rel_type = rel.get("relationship")
        source = rel.get("source", "")
        target = rel.get("target", "")
        
        if rel_type == "COMPETES_WITH":
            if target and target not in competitors and len(competitors) < max_competitors:
                competitors.add(target)
                existing_rels.append(rel)
        
        elif rel_type == "OFFERS_PRODUCT":
            if not source or not target:
                continue
            if target in ["ModelName", "model", "N/A", "", "Unknown"]:
                continue
            
            if source not in products_by_company:
                products_by_company[source] = set()
            
            if target not in products_by_company[source] and len(products_by_company[source]) < max_products:
                products_by_company[source].add(target)
                existing_rels.append(rel)
        
        elif rel_type == "HAS_PRICE":
            if not target or target in ["N/A", "", "Unknown", "Price"]:
                continue
            
            if source and source not in prices_by_product:
                prices_by_product[source] = target
                existing_rels.append(rel)
        
        elif rel_type == "HAS_SPEC":
            spec_name = rel.get("spec_name", "")
            if not target or not spec_name:
                continue
            
            if source not in specs_by_product:
                specs_by_product[source] = set()
            
            # Add spec if we don't have this spec type for this product
            if spec_name not in specs_by_product[source]:
                specs_by_product[source].add(spec_name)
                existing_rels.append(rel)
    
    result = dict(existing)
    result["Relationships"] = existing_rels
    return result
