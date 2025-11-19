"""
LLM Node - Extracts structured competitive intelligence from web content.

This node uses GPT-4o-mini to analyze web pages and extract:
- Competitor companies (COMPETES_WITH relationships)
- Products offered by competitors (OFFERS_PRODUCT relationships)  
- Prices for products (HAS_PRICE relationships)

Key Features:
- Price validation: Rejects hallucinated prices by checking source text
- Product name consistency: Passes existing product names to maintain consistency
- Data merging: Accumulates data across iterations up to configured limits
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from src.config.settings import get_openai_api_key


def _normalize_company_name(name: str) -> str:
    """
    Standardize company names for consistency across extractions.
    
    Examples: "sick" -> "Sick", "e+h" -> "Endress+Hauser", "microsensor" -> "Micro Sensor"
    """
    if not name:
        return name
    
    name = name.strip()
    name_lower = name.lower()
    
    # Common company normalizations
    if name_lower == "sick":
        return "Sick"
    if name_lower in ["wika", "emerson", "siemens", "abb", "yokogawa"]:
        return name.capitalize()
    if "endress" in name_lower and "hauser" in name_lower:
        return "Endress+Hauser"
    if "micro" in name_lower and "sensor" in name_lower:
        return "Micro Sensor"
    if "te connectivity" in name_lower:
        return "TE Connectivity"
    if "holykell" in name_lower:
        return "Holykell"
    if "honeywell" in name_lower:
        return "Honeywell"
    
    return name


def _build_prompt(text: str, schema: Dict[str, Any], existing_products: dict = None) -> str:
    """
    Build extraction prompt with schema and existing product context.
    
    The prompt instructs the LLM to:
    1. Extract ALL competitors mentioned (up to 5)
    2. Extract specific product model numbers
    3. Extract prices with currency symbols
    4. Use exact product names consistently
    5. Only extract what's visible in the text (no hallucinations)
    """
    # Build context of existing products for name consistency
    product_context = ""
    if existing_products:
        product_context = "\n=== EXISTING PRODUCTS (USE THESE EXACT NAMES) ===\n"
        for company, products in existing_products.items():
            for product in products:
                product_context += f"- {company} → '{product}'\n"
        product_context += "\n⚠️ When creating HAS_PRICE, use these EXACT product names!\n\n"
    
    return (
        "Extract competitive intelligence for Honeywell pressure transmitters.\n\n"
        + product_context
        + "SCHEMA:\n" + json.dumps(schema, indent=2) + "\n\n"
        
        + "EXTRACTION RULES:\n\n"
        
        + "1. COMPETES_WITH: Honeywell → Competitor Company Name\n"
        + "   Example: {source:'Honeywell', source_type:'Company', relationship:'COMPETES_WITH', target:'Wika', target_type:'Company'}\n"
        + "   - Extract ALL competitors mentioned in the text (up to 5 maximum)\n"
        + "   - Use full company names: 'Endress+Hauser' not 'E+H'\n\n"
        
        + "2. OFFERS_PRODUCT: Company → Specific Product Model\n"
        + "   Example: {source:'Wika', source_type:'Company', relationship:'OFFERS_PRODUCT', target:'A-10', target_type:'Product'}\n"
        + "   - Extract specific model names/numbers (e.g., A-10, PMP21, Rosemount 3051, MPM281)\n"
        + "   - DO NOT use generic terms like 'pressure transmitter' or 'sensor'\n"
        + "   - If a product has a company prefix (e.g., 'Rosemount 3051'), that company owns it\n"
        + "   - If the page is clearly about one company's products, associate products with that company\n\n"
        
        + "3. HAS_PRICE: Product → Price with Currency Symbol\n"
        + "   Example: {source:'A-10', source_type:'Product', relationship:'HAS_PRICE', target:'$161.09', target_type:'Price'}\n"
        + "   - Use the EXACT SAME product name from OFFERS_PRODUCT\n"
        + "   - Only extract prices you can SEE in the text with currency symbols ($, £, €, ¥)\n"
        + "   - DO NOT make up or guess prices\n\n"
        
        + "4. HAS_SPECIFICATION: Product → Specifications String\n"
        + "   Example: {source:'A-10', source_type:'Product', relationship:'HAS_SPECIFICATION', target:'Range: 0-6000 psi, Accuracy: ±0.075%, Output: 4-20mA', target_type:'Specification'}\n"
        + "   - Use the EXACT SAME product name from OFFERS_PRODUCT\n"
        + "   - Extract technical specifications: pressure range, accuracy, output signal, material, temperature range, connection type\n"
        + "   - Combine multiple specs into a single concise string (max 200 chars)\n"
        + "   - Only extract specs you can SEE in the text - no guessing\n\n"
        
        + "IMPORTANT:\n"
        + "- ONLY extract data visible in this text - no hallucinations\n"
        + "- For OFFERS_PRODUCT: Look for product model names associated with companies\n"
        + "- For HAS_PRICE: Price must be explicitly shown in the text\n"
        + "- Use consistent naming: same product name in OFFERS_PRODUCT and HAS_PRICE\n\n"
        
        + "TEXT:\n" + text[:5000] + "\n\n"
        + "Return valid JSON with the extracted relationships."
    )


def _coerce_to_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON from LLM response, handling various formats.
    
    Tries multiple strategies:
    1. Direct JSON parse
    2. Extract from ```json code blocks
    3. Find first { to last }
    
    If LLM returns an array, take the first element.
    """
    if not text:
        return {"Relationships": []}
    
    # Try direct parse
    try:
        parsed = json.loads(text)
        # If it's a list, take the first element
        if isinstance(parsed, list):
            return parsed[0] if parsed else {"Relationships": []}
        return parsed
    except Exception:
        pass
    
    # Try extracting from code block
    match = re.search(r"```json\s*([\s\S]*?)```", text)
    if match:
        try:
            parsed = json.loads(match.group(1))
            # If it's a list, take the first element
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
            # If it's a list, take the first element
            if isinstance(parsed, list):
                return parsed[0] if parsed else {"Relationships": []}
            return parsed
        except Exception:
            pass
    
    return {"Relationships": []}


def extract_with_schema(schema: Dict[str, Any], raw_content: str, source_url: str, chunk_ids: List[str] = None, existing_products: dict = None) -> Dict[str, Any]:
    """
    Use LLM to extract structured data from raw content.
    
    Steps:
    1. Build prompt with schema and existing product context
    2. Call GPT-4o-mini for extraction
    3. Validate HAS_PRICE relationships (check price exists in source)
    4. Normalize company names
    5. Tag each relationship with source URL and evidence_id (ChromaDB chunk IDs)
    """
    llm = ChatOpenAI(
        api_key=get_openai_api_key(),
        model="gpt-4o-mini",
        temperature=0,  # Deterministic output
    )
    
    prompt = _build_prompt(raw_content, schema, existing_products)
    
    print(f"[llm] Analyzing: {source_url}")
    print(f"[llm] Content: {len(raw_content)} chars")
    print(f"[llm] Evidence chunks: {len(chunk_ids or [])}")
    
    # Call LLM
    response = llm.invoke(prompt)
    response_text = getattr(response, "content", str(response))
    print(f"[llm] Response preview: {response_text[:100]}...")
    
    # Parse JSON
    data = _coerce_to_json(response_text)
    
    # Validate and tag relationships
    validated_rels = []
    
    for rel in data.get("Relationships", []):
        # Normalize company names for consistency
        if rel.get("source_type") == "Company":
            rel["source"] = _normalize_company_name(rel.get("source", ""))
        if rel.get("target_type") == "Company":
            rel["target"] = _normalize_company_name(rel.get("target", ""))
        
        # CRITICAL: Validate prices to prevent hallucinations
        if rel.get("relationship") == "HAS_PRICE":
            price = rel.get("target", "")
            # Extract numeric part (e.g., "161.09" from "$161.09")
            price_numbers = re.sub(r'[^\d\.]', '', price)
            
            # Check if these numbers appear in the source text
            if price_numbers and price_numbers in raw_content:
                print(f"[llm] ✅ Validated price '{price}' found in source")
                rel["source_url"] = source_url
                rel["evidence_ids"] = chunk_ids or []  # Link to ChromaDB chunks
                validated_rels.append(rel)
            else:
                print(f"[llm] ❌ Rejected price '{price}' - not found in source")
                continue  # Skip this hallucinated price
        else:
            # Tag with source and evidence
            rel["source_url"] = source_url
            rel["evidence_ids"] = chunk_ids or []  # Link to ChromaDB chunks
            validated_rels.append(rel)
    
    # Wrap in schema structure
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
    
    Process:
    1. Get existing data from state
    2. Build map of existing products (for name consistency)
    3. Extract new data from current search result
    4. Merge with existing data (respecting limits)
    5. Return updated data
    """
    schema = state.get("schema")
    results = state.get("results", []) or []
    iteration = state.get("iteration", 0)
    existing_data = state.get("data", {})
    max_competitors = state.get("max_competitors", 5)
    max_products = state.get("max_products_per_company", 1)
    
    if not results:
        return {"data": schema}
    
    # Build map of existing products: {"Wika": ["A-10"], "Micro Sensor": ["MPM281 Series"], ...}
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
    
    # Process first search result (we get 1 URL per iteration)
    result = results[0]
    text = result.get("raw_content") or result.get("content") or ""
    url = result.get("url", "")
    chunk_ids = result.get("chunk_ids", [])  # Get ChromaDB chunk IDs for evidence linking
    
    # Limit text size for LLM context
    if len(text) > 8000:
        text = text[:8000]
    
    text_with_url = f"SOURCE: {url}\n{text}"
    
    print(f"[llm] Iteration {iteration}")
    if existing_products:
        print(f"[llm] Existing products: {existing_products}")
    
    # Extract new data with evidence linking
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
    all_urls = list(dict.fromkeys(list(existing_urls) + [url]))  # Remove duplicates
    merged["Doc"] = {"source_url": all_urls}
    
    return {"data": merged}


def _merge_data(existing: Dict[str, Any], new: Dict[str, Any], max_competitors: int, max_products: int) -> Dict[str, Any]:
    """
    Merge new extractions with existing data, respecting configured limits.
    
    Limits:
    - Max 5 competitors (COMPETES_WITH)
    - Max 1 product per competitor (OFFERS_PRODUCT)
    - Max 1 price per product (HAS_PRICE)
    - Max 1 specification per product (HAS_SPECIFICATION)
    
    Strategy: Keep existing data, add new data only if under limits.
    """
    existing_rels = list(existing.get("Relationships", []))  # Make a copy
    new_rels = new.get("Relationships", [])
    
    # Track what we already have
    competitors = set()  # Use set for faster lookup
    products_by_company = {}
    prices_by_product = {}
    specs_by_product = {}
    
    # Build tracking from existing relationships
    for rel in existing_rels:
        rel_type = rel.get("relationship")
        source = rel.get("source", "")
        target = rel.get("target", "")
        
        if rel_type == "COMPETES_WITH":
            if target:
                competitors.add(target)
        elif rel_type == "OFFERS_PRODUCT":
            if source:
                if source not in products_by_company:
                    products_by_company[source] = set()
                if target:
                    products_by_company[source].add(target)
        elif rel_type == "HAS_PRICE":
            if source and target:
                prices_by_product[source] = target
        elif rel_type == "HAS_SPECIFICATION":
            if source and target:
                specs_by_product[source] = target
    
    # Add new relationships if under limits
    for rel in new_rels:
        rel_type = rel.get("relationship")
        source = rel.get("source", "")
        target = rel.get("target", "")
        
        # Add new competitor
        if rel_type == "COMPETES_WITH":
            if target and target not in competitors and len(competitors) < max_competitors:
                competitors.add(target)
                existing_rels.append(rel)
        
        # Add new product
        elif rel_type == "OFFERS_PRODUCT":
            # Must have valid source and target
            if not source or not target:
                continue
            # Skip placeholder names
            if target in ["ModelName", "model", "N/A", "", "Unknown"]:
                continue
            
            # Initialize company if new
            if source not in products_by_company:
                products_by_company[source] = set()
            
            # Add if under limit
            if target not in products_by_company[source] and len(products_by_company[source]) < max_products:
                products_by_company[source].add(target)
                existing_rels.append(rel)
        
        # Add new price
        elif rel_type == "HAS_PRICE":
            # Must have valid target
            if not target or target in ["N/A", "", "Unknown", "Price"]:
                continue
            
            # Add if we don't have a price for this product yet
            if source and source not in prices_by_product:
                prices_by_product[source] = target
                existing_rels.append(rel)
        
        # Add new specification
        elif rel_type == "HAS_SPECIFICATION":
            # Must have valid target
            if not target or target in ["N/A", "", "Unknown"]:
                continue
            
            # Add if we don't have specs for this product yet
            if source and source not in specs_by_product:
                specs_by_product[source] = target
                existing_rels.append(rel)
    
    # Return merged result
    result = dict(existing)
    result["Relationships"] = existing_rels
    return result
