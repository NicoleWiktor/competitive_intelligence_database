from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
import re

from src.config.settings import get_openai_api_key


def _load_schema() -> Dict[str, Any]:
    return json.loads(Path("src/schemas/schema.json").read_text(encoding="utf-8"))


def _make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=get_openai_api_key(),
        model="gpt-4o-mini",
        temperature=0,
    )


def _build_prompt(chunk_text: str, schema: Dict[str, Any]) -> str:
    return """Extract Honeywell competitors and ONLY their SPECIFIC PRODUCT MODEL NAMES/NUMBERS.

Return JSON: {"Relationships": [...]}

Relationship format: {"source_type": "Company", "source": "Name", "relationship": "TYPE", "target_type": "Company/Product", "target": "Name"}

RULES:

1. COMPETES_WITH relationships:
   - Format: {"source_type": "Company", "source": "Honeywell", "relationship": "COMPETES_WITH", "target_type": "Company", "target": "CompetitorName"}
   - Extract up to 5 competitors max

2. OFFERS_PRODUCT relationships - READ CAREFULLY:
   - Format: {"source_type": "Company", "source": "CompanyName", "relationship": "OFFERS_PRODUCT", "target_type": "Product", "target": "ModelName"}
   - Extract up to 3 products per company max. View the JSON, if you already see 3 products, do not extract more. 
   
   ✅ ONLY extract if text shows ACTUAL MODEL NAME/NUMBER:
   - YES: "3051", "A-10", "S-20", "PMD75", "U5300", "MPM281", "HK7", "Rosemount 3051"
   - NO GENERIC PRODUCT NAMES: "pressure transmitter", "pressure sensor", "transmitter", "sensor"
   - If the text does not contain the information you need, leave it empty.
   - DO NOT PUT ModelName, model, N/A, Unknown, etc. in the target field. Only EMPTY STRING if no real name.
   
3. Normalize company names consistently (Wika not WIKA, MicroSensor not Microsensor)

TEXT:
""" + chunk_text[:5000] + "\n\nReturn JSON. Remember: ONLY extract specific model names, NOT generic sensor types!"


def extract_with_schema(schema: Dict[str, Any], raw_content: str, source_url: str) -> Dict[str, Any]:
    """Primary entry: receive schema dict and raw text; return schema-shaped JSON."""
    llm = _make_llm()
    prompt = _build_prompt(raw_content, schema)
    msg = llm.invoke(prompt)
    txt = getattr(msg, "content", str(msg))
    print("[llm] raw_preview=", (txt or "")[:150].replace("\n", " "))

    data = _coerce_to_json(txt, schema)
    
    # Wrap in full schema structure
    result = {
        "Industry": "",
        "CustomerSegment": "",
        "CustomerNeed": [],
        "HoneywellProduct": "",
        "Competitor": "",
        "CompetitiveProduct": "",
        "Relationships": data.get("Relationships", []),
        "Doc": {"source_url": source_url},
        "Flags": {}
    }
    return result


def _coerce_to_json(text: str, fallback_schema: Dict[str, Any]) -> Dict[str, Any]:
    if not text:
        return {"Relationships": []}
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```json\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    return {"Relationships": []}


def llm_state_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract data from search results and merge with existing data."""
    schema = state.get("schema") 
    results = state.get("results", []) or []
    iteration = state.get("iteration", 0)
    existing_data = state.get("data", {})
    max_competitors = state.get("max_competitors", 5)
    max_products = state.get("max_products_per_company", 3)
    
    if not results:
        return {"data": schema}
    
    # Bundle search results (limit to prevent huge prompts)
    MAX_CHARS_PER_RESULT = 8000  # ~2k tokens per result
    parts: List[str] = []
    urls: List[str] = []
    
    for r in results:
        text = r.get("raw_content") or r.get("content") or ""
        if text:
            url = r.get("url", "")
            urls.append(url)
            if len(text) > MAX_CHARS_PER_RESULT:
                text = text[:MAX_CHARS_PER_RESULT]
            parts.append(f"SOURCE: {url}\n{text}")
    
    raw_bundle = "\n\n---\n\n".join(parts)
    
    print(f"[llm] Iteration {iteration}, bundle: {len(raw_bundle)} chars")
    
    # Extract new data
    new_data = extract_with_schema(schema, raw_bundle, urls[0] if urls else "")
    
    # Simple Python merge
    if iteration > 0 and existing_data and existing_data.get("Relationships"):
        data = _merge_data(existing_data, new_data, max_competitors, max_products)
    else:
        data = new_data
    
    # Accumulate URLs
    existing_urls = existing_data.get("Doc", {}).get("source_url", []) if existing_data else []
    if isinstance(existing_urls, str):
        existing_urls = [existing_urls]
    all_urls = list(dict.fromkeys(list(existing_urls) + urls))
    data["Doc"] = {"source_url": all_urls}
    
    return {"data": data}


def _merge_data(existing: Dict[str, Any], new: Dict[str, Any], max_competitors: int, max_products: int) -> Dict[str, Any]:
    """Simple Python merge - keep existing, add new up to limits."""
    
    # Start with existing relationships
    existing_rels = existing.get("Relationships", [])
    new_rels = new.get("Relationships", [])
    
    # Track what we have
    competes_with = []
    offers_product = {}
    
    for rel in existing_rels:
        if rel.get("relationship") == "COMPETES_WITH":
            target = rel.get("target", "")
            if target and target not in competes_with:
                competes_with.append(target)
        elif rel.get("relationship") == "OFFERS_PRODUCT":
            source = rel.get("source", "")
            target = rel.get("target", "")
            if source not in offers_product:
                offers_product[source] = []
            if target and target not in offers_product[source]:
                offers_product[source].append(target)
    
    # Add new relationships up to limits
    for rel in new_rels:
        if rel.get("relationship") == "COMPETES_WITH":
            target = rel.get("target", "")
            if target and target not in competes_with and len(competes_with) < max_competitors:
                competes_with.append(target)
                existing_rels.append(rel)
        elif rel.get("relationship") == "OFFERS_PRODUCT":
            source = rel.get("source", "")
            target = rel.get("target", "")
            
            # Skip placeholder/invalid product names
            if not target or target in ["ModelName", "model", "N/A", "", "Unknown"]:
                continue
            
            if source not in offers_product:
                offers_product[source] = []
            if target and target not in offers_product[source] and len(offers_product[source]) < max_products:
                offers_product[source].append(target)
                existing_rels.append(rel)
    
    # Return merged data
    result = {**existing}
    result["Relationships"] = existing_rels
    return result
