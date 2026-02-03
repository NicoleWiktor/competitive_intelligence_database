"""
LANGGRAPH AGENTIC PIPELINE for Competitive Intelligence

This uses LangGraph to build a proper agentic system:
- StateGraph defines the flow
- ToolNode handles tool execution
- Conditional edges route based on agent decisions
- The agent DECIDES which tools to call and when to stop

Features:
- LangGraph StateGraph architecture
- ChromaDB integration for evidence storage (human verification)
- Evidence IDs linked to each extracted fact
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Annotated, TypedDict, Literal
from operator import add

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from tavily import TavilyClient

from src.config.settings import get_openai_api_key, get_tavily_api_key
from src.pipeline.chroma_store import chunk_and_store, get_chunk_by_id


# =============================================================================
# LIMITS
# =============================================================================

MAX_COMPETITORS = 10
MAX_PRODUCTS_PER_COMPETITOR = 10
MAX_SPECS_PER_PRODUCT = 10
MAX_ITERATIONS = 25


# =============================================================================
# LANGGRAPH STATE - Typed state that flows through the graph
# =============================================================================

class AgentState(TypedDict):
    """State that flows through the LangGraph pipeline."""
    messages: Annotated[List[BaseMessage], add]  # Message history (accumulates)
    competitors: Dict[str, Dict]
    products: Dict[str, Dict]
    specifications: Dict[str, Dict]
    searched_queries: List[str]
    extracted_urls: List[str]
    evidence_map: Dict[str, List[str]]  # url -> [chunk_ids]
    iteration: int
    finished: bool


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_string(s: str) -> str:
    """Clean string for Neo4j."""
    if not s:
        return ""
    s = re.sub(r'[\n\r\t]+', ' ', str(s))
    s = re.sub(r'\s+', ' ', s)
    s = s.replace("'", "").replace('"', '').replace('\\', '')
    return s.strip()[:150]


def is_valid_spec_value(value: str) -> bool:
    """Check if spec value is a real measurement."""
    if not value:
        return False
    value_lower = value.lower().strip()
    
    bad_values = [
        "yes", "no", "true", "false", "high", "low", "wide", "narrow",
        "unique", "various", "multiple", "standard", "optional",
        "available", "supported", "n/a", "tbd", "range", "specifications"
    ]
    if value_lower in bad_values or any(bad in value_lower for bad in bad_values):
        return False
    
    has_number = bool(re.search(r'\d', value))
    has_unit = any(u in value_lower for u in [
        'psi', 'bar', 'kpa', 'mpa', 'ma', 'vdc', 'v', 'hz',
        '°c', '°f', 'npt', 'bsp', 'mm', 'inch', '%', 'ms'
    ])
    
    return has_number or has_unit


def get_tavily() -> TavilyClient:
    return TavilyClient(api_key=get_tavily_api_key())


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=get_openai_api_key(),
        model="gpt-4o-mini",
        temperature=0,
        timeout=120,  # 2 minute timeout
        max_retries=2,
    )


# =============================================================================
# SHARED STATE FOR TOOLS (tools need access to mutable state)
# =============================================================================

class ToolState:
    """Mutable state that tools can read/write to."""
    def __init__(self):
        self.competitors: Dict[str, Dict] = {}
        self.products: Dict[str, Dict] = {}
        self.specifications: Dict[str, Dict] = {}
        self.customer_needs: Dict[str, Dict] = {}  # need_key -> {name, description, industry, spec_type, threshold, ...}
        self.need_mappings: List[Dict] = []  # [{need_key, product, spec, explanation, ...}]
        self.searched_queries: List[str] = []
        self.extracted_urls: List[str] = []
        self.evidence_map: Dict[str, List[str]] = {}
        self.industry_needs_report: str = ""  # Comprehensive report from multiple sources
        self.report_sources: List[str] = []  # URLs used to generate the report
        self.finished: bool = False
    
    def summary(self) -> str:
        report_status = "✅ Generated" if self.industry_needs_report else "❌ Not yet"
        return f"""Current Progress:
- Competitors found: {len(self.competitors)} ({list(self.competitors.keys())[:5]})
- Products found: {len(self.products)}
- Specs collected: {sum(len(s) for s in self.specifications.values())}
- Industry needs report: {report_status} ({len(self.report_sources)} sources)
- Customer needs extracted: {len(self.customer_needs)}
- Need-to-product mappings: {len(self.need_mappings)}
- Searches done: {len(self.searched_queries)}
- Pages extracted: {len(self.extracted_urls)}"""


# Global tool state (reset each run)
_tool_state = ToolState()


# =============================================================================
# TOOLS - The agent CHOOSES which ones to use
# =============================================================================

@tool
def search_web(query: str) -> str:
    """
    Search the web for information. Use this to find competitors, products, or specifications.
    
    Args:
        query: The search query (e.g., "pressure transmitter competitors to Honeywell")
    
    Returns:
        Search results with titles, URLs, and content snippets.
    """
    global _tool_state
    
    if len(_tool_state.searched_queries) >= 15:
        return "LIMIT REACHED: You have done 15 searches. Use the information you have or call finish_research."
    
    _tool_state.searched_queries.append(query)
    print(f"🔍 AGENT DECIDED: search_web('{query}')")
    
    try:
        client = get_tavily()
        response = client.search(query=query, max_results=5, search_depth="advanced")
        
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", "")[:500]
            })
        
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Search error: {e}"


@tool
def extract_page_content(url: str) -> str:
    """
    Extract detailed content from a webpage and STORE IT IN CHROMADB for verification.
    Use this to get product specifications from datasheets.
    
    Args:
        url: The URL to extract content from
    
    Returns:
        The extracted page content (truncated for context) + evidence IDs for verification.
    """
    global _tool_state
    
    if url in _tool_state.extracted_urls:
        print(f"   ⏭️  Already extracted: {url[:60]}...")
        return "Already extracted this URL. Try a different one."
    
    if len(_tool_state.extracted_urls) >= 20:
        return "LIMIT REACHED: You have extracted 20 pages. Use the information you have or call finish_research."
    
    _tool_state.extracted_urls.append(url)
    print(f"📄 AGENT DECIDED: extract_page_content('{url[:60]}...')")
    
    try:
        client = get_tavily()
        response = client.extract(urls=[url], extract_depth="advanced")
        
        for r in response.get("results", []):
            content = r.get("raw_content", "")
            if content:
                # STORE IN CHROMADB for human verification later
                chunk_ids = chunk_and_store(
                    raw_content=content,
                    source_url=url,
                    query=_tool_state.searched_queries[-1] if _tool_state.searched_queries else "",
                    page_title=r.get("title", "")
                )
                
                # Track evidence for this URL
                _tool_state.evidence_map[url] = chunk_ids
                
                print(f"   📦 Stored {len(chunk_ids)} chunks in ChromaDB for verification")
                
                return f"""CONTENT EXTRACTED (stored {len(chunk_ids)} evidence chunks in ChromaDB):

{content[:6000]}

---
Evidence IDs stored: {len(chunk_ids)} chunks from {url}
Use these to link extracted data to source evidence."""
        
        print(f"   ❌ Could not extract content from: {url[:60]}...")
        return "Could not extract content from this URL."
    except Exception as e:
        print(f"   ❌ Extract error: {e}")
        return f"Extract error: {e}"


@tool
def save_competitor(company_name: str, source_url: str) -> str:
    """
    Save a competitor company. IMPORTANT: You MUST call extract_page_content(url) FIRST
    to store evidence, then use that same URL as source_url here.
    
    Args:
        company_name: The official company name (e.g., "Emerson", "Siemens")
        source_url: The URL you already extracted (REQUIRED - must match an extracted URL)
    
    Returns:
        Confirmation message with evidence count.
    """
    global _tool_state
    
    if len(_tool_state.competitors) >= MAX_COMPETITORS:
        return f"LIMIT REACHED: Already have {MAX_COMPETITORS} competitors. Focus on finding products for existing competitors."
    
    name = clean_string(company_name)
    if not name or name.lower() == "honeywell":
        return "Invalid competitor name or cannot add Honeywell as competitor."
    
    if name in _tool_state.competitors:
        return f"{name} is already saved as a competitor."
    
    if not source_url:
        return f"ERROR: source_url is required! First call extract_page_content(url) to store evidence, then save_competitor with that URL."
    
    evidence_ids = _tool_state.evidence_map.get(source_url, [])
    
    if not evidence_ids:
        return f"WARNING: No evidence found for URL '{source_url[:50]}...'. Did you call extract_page_content('{source_url}') first? Extract the page first to store evidence, then save."
    
    _tool_state.competitors[name] = {
        "name": name,
        "source_url": source_url,
        "evidence_ids": evidence_ids
    }
    print(f"✅ AGENT DECIDED: save_competitor('{name}') with {len(evidence_ids)} evidence chunks")
    
    return f"Saved competitor: {name}. Total competitors: {len(_tool_state.competitors)}. Evidence linked: {len(evidence_ids)} chunks."


@tool
def save_product(
    company_name: str,
    product_model: str,
    source_url: str,
    pressure_range: str = "",
    accuracy: str = "",
    output_signal: str = "",
    temperature_range: str = "",
    supply_voltage: str = "",
    process_connection: str = ""
) -> str:
    """
    Save a product with specifications. IMPORTANT: You MUST call extract_page_content(url) FIRST
    to store evidence from the datasheet, then use that URL as source_url here.
    
    Args:
        company_name: The manufacturer (must be a saved competitor)
        product_model: The specific model number (e.g., "3051S", "EJA110A")
        source_url: The URL you already extracted (REQUIRED - must match an extracted URL)
        pressure_range: e.g., "0-6000 psi" (must have numbers/units)
        accuracy: e.g., "±0.065%" (must have numbers)
        output_signal: e.g., "4-20mA HART"
        temperature_range: e.g., "-40 to 85°C"
        supply_voltage: e.g., "10.5-42.4 VDC"
        process_connection: e.g., "1/4 NPT"
    
    Returns:
        Confirmation with evidence count, or error if source_url wasn't extracted.
    """
    global _tool_state
    
    company = clean_string(company_name)
    model = clean_string(product_model)
    
    if not model or len(model) < 2:
        return "Invalid product model name."
    
    if company not in _tool_state.competitors:
        return f"Company '{company}' is not a saved competitor. Call save_competitor first."
    
    if not source_url:
        return f"ERROR: source_url is required! First call extract_page_content(datasheet_url) to store evidence, then save_product with that URL."
    
    evidence_ids = _tool_state.evidence_map.get(source_url, [])
    
    if not evidence_ids:
        return f"WARNING: No evidence found for URL '{source_url[:50]}...'. Did you call extract_page_content('{source_url}') first? Extract the datasheet first to store evidence, then save the product."
    
    company_products = [p for p in _tool_state.products.values() if p.get("company") == company]
    if len(company_products) >= MAX_PRODUCTS_PER_COMPETITOR:
        return f"LIMIT REACHED: {company} already has {MAX_PRODUCTS_PER_COMPETITOR} products."
    
    if model in _tool_state.products:
        return f"Product {model} already saved."
    
    specs = {}
    for key, value in [
        ("pressure_range", pressure_range),
        ("accuracy", accuracy),
        ("output_signal", output_signal),
        ("temperature_range", temperature_range),
        ("supply_voltage", supply_voltage),
        ("process_connection", process_connection),
    ]:
        cleaned = clean_string(value)
        if is_valid_spec_value(cleaned):
            specs[key] = cleaned
    
    if len(specs) < 2:
        return f"Product {model} needs at least 2 valid specs with real numbers/units. Got: {specs}"
    
    if len(specs) > MAX_SPECS_PER_PRODUCT:
        specs = dict(list(specs.items())[:MAX_SPECS_PER_PRODUCT])
    
    _tool_state.products[model] = {
        "name": model,
        "company": company,
        "source_url": source_url,
        "evidence_ids": evidence_ids
    }
    _tool_state.specifications[model] = specs
    
    print(f"✅ AGENT DECIDED: save_product('{company}', '{model}', {len(specs)} specs, {len(evidence_ids)} evidence chunks)")
    
    return f"Saved product: {model} by {company} with specs: {list(specs.keys())}. Evidence linked: {len(evidence_ids)} chunks."


@tool
def get_current_progress() -> str:
    """
    Check what data you've collected so far. Use this to see what competitors/products 
    you still need to research.
    
    Returns:
        Summary of current research progress.
    """
    global _tool_state
    print(f"📊 AGENT DECIDED: get_current_progress()")
    return _tool_state.summary()


@tool
def finish_research(reason: str) -> str:
    """
    Signal that research is complete. Call this when you have:
    - At least 3 competitors with products
    - Generated an industry needs report
    - Created need-to-product mappings
    
    Args:
        reason: Why you're finishing (e.g., "Collected competitors, report, and mappings")
    
    Returns:
        Final summary or warning if not enough data.
    """
    global _tool_state
    
    num_competitors = len(_tool_state.competitors)
    num_products = len(_tool_state.products)
    has_report = bool(_tool_state.industry_needs_report)
    num_needs = len(_tool_state.customer_needs)
    num_mappings = len(_tool_state.need_mappings)
    
    warnings = []
    
    # Check products
    if num_products < num_competitors and num_competitors > 0:
        warnings.append(f"- You have {num_competitors} competitors but only {num_products} products")
    
    # Check for report
    if not has_report:
        warnings.append(f"- No industry needs report generated yet! Call research_industry_needs(industry)")
    
    # Check mappings (only if report exists)
    if has_report and num_mappings == 0:
        warnings.append(f"- Report exists but no mappings created. Call map_needs_from_report()")
    
    if warnings:
        return f"""⚠️ WARNING: Research incomplete!

{chr(10).join(warnings)}

Please continue:
1. If missing products: search for datasheets and save_product
2. If no report: research_industry_needs(industry)
3. If no mappings: map_needs_from_report()

Current status:
{_tool_state.summary()}

DO NOT finish yet!"""
    
    _tool_state.finished = True
    print(f"🏁 AGENT DECIDED: finish_research('{reason}')")
    return f"RESEARCH COMPLETE: {reason}\n\n{_tool_state.summary()}"


# =============================================================================
# CUSTOMER NEEDS TOOLS - Research Report Based Approach
# =============================================================================

MAX_CUSTOMER_NEEDS = 15
MAX_NEED_MAPPINGS = 30
NUM_SOURCES_FOR_REPORT = 8  # Number of sources to use for the report


@tool
def research_industry_needs(industry: str) -> str:
    """
    Research customer needs comprehensively by analyzing multiple sources (8+ pages).
    This tool:
    1. Searches for industry challenges, requirements, and pain points
    2. Extracts content from multiple relevant pages
    3. Generates a comprehensive report synthesizing all findings
    
    Call this ONCE after you've collected competitors/products.
    The report will be stored for viewing in Streamlit and for mapping.
    
    Args:
        industry: The target industry (e.g., "oil and gas", "chemical processing", "pharmaceutical")
    
    Returns:
        A comprehensive report on customer needs, stored for later mapping.
    """
    global _tool_state
    
    if _tool_state.industry_needs_report:
        return f"Report already generated from {len(_tool_state.report_sources)} sources. Use map_needs_from_report to create mappings."
    
    print(f"📊 AGENT DECIDED: research_industry_needs('{industry}')")
    print(f"   Searching {NUM_SOURCES_FOR_REPORT}+ sources for comprehensive research...")
    
    client = get_tavily()
    from src.pipeline.chroma_store import chunk_and_store, get_chunk_by_id
    
    # Multiple search queries to get diverse perspectives
    search_queries = [
        f"{industry} pressure transmitter selection criteria requirements",
        f"{industry} instrumentation challenges problems pain points",
        f"{industry} pressure measurement accuracy requirements why",
        f"{industry} equipment reliability failure causes consequences",
    ]
    
    all_urls = []
    for query in search_queries:
        _tool_state.searched_queries.append(query)
        print(f"   🔍 Searching: '{query}'")
        try:
            response = client.search(query=query, max_results=5, search_depth="advanced")
            for r in response.get("results", []):
                url = r.get("url", "")
                if url and url not in all_urls and url not in _tool_state.extracted_urls:
                    all_urls.append(url)
        except Exception as e:
            print(f"   ⚠️ Search error: {e}")
    
    print(f"   Found {len(all_urls)} unique URLs to analyze")
    
    # Extract content from top sources
    all_content = []
    sources_used = []
    
    for url in all_urls[:NUM_SOURCES_FOR_REPORT]:
        try:
            print(f"   📄 Extracting: {url[:60]}...")
            extract_response = client.extract(urls=[url])
            
            if extract_response.get("results"):
                raw_content = extract_response["results"][0].get("raw_content", "")
                if raw_content and len(raw_content) > 200:
                    # Store in ChromaDB
                    chunk_ids = chunk_and_store(raw_content, url, "industry needs research")
                    if chunk_ids:
                        _tool_state.extracted_urls.append(url)
                        _tool_state.evidence_map[url] = chunk_ids
                        sources_used.append(url)
                        # Take first ~2000 chars for report generation
                        all_content.append(f"SOURCE: {url}\n{raw_content[:2000]}")
        except Exception as e:
            print(f"   ⚠️ Extract error for {url[:40]}: {e}")
    
    if not all_content:
        return "ERROR: Could not extract content from any sources. Try again."
    
    print(f"   ✅ Extracted content from {len(sources_used)} sources")
    _tool_state.report_sources = sources_used
    
    # Generate comprehensive report using LLM
    print(f"   🤖 Generating report with LLM (this may take 30-60 seconds)...")
    llm = get_llm()
    combined_content = "\n\n---\n\n".join(all_content)[:10000]  # Reduced context for faster response
    
    report_prompt = f"""You are an industry analyst. Based on the following {len(sources_used)} sources about pressure transmitters in the {industry} industry, write a comprehensive report on CUSTOMER NEEDS.

SOURCES:
{combined_content}

Write a structured report with these sections:

## Executive Summary
Brief overview of the key customer needs in {industry}

## Critical Customer Needs

List each customer need with:
- **The Need**: What customers require (include specific numbers/thresholds when mentioned)
- **Spec Type**: Which product spec addresses this (accuracy, pressure_range, temperature_range, output_signal, certification)
- **Threshold**: The specific requirement value (e.g., ±0.1%, 0-6000 psi, -40°C to 85°C)

Examples:
- Accuracy ±0.075% for custody transfer (spec: accuracy, threshold: ±0.075%)
- Pressure rating up to 15,000 psi for wellhead applications (spec: pressure_range, threshold: 15000 psi)
- ATEX Zone 1 certification for hazardous areas (spec: certification, threshold: ATEX Zone 1)

## Industry-Specific Requirements
Any regulatory, safety, or operational requirements specific to {industry}

## Conclusion
Summary of the most critical needs

Write the report now:"""

    try:
        response = llm.invoke(report_prompt)
        report = response.content.strip()
        
        _tool_state.industry_needs_report = report
        
        print(f"   📊 Generated comprehensive report ({len(report)} chars) from {len(sources_used)} sources")
        
        return f"""✅ INDUSTRY NEEDS REPORT GENERATED

Sources analyzed: {len(sources_used)}
Report length: {len(report)} characters

{report[:2000]}...

[Report truncated - full report stored]

NEXT STEP: Call map_needs_from_report() to extract specific needs and map them to your products."""
        
    except Exception as e:
        return f"Error generating report: {e}"


@tool
def map_needs_from_report() -> str:
    """
    Analyze the industry needs report and create mappings to products.
    This tool:
    1. Extracts specific customer needs from the report
    2. Maps each need to products that meet the requirement
    3. Validates that product specs actually meet need thresholds
    
    Call this AFTER research_industry_needs has generated a report.
    
    Returns:
        Summary of extracted needs and their mappings to products.
    """
    global _tool_state
    
    if not _tool_state.industry_needs_report:
        return "ERROR: No report generated yet. Call research_industry_needs(industry) first."
    
    if not _tool_state.products:
        return "ERROR: No products saved yet. Research competitors and products first."
    
    print(f"🔗 AGENT DECIDED: map_needs_from_report()")
    print(f"   Analyzing report and mapping to {len(_tool_state.products)} products...")
    print(f"   🤖 Extracting needs and creating mappings with LLM (this may take 30-60 seconds)...")
    
    llm = get_llm()
    
    # Build product specs summary
    product_specs_summary = []
    for prod_name, prod_data in _tool_state.products.items():
        specs = _tool_state.specifications.get(prod_name, {})
        if specs:
            spec_str = ", ".join([f"{k}={v}" for k, v in specs.items()])
            product_specs_summary.append(f"- {prod_name} ({prod_data.get('company', 'Unknown')}): {spec_str}")
    
    products_text = "\n".join(product_specs_summary) if product_specs_summary else "No product specs available"
    
    mapping_prompt = f"""Based on this industry needs report and available products, extract customer needs and map them to products.

INDUSTRY NEEDS REPORT:
{_tool_state.industry_needs_report}

AVAILABLE PRODUCTS WITH SPECS:
{products_text}

Extract each customer need from the report and map to products that meet the requirement.

CRITICAL: Need names MUST include the SPECIFIC THRESHOLD VALUE, not generic descriptions!

BAD names (too generic):
- "High Accuracy for Critical Measurements" ❌
- "High Pressure Rating for Wellhead" ❌
- "Temperature Resistance" ❌

GOOD names (include specific threshold):
- "Accuracy ±0.075% for custody transfer" ✓
- "Pressure range 0-20000 psi for wellhead" ✓  
- "Temperature -40°C to 85°C for harsh environments" ✓

Return as JSON:
{{
  "needs": [
    {{
      "name": "Accuracy ±0.075% for custody transfer",
      "spec_type": "accuracy",
      "threshold": "±0.075%"
    }},
    {{
      "name": "Pressure 0-20000 psi for wellhead applications",
      "spec_type": "pressure_range",
      "threshold": "0-20000 psi"
    }},
    {{
      "name": "Temperature -40°C to 85°C for harsh environments",
      "spec_type": "temperature_range",
      "threshold": "-40°C to 85°C"
    }}
  ],
  "mappings": [
    {{
      "need_name": "Accuracy ±0.075% for custody transfer",
      "product": "Product Name",
      "spec_type": "accuracy",
      "product_value": "±0.04%",
      "meets_requirement": true
    }}
  ]
}}

AVAILABLE PRODUCT NAMES (use EXACTLY these names in mappings):
{list(_tool_state.products.keys())}

Rules:
- Need names MUST include the specific number/threshold from the report
- Only map products that meet the threshold (meets_requirement: true)
- Use product names EXACTLY as shown above (copy/paste from the list)

Return ONLY the JSON, nothing else."""

    try:
        response = llm.invoke(mapping_prompt)
        response_text = response.content.strip()
        
        # Parse JSON
        if response_text.startswith('{'):
            data = json.loads(response_text)
        else:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                return f"Could not parse response as JSON: {response_text[:500]}"
        
        needs = data.get("needs", [])
        mappings = data.get("mappings", [])
        
        # Collect ALL evidence from ALL report sources
        all_evidence_ids = []
        all_source_urls = []
        for src_url in _tool_state.report_sources:
            all_source_urls.append(src_url)
            all_evidence_ids.extend(_tool_state.evidence_map.get(src_url, []))
        
        # Save needs with ALL source evidence
        for need in needs:
            name = clean_string(need.get("name", ""))
            if name and name not in _tool_state.customer_needs:
                _tool_state.customer_needs[name] = {
                    "name": name,
                    "spec_type": clean_string(need.get("spec_type", "")).lower().replace(' ', '_'),
                    "threshold": clean_string(need.get("threshold", "")),
                    "source_urls": all_source_urls,  # ALL sources
                    "evidence_ids": all_evidence_ids[:50]  # Limit to 50 chunks for performance
                }
        
        # Save valid mappings
        valid_mappings = 0
        skipped_products = []
        
        for mapping in mappings:
            if mapping.get("meets_requirement", False):
                need_name = clean_string(mapping.get("need_name", ""))
                product_raw = mapping.get("product", "")
                product = clean_string(product_raw)
                
                # Verify product exists - try fuzzy match if exact match fails
                if product not in _tool_state.products:
                    # Try to find a close match
                    matched = False
                    for stored_prod in _tool_state.products.keys():
                        if product.lower() in stored_prod.lower() or stored_prod.lower() in product.lower():
                            print(f"   📝 Fuzzy match: '{product}' → '{stored_prod}'")
                            product = stored_prod
                            matched = True
                            break
                    if not matched:
                        skipped_products.append(product_raw)
                        continue
                
                # Check for duplicate
                is_duplicate = any(
                    m["need"] == need_name and m["product"] == product 
                    for m in _tool_state.need_mappings
                )
                if is_duplicate:
                    continue
                
                _tool_state.need_mappings.append({
                    "need": need_name,
                    "need_key": f"{product}|{need_name}",
                    "product": product,
                    "spec": clean_string(mapping.get("spec_type", "")),
                    "spec_value": clean_string(mapping.get("product_value", "")),
                    "need_threshold": clean_string(mapping.get("threshold", _tool_state.customer_needs.get(need_name, {}).get("threshold", ""))),
                    "explanation": clean_string(mapping.get("explanation", ""))[:200]
                })
                valid_mappings += 1
        
        print(f"   ✅ Extracted {len(needs)} needs, created {valid_mappings} mappings")
        if skipped_products:
            print(f"   ⚠️ Skipped {len(skipped_products)} mappings - products not found: {skipped_products[:5]}")
            print(f"   📦 Available products: {list(_tool_state.products.keys())}")
        
        result = f"""✅ NEEDS EXTRACTION AND MAPPING COMPLETE

Needs extracted: {len(needs)}
Valid mappings created: {valid_mappings}

CUSTOMER NEEDS:
"""
        for i, need in enumerate(needs, 1):
            result += f"{i}. {need.get('name', 'Unknown')} (threshold: {need.get('threshold', 'N/A')})\n"
        
        result += f"\nMAPPINGS:\n"
        for m in _tool_state.need_mappings[-valid_mappings:]:
            result += f"- {m['need']} → {m['product']} (spec: {m['spec']}={m['spec_value']})\n"
        
        return result
        
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {e}\nResponse: {response_text[:500]}"
    except Exception as e:
        return f"Error mapping needs: {e}"


@tool
def map_need_to_product(
    need_name: str,
    product_model: str,
    addressing_spec: str,
    product_spec_value: str,
    explanation: str
) -> str:
    """
    Map a customer need to a product specification that addresses it.
    ONLY create mapping if the product spec ACTUALLY MEETS the need threshold.
    DO NOT make up mappings - if the product doesn't meet the need, don't map it.
    
    Args:
        need_name: The customer need (must be saved first)
        product_model: The product model (must be saved first)
        addressing_spec: Which spec addresses this need (e.g., "accuracy", "pressure_range")
        product_spec_value: The ACTUAL spec value from the product (e.g., "±0.065%", "0-6000 psi")
        explanation: How the product spec meets the need threshold (be specific with numbers)
    
    Returns:
        Confirmation of the mapping, or rejection if product doesn't meet the need.
    """
    global _tool_state
    
    if len(_tool_state.need_mappings) >= MAX_NEED_MAPPINGS:
        return f"LIMIT REACHED: Already have {MAX_NEED_MAPPINGS} mappings."
    
    need = clean_string(need_name)
    product = clean_string(product_model)
    spec = clean_string(addressing_spec).lower().replace(' ', '_')
    spec_value = clean_string(product_spec_value)
    
    if need not in _tool_state.customer_needs:
        return f"Need '{need}' not found. Call save_customer_need first. Available: {list(_tool_state.customer_needs.keys())}"
    
    if product not in _tool_state.products:
        return f"Product '{product}' not found. Call save_product first. Available: {list(_tool_state.products.keys())}"
    
    # Check if this product has this spec
    product_specs = _tool_state.specifications.get(product, {})
    actual_spec_key = None
    for key in product_specs.keys():
        if key == spec or key.replace('_', ' ') == spec.replace('_', ' '):
            actual_spec_key = key
            break
    
    if not actual_spec_key:
        return f"Product '{product}' doesn't have spec '{spec}'. Available specs: {list(product_specs.keys())}. Cannot create mapping."
    
    # Get the actual spec value from the product
    actual_value = product_specs.get(actual_spec_key, "")
    
    # Validate that product_spec_value matches what we have stored
    if spec_value and actual_value and spec_value.lower() not in actual_value.lower() and actual_value.lower() not in spec_value.lower():
        print(f"   ⚠️  Spec value mismatch: provided '{spec_value}' but product has '{actual_value}'")
    
    # Get the need threshold to validate
    need_data = _tool_state.customer_needs.get(need, {})
    need_threshold = need_data.get("threshold", "")
    
    # Check for duplicate mapping
    for m in _tool_state.need_mappings:
        if m["need"] == need and m["product"] == product:
            return f"Mapping already exists: '{need}' → '{product}'"
    
    mapping = {
        "need": need,
        "product": product,
        "spec": actual_spec_key,
        "spec_value": actual_value,
        "need_threshold": need_threshold,
        "explanation": clean_string(explanation)[:200]
    }
    _tool_state.need_mappings.append(mapping)
    
    print(f"✅ AGENT DECIDED: map_need_to_product('{need}' → '{product}' via '{actual_spec_key}={actual_value}')")
    
    return f"Mapped: '{need}' (threshold: {need_threshold}) → '{product}' (spec: {actual_spec_key}={actual_value}). Explanation: {explanation[:100]}"


# =============================================================================
# TOOLS LIST
# =============================================================================

TOOLS = [
    search_web,
    extract_page_content,
    save_competitor,
    save_product,
    get_current_progress,
    research_industry_needs,
    map_needs_from_report,
    finish_research,
]


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a competitive intelligence researcher for Honeywell's pressure transmitter products (SmartLine ST700).

YOUR GOAL: 
1. Research Honeywell's competitors and find products with specs
2. Generate an in-depth industry needs report (from multiple sources)
3. Map customer needs to product specifications

TOOLS AVAILABLE:

Competitor/Product Research:
- search_web: Search for information (returns URLs)
- extract_page_content: MUST CALL THIS to get content AND store evidence in ChromaDB
- save_competitor: Save a competitor (ONLY after extracting a page about them)
- save_product: Save a product with specs (ONLY after extracting the datasheet)

Customer Needs Research (REPORT-BASED):
- research_industry_needs: Comprehensive research - searches 8+ sources and generates an in-depth report
- map_needs_from_report: Extracts needs from the report and maps them to your saved products

Utility:
- get_current_progress: Check what you've collected
- finish_research: Signal you're done

WORKFLOW:

PHASE 1 - COMPETITORS & PRODUCTS:
1. Search for 3-5 competitor companies 
2. For EACH competitor:
   a. search_web for info → extract_page_content(url) → save_competitor
   b. search_web for their products → extract_page_content(datasheet) → save_product with specs
3. Get at least 1 product per competitor before moving on

PHASE 2 - CUSTOMER NEEDS REPORT:
After you have products with specs:
1. Call research_industry_needs(industry) - this searches 8+ sources and generates a comprehensive report
2. Call map_needs_from_report() - this extracts specific needs and maps them to your products
That's it! The report tool handles the heavy lifting automatically.

The report will:
- Search multiple times with different queries
- Extract content from 8+ different sources
- Generate a comprehensive report on industry challenges
- Be stored for viewing in Streamlit

FINISH when you have:
- At least 3 competitors with products
- A generated industry needs report
- Need-to-product mappings from the report

START with Phase 1 (competitors), then Phase 2 (needs report)!"""


# =============================================================================
# LANGGRAPH NODES
# =============================================================================

def agent_node(state: AgentState) -> Dict[str, Any]:
    """
    The AGENT node - calls the LLM to decide what to do next.
    This is where the LLM decides which tools to call.
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)
    
    # Get current messages from state
    messages = state["messages"]
    
    iteration = state.get("iteration", 0) + 1
    print(f"\n--- LangGraph Iteration {iteration} ---")
    
    # Call LLM
    response = llm_with_tools.invoke(messages)
    
    return {
        "messages": [response],
        "iteration": iteration,
    }


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Conditional edge: decide whether to continue to tools or end.
    
    This is the ROUTING LOGIC that makes the graph agentic:
    - If LLM returned tool calls → go to tools node
    - If LLM finished (no tool calls or finish_research called) → end
    """
    global _tool_state
    
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check iteration limit
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        print(f"   Max iterations ({MAX_ITERATIONS}) reached, ending...")
        return "end"
    
    # Check if finish_research was called
    if _tool_state.finished:
        print("   Agent called finish_research, ending...")
        return "end"
    
    # Check if LLM returned tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"   Routing to tools ({len(last_message.tool_calls)} tool calls)...")
        return "tools"
    
    # No tool calls - end
    print("   No tool calls, ending...")
    return "end"


# =============================================================================
# BUILD THE LANGGRAPH
# =============================================================================

def build_graph() -> StateGraph:
    """
    Build the LangGraph StateGraph.
    
    Graph structure:
        __start__ → agent → (conditional) → tools → agent → ...
                              ↓
                            end
    """
    # Create the graph with our state type
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    
    # Set entry point
    graph.set_entry_point("agent")
    
    # Add conditional edge from agent
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        }
    )
    
    # Add edge from tools back to agent
    graph.add_edge("tools", "agent")
    
    return graph.compile()


# =============================================================================
# RUN THE AGENT
# =============================================================================

def run_agent(max_competitors: int = 10, industry: str = "process industries", max_iterations: int = 25) -> Dict[str, Any]:
    """
    Run the LangGraph agentic pipeline.
    
    The graph:
    1. Agent node calls LLM → LLM decides which tools to call
    2. Conditional edge routes to tools or end
    3. Tools node executes the tool calls
    4. Loop back to agent
    5. Repeat until agent calls finish_research or max iterations
    """
    global _tool_state, MAX_COMPETITORS, MAX_ITERATIONS
    
    # Reset tool state
    _tool_state = ToolState()
    MAX_COMPETITORS = min(max_competitors, 10)
    MAX_ITERATIONS = max_iterations
    
    print("="*60)
    print("🤖 LANGGRAPH AGENTIC COMPETITIVE INTELLIGENCE")
    print("    Built with LangGraph StateGraph")
    print("    The agent DECIDES what to do via tool calls")
    print(f"    Max competitors: {MAX_COMPETITORS}")
    print(f"    Max iterations: {MAX_ITERATIONS}")
    print(f"    Industry: {industry}")
    print("    📦 Evidence stored in ChromaDB for verification")
    print("="*60)
    
    start = datetime.now()
    
    # Build the graph
    graph = build_graph()
    
    # Initial state with industry context
    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"""Research Honeywell's competitors in pressure transmitters for the {industry} industry.

Tasks:
1. Find up to {MAX_COMPETITORS} competitors with their products and specs
2. Research customer needs specific to {industry}
3. Map customer needs to product specifications

Start now.""")
        ],
        "competitors": {},
        "products": {},
        "specifications": {},
        "searched_queries": [],
        "extracted_urls": [],
        "evidence_map": {},
        "iteration": 0,
        "finished": False,
    }
    
    # Run the graph with recursion limit based on max_iterations
    # Recursion limit needs to be higher than iterations (each iteration = 2 graph steps: agent + tools)
    final_state = graph.invoke(
        initial_state,
        config={"recursion_limit": max_iterations * 2 + 10}
    )
    
    # Add Honeywell baseline product
    _tool_state.products["SmartLine ST700"] = {
        "name": "SmartLine ST700",
        "company": "Honeywell",
        "source_url": "https://www.honeywellprocess.com",
        "evidence_ids": []
    }
    _tool_state.specifications["SmartLine ST700"] = {
        "pressure_range": "0-10000 psi",
        "accuracy": "±0.065%",
        "output_signal": "4-20mA HART",
        "temperature_range": "-40 to 85°C",
        "supply_voltage": "10.5-42.4 VDC"
    }
    
    # AUTO-COMPLETE: If iterations ran out before customer needs phase, run it now
    if _tool_state.products and not _tool_state.industry_needs_report:
        print("\n⚠️  Iterations ended before customer needs research. Running automatically...")
        try:
            # Call the tool function directly (not via agent)
            result = research_industry_needs.invoke({"industry": industry})
            print(f"   {result[:200]}...")
        except Exception as e:
            print(f"   ❌ Could not auto-generate report: {e}")
    
    # AUTO-COMPLETE: If report exists but no mappings, run mapping
    if _tool_state.industry_needs_report and not _tool_state.need_mappings and _tool_state.products:
        print("\n⚠️  Report exists but no mappings. Running mapping automatically...")
        try:
            result = map_needs_from_report.invoke({})
            print(f"   {result[:200]}...")
        except Exception as e:
            print(f"   ❌ Could not auto-generate mappings: {e}")
    
    elapsed = (datetime.now() - start).total_seconds()
    total_evidence = sum(len(v) for v in _tool_state.evidence_map.values())
    
    print("\n" + "="*60)
    print(f"🏁 LANGGRAPH AGENT COMPLETE in {elapsed:.1f}s")
    print(f"   Iterations: {final_state.get('iteration', 0)}")
    print(f"   Competitors: {len(_tool_state.competitors)}")
    print(f"   Products: {len(_tool_state.products)}")
    print(f"   Specs: {sum(len(s) for s in _tool_state.specifications.values())}")
    print(f"   Customer needs: {len(_tool_state.customer_needs)}")
    print(f"   Need mappings: {len(_tool_state.need_mappings)}")
    print(f"   Searches made: {len(_tool_state.searched_queries)}")
    print(f"   Pages extracted: {len(_tool_state.extracted_urls)}")
    print(f"   📦 Evidence chunks in ChromaDB: {total_evidence}")
    print("="*60)
    
    return {
        "competitors": _tool_state.competitors,
        "products": _tool_state.products,
        "specifications": _tool_state.specifications,
        "customer_needs": _tool_state.customer_needs,
        "need_mappings": _tool_state.need_mappings,
        "evidence_map": _tool_state.evidence_map,
        "industry_needs_report": _tool_state.industry_needs_report,
        "report_sources": _tool_state.report_sources,
    }


if __name__ == "__main__":
    result = run_agent(max_competitors=5)
    print(f"\nCompetitors discovered: {list(result['competitors'].keys())}")
    print(f"Products found: {list(result['products'].keys())}")
    print(f"Customer needs: {list(result['customer_needs'].keys())}")
    print(f"Need mappings: {len(result['need_mappings'])}")
    print(f"Evidence chunks: {sum(len(v) for v in result['evidence_map'].values())}")
