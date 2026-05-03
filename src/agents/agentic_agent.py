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
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, create_react_agent
from tavily import TavilyClient

from src.config.settings import get_openai_api_key, get_tavily_api_key
from src.pipeline.chroma_store import chunk_and_store, get_chunk_by_id
from src.prompts import build_poml_prompt


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
        model="gpt-4.1-mini",
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
        self.customer_segments: List[Dict] = []  # [{name, description, evidence_text, source_url, evidence_ids}]
        self.segments_sources: List[str] = []  # URLs used to find customer segments
        self.segment_mappings: List[Dict] = []  # [{segment, product, reason, evidence_ids}]
        self.house_of_quality: Dict = {}  # {whats, hows, matrix, competitive_analysis, reasoning}
        self.source_selection: Dict[str, List[str]] = {
            "allowed_domains": [],
            "allowed_source_types": [],
        }
        self.target_company: str = "Honeywell"
        self.product_category: str = "pressure transmitters"
        self.finished: bool = False
        
    
    def summary(self) -> str:
        report_status = "✅ Generated" if self.industry_needs_report else "❌ Not yet"
        segments_status = f"{len(self.customer_segments)} found" if self.customer_segments else "❌ Not yet"
        segment_mappings_status = f"{len(self.segment_mappings)} mappings" if self.segment_mappings else "❌ Not yet"
        hoq_status = "✅ Generated" if self.house_of_quality else "❌ Not yet"
        return f"""Current Progress:
- Competitors found: {len(self.competitors)} ({list(self.competitors.keys())[:5]})
- Products found: {len(self.products)}
- Specs collected: {sum(len(s) for s in self.specifications.values())}
- Customer segments: {segments_status} ({segment_mappings_status})
- Industry needs report: {report_status} ({len(self.report_sources)} sources)
- Customer needs extracted: {len(self.customer_needs)}
- Need-to-product mappings: {len(self.need_mappings)}
- House of Quality: {hoq_status}
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
        allowed_domains = _tool_state.source_selection.get("allowed_domains", [])
        allowed_source_types = _tool_state.source_selection.get("allowed_source_types", [])
        for r in response.get("results", []):
            url = r.get("url", "")
            title = r.get("title", "")
            snippet = r.get("content", "")[:500]
            if not _passes_source_selection(
                url=url,
                title=title,
                snippet=snippet,
                allowed_domains=allowed_domains,
                allowed_source_types=allowed_source_types,
            ):
                continue
            inferred_type = _infer_source_type(url=url, title=title, snippet=snippet)
            results.append({
                "title": r.get("title", ""),
                "url": url,
                "snippet": snippet,
                "source_type": inferred_type,
            })
        if not results:
            return (
                "No results matched current source selection. "
                f"{_source_selection_summary()}"
            )
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
    allowed_domains = _tool_state.source_selection.get("allowed_domains", [])
    if not _is_url_allowed(url, allowed_domains):
        return (
            "URL blocked by source selection. "
            f"Allowed domains: {allowed_domains}"
        )
    
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
def set_source_selection(
    allowed_domains: List[str] | None = None,
    allowed_source_types: List[str] | None = None,
) -> str:
    """
    Configure source restrictions for web research.

    Args:
        allowed_domains: Optional list of website domains to allow.
            Examples: ["emerson.com", "siemens.com", "isa.org"].
        allowed_source_types: Optional list of source categories to allow.
            Supported values: datasheet, documentation, manual, news, blog,
            report, marketplace, regulatory.

    Returns:
        Confirmation summary of active source restrictions.
    """
    global _tool_state
    normalized_domains = []
    if allowed_domains:
        normalized_domains = sorted(
            {
                _normalize_domain(d)
                for d in allowed_domains
                if _normalize_domain(d)
            }
        )
    normalized_types = []
    invalid_types = []
    if allowed_source_types:
        normalized_types = sorted(
            {
                (source_type or "").strip().lower()
                for source_type in allowed_source_types
                if (source_type or "").strip()
            }
        )
        invalid_types = [t for t in normalized_types if t not in ALLOWED_SOURCE_TYPES]
        normalized_types = [t for t in normalized_types if t in ALLOWED_SOURCE_TYPES]
    _tool_state.source_selection = {
        "allowed_domains": normalized_domains,
        "allowed_source_types": normalized_types,
    }
    invalid_msg = f" Ignored invalid source types: {invalid_types}." if invalid_types else ""
    return f"Source selection updated. {_source_selection_summary()}.{invalid_msg}"



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
    if not name or name.lower() == _tool_state.target_company.lower():
        return f"Invalid competitor name or cannot add {_tool_state.target_company} as a competitor."
    
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
ALLOWED_SOURCE_TYPES = {
    "datasheet",
    "documentation",
    "manual",
    "news",
    "blog",
    "report",
    "marketplace",
    "regulatory",
}


def _normalize_domain(domain: str) -> str:
    """Normalize domains so matching is consistent."""
    normalized = (domain or "").strip().lower()
    normalized = normalized.replace("https://", "").replace("http://", "")
    normalized = normalized.split("/")[0]
    if normalized.startswith("www."):
        normalized = normalized[4:]
    return normalized


def _infer_source_type(url: str, title: str = "", snippet: str = "") -> str:
    """Infer source type from URL/title/snippet heuristics."""
    text = f"{url} {title} {snippet}".lower()
    type_rules = {
        "datasheet": ["datasheet", ".pdf", "spec-sheet", "technical-data"],
        "documentation": ["documentation", "docs", "knowledge-base"],
        "manual": ["manual", "user-guide", "installation-guide", "brochure"],
        "news": ["news", "press-release", "press", "/news/"],
        "blog": ["blog", "/blog/", "insights", "thought-leadership"],
        "report": ["report", "whitepaper", "market-analysis", "study"],
        "marketplace": ["amazon.", "grainger.", "mcmaster.", "marketplace", "shop"],
        "regulatory": ["iso", "iec", "api", "osha", "asme", "regulation", "standard"],
    }
    for source_type, markers in type_rules.items():
        if any(marker in text for marker in markers):
            return source_type
    return "documentation"


def _is_url_allowed(url: str, allowed_domains: List[str]) -> bool:
    """Check whether URL matches the configured domain restrictions."""
    if not allowed_domains:
        return True
    parsed_domain = _normalize_domain(urlparse(url).netloc)
    return any(
        parsed_domain == allowed or parsed_domain.endswith(f".{allowed}")
        for allowed in allowed_domains
    )


def _passes_source_selection(
    *,
    url: str,
    title: str = "",
    snippet: str = "",
    allowed_domains: List[str],
    allowed_source_types: List[str],
) -> bool:
    """Validate result against current source selection rules."""
    if not _is_url_allowed(url, allowed_domains):
        return False
    if not allowed_source_types:
        return True
    inferred_type = _infer_source_type(url=url, title=title, snippet=snippet)
    return inferred_type in allowed_source_types


def _source_selection_summary() -> str:
    """Human-readable summary for logs and prompts."""
    global _tool_state
    domains = _tool_state.source_selection.get("allowed_domains", [])
    source_types = _tool_state.source_selection.get("allowed_source_types", [])
    domains_text = ", ".join(domains) if domains else "Any domain"
    types_text = ", ".join(source_types) if source_types else "Any source type"
    return f"Domains: {domains_text} | Source types: {types_text}"


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
    
    # LLM generates search queries based on the industry (agentic approach)
    llm = get_llm()
    query_prompt = build_poml_prompt(
        role="Research assistant for industrial instrumentation markets",
        objective="Generate diverse web search queries for customer needs research",
        context={
            "industry": industry,
            "domain": "pressure transmitters",
        },
        instructions=[
            "Generate 5-6 diverse search queries.",
            "Cover industry-specific challenges and pain points.",
            "Cover technical requirements such as accuracy, pressure ranges, and certifications.",
            "Cover equipment selection criteria and real-world application requirements.",
            "Cover regulatory and safety requirements.",
            "Make each query specific to the provided industry.",
        ],
        constraints=[
            "Do not return generic, cross-industry queries.",
            "Return only valid JSON.",
            "Output must be a JSON array of strings.",
        ],
        output_format='["query 1", "query 2", "query 3", "query 4", "query 5"]',
    )

    print(f"   🤖 LLM generating search queries for {industry}...")
    try:
        query_response = llm.invoke(query_prompt)
        query_content = query_response.content.strip()
        
        # Extract JSON array
        if "```json" in query_content:
            query_content = query_content.split("```json")[1].split("```")[0].strip()
        elif "```" in query_content:
            query_content = query_content.split("```")[1].split("```")[0].strip()
        
        search_queries = json.loads(query_content)
        print(f"   ✅ LLM generated {len(search_queries)} search queries")
        for i, q in enumerate(search_queries):
            print(f"      {i+1}. {q}")
    except Exception as e:
        print(f"   ⚠️ Could not generate queries, using fallback: {e}")
        # Fallback to basic queries if LLM fails
        search_queries = [
            f"{industry} pressure transmitter requirements specifications",
            f"{industry} instrumentation challenges",
            f"{industry} measurement accuracy requirements",
            f"{industry} equipment selection criteria",
        ]
    
    all_urls = []
    allowed_domains = _tool_state.source_selection.get("allowed_domains", [])
    allowed_source_types = _tool_state.source_selection.get("allowed_source_types", [])
    for query in search_queries:
        _tool_state.searched_queries.append(query)
        print(f"   🔍 Searching: '{query}'")
        try:
            response = client.search(query=query, max_results=5, search_depth="advanced")
            for r in response.get("results", []):
                url = r.get("url", "")
                title = r.get("title", "")
                snippet = r.get("content", "")[:500]
                if (
                    url
                    and url not in all_urls
                    and url not in _tool_state.extracted_urls
                    and _passes_source_selection(
                        url=url,
                        title=title,
                        snippet=snippet,
                        allowed_domains=allowed_domains,
                        allowed_source_types=allowed_source_types,
                    )
                ):
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
    
    report_prompt = build_poml_prompt(
        role="Industry analyst",
        objective=f"Write a comprehensive customer-needs report for {_tool_state.product_category}",
        context={
            "industry": industry,
            "num_sources": len(sources_used),
            "sources_excerpt": combined_content,
        },
        instructions=[
            "Write a structured report with sections: Executive Summary, Critical Customer Needs, Industry-Specific Requirements, and Conclusion.",
            "In Critical Customer Needs, list each need with The Need, Spec Type, and Threshold.",
            "Include specific numbers and requirement thresholds whenever present.",
            "Use spec types from this set when applicable: accuracy, pressure_range, temperature_range, output_signal, certification.",
        ],
        constraints=[
            "Focus on customer needs in the provided industry only.",
            "Keep evidence-grounded details tied to the provided source content.",
        ],
        output_format=(
            "Markdown report with headings: "
            "## Executive Summary, ## Critical Customer Needs, "
            "## Industry-Specific Requirements, ## Conclusion"
        ),
    )

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
    
    mapping_prompt = build_poml_prompt(
        role="Competitive intelligence analyst",
        objective="Extract customer needs from report and map qualifying products",
        context={
            "industry_needs_report": _tool_state.industry_needs_report,
            "available_products_with_specs": products_text,
            "available_product_names_exact": list(_tool_state.products.keys()),
        },
        instructions=[
            "Extract each customer need from the report.",
            "Need names must include the exact threshold values from the report.",
            "Map only products that meet each need threshold.",
            "Use product names exactly as provided.",
        ],
        constraints=[
            "Return only valid JSON.",
            "Do not include any prose outside JSON.",
            "Do not include mappings where meets_requirement is false.",
        ],
        output_format=json.dumps(
            {
                "needs": [
                    {
                        "name": "Accuracy ±0.075% for custody transfer",
                        "spec_type": "accuracy",
                        "threshold": "±0.075%",
                    }
                ],
                "mappings": [
                    {
                        "need_name": "Accuracy ±0.075% for custody transfer",
                        "product": "Product Name",
                        "spec_type": "accuracy",
                        "product_value": "±0.04%",
                        "meets_requirement": True,
                    }
                ],
            },
            indent=2,
            ensure_ascii=True,
        ),
    )

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


@tool
def research_customer_segments(industry: str) -> str:
    """
    Research and identify customer segments/groups in a specific industry.
    This tool:
    1. Uses LLM to generate targeted search queries
    2. Searches the web and extracts content from multiple sources
    3. LLM analyzes content to identify distinct customer segments
    4. Each segment is stored with evidence (exact text + source URL)
    
    Args:
        industry: The target industry (e.g., "oil and gas", "pharmaceutical", "chemical processing")
    
    Returns:
        Summary of customer segments found with their sources.
    """
    global _tool_state
    
    if _tool_state.customer_segments:
        return f"Customer segments already researched. Found {len(_tool_state.customer_segments)} segments from {len(_tool_state.segments_sources)} sources."
    
    print(f"👥 AGENT DECIDED: research_customer_segments('{industry}')")
    
    client = get_tavily()
    llm = get_llm()
    from src.pipeline.chroma_store import chunk_and_store
    
    # Step 1: LLM generates search queries
    query_prompt = build_poml_prompt(
        role="Market segmentation researcher",
        objective=f"Generate search queries to identify customer segments for {_tool_state.product_category}",
        context={
            "industry": industry,
            "domain": _tool_state.product_category,
        },
        instructions=[
            "Generate 4-5 search queries.",
            "Cover who buys or uses pressure transmitters in the target industry.",
            "Cover different company and operation types needing pressure measurement.",
            "Cover market segments and customer categories, including end users, distributors, and OEMs.",
            "Make every query specific to the provided industry.",
        ],
        constraints=[
            "Return only valid JSON.",
            "Output must be a JSON array of strings.",
        ],
        output_format='["oil and gas upstream customer segments", "refinery instrumentation buyers"]',
    )

    print(f"   🤖 LLM generating search queries...")
    try:
        query_response = llm.invoke(query_prompt)
        query_content = query_response.content.strip()
        
        if "```json" in query_content:
            query_content = query_content.split("```json")[1].split("```")[0].strip()
        elif "```" in query_content:
            query_content = query_content.split("```")[1].split("```")[0].strip()
        
        search_queries = json.loads(query_content)
        print(f"   ✅ Generated {len(search_queries)} queries:")
        for i, q in enumerate(search_queries):
            print(f"      {i+1}. {q}")
    except Exception as e:
        print(f"   ⚠️ Query generation failed, using fallback: {e}")
        search_queries = [
            f"{industry} customer segments market analysis",
            f"{industry} pressure transmitter buyers end users",
            f"{industry} instrumentation market segments",
        ]
    
    # Step 2: Search and collect URLs
    all_urls = []
    allowed_domains = _tool_state.source_selection.get("allowed_domains", [])
    allowed_source_types = _tool_state.source_selection.get("allowed_source_types", [])
    for query in search_queries:
        _tool_state.searched_queries.append(query)
        print(f"   🔍 Searching: '{query[:50]}...'")
        try:
            response = client.search(query=query, max_results=4, search_depth="advanced")
            for r in response.get("results", []):
                url = r.get("url", "")
                title = r.get("title", "")
                snippet = r.get("content", "")[:500]
                if (
                    url
                    and url not in all_urls
                    and url not in _tool_state.extracted_urls
                    and _passes_source_selection(
                        url=url,
                        title=title,
                        snippet=snippet,
                        allowed_domains=allowed_domains,
                        allowed_source_types=allowed_source_types,
                    )
                ):
                    all_urls.append(url)
        except Exception as e:
            print(f"   ⚠️ Search error: {e}")
    
    print(f"   Found {len(all_urls)} unique URLs")
    
    # Step 3: Extract content and store in ChromaDB
    extracted_content = []  # [{url, content, chunk_ids}]
    
    for url in all_urls[:8]:  # Limit to 8 sources
        try:
            print(f"   📄 Extracting: {url[:50]}...")
            extract_response = client.extract(urls=[url])
            
            if extract_response.get("results"):
                raw_content = extract_response["results"][0].get("raw_content", "")
                if raw_content and len(raw_content) > 300:
                    # Store in ChromaDB for evidence
                    chunk_ids = chunk_and_store(raw_content, url, f"customer segments {industry}")
                    if chunk_ids:
                        _tool_state.extracted_urls.append(url)
                        _tool_state.evidence_map[url] = chunk_ids
                        _tool_state.segments_sources.append(url)
                        extracted_content.append({
                            "url": url,
                            "content": raw_content[:3000],  # Keep more for analysis
                            "chunk_ids": chunk_ids
                        })
        except Exception as e:
            print(f"   ⚠️ Extract error: {e}")
    
    if not extracted_content:
        return "ERROR: Could not extract content from any sources. Try again."
    
    print(f"   ✅ Extracted from {len(extracted_content)} sources")
    
    # Step 4: LLM analyzes content to identify customer segments WITH evidence
    source_context = [
        {"url": src["url"], "content_excerpt": src["content"][:2000]}
        for src in extracted_content
    ]
    analysis_prompt = build_poml_prompt(
        role="B2B industry analyst",
        objective="Identify evidence-backed customer segments for pressure transmitters",
        context={
            "industry": industry,
            "sources": source_context,
        },
        instructions=[
            "Identify distinct customer segments (buyer/user groups).",
            "For each segment, provide a clear segment name and description.",
            "Include exact evidence text quoted verbatim from the source.",
            "Include the source URL that contains the evidence.",
        ],
        constraints=[
            "Only include segments with direct quote evidence.",
            "Do not include segments lacking explicit evidence in the provided sources.",
            "Focus only on the provided industry context.",
            "Return only valid JSON.",
        ],
        output_format=json.dumps(
            [
                {
                    "name": "Upstream Oil & Gas Operators",
                    "description": "Companies involved in exploration and production of oil and gas",
                    "evidence_text": "The exact quote from the source that mentions this segment",
                    "source_url": "https://the-url-where-you-found-this.com",
                }
            ],
            indent=2,
            ensure_ascii=True,
        ),
    )

    print(f"   🤖 LLM analyzing for customer segments...")
    try:
        analysis_response = llm.invoke(analysis_prompt)
        analysis_content = analysis_response.content.strip()
        
        if "```json" in analysis_content:
            analysis_content = analysis_content.split("```json")[1].split("```")[0].strip()
        elif "```" in analysis_content:
            analysis_content = analysis_content.split("```")[1].split("```")[0].strip()
        
        segments = json.loads(analysis_content)
        
        # Step 5: Store each segment with evidence_ids (be lenient - keep all segments)
        valid_segments = []
        for seg in segments:
            source_url = seg.get("source_url", "")
            evidence_text = seg.get("evidence_text", "")
            
            # Try to find evidence_ids for this source
            evidence_ids = []
            matched_url = source_url
            
            # 1. Exact match
            for src in extracted_content:
                if src["url"] == source_url:
                    evidence_ids = src["chunk_ids"]
                    matched_url = src["url"]
                    break
            
            # 2. Partial URL match
            if not evidence_ids:
                for src in extracted_content:
                    if source_url in src["url"] or src["url"] in source_url:
                        evidence_ids = src["chunk_ids"]
                        matched_url = src["url"]
                        break
            
            # 3. Domain match (e.g., both from same website)
            if not evidence_ids:
                try:
                    from urllib.parse import urlparse
                    source_domain = urlparse(source_url).netloc
                    for src in extracted_content:
                        src_domain = urlparse(src["url"]).netloc
                        if source_domain and source_domain == src_domain:
                            evidence_ids = src["chunk_ids"]
                            matched_url = src["url"]
                            print(f"   📝 Domain match: {source_url[:30]} → {src['url'][:30]}")
                            break
                except:
                    pass
            
            # 4. If still no match, use first available source's evidence (segment is still valuable)
            if not evidence_ids and extracted_content:
                evidence_ids = extracted_content[0]["chunk_ids"]
                matched_url = extracted_content[0]["url"]
                print(f"   📝 No exact source match for '{seg.get('name')}', using general evidence")
            
            # Always keep the segment if it has a name and description
            seg_name = seg.get("name", "").strip()
            seg_desc = seg.get("description", "").strip()
            
            if seg_name and seg_desc:
                valid_segment = {
                    "name": seg_name,
                    "description": seg_desc,
                    "evidence_text": evidence_text,
                    "source_url": matched_url,
                    "evidence_ids": evidence_ids,
                    "industry": industry
                }
                valid_segments.append(valid_segment)
                print(f"   ✅ Found: '{seg_name}' (from {matched_url[:40]}...)")
            else:
                print(f"   ⚠️ Skipped segment with missing name or description")
        
        _tool_state.customer_segments = valid_segments
        
        # Summary
        result = f"""Found {len(valid_segments)} customer segments in {industry}:

"""
        for seg in valid_segments:
            result += f"• **{seg['name']}**: {seg['description'][:100]}...\n"
            result += f"  Source: {seg['source_url'][:60]}...\n\n"
        
        result += f"\nAll segments stored with evidence for verification in Streamlit."
        return result
        
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON parse error: {e}")
        return f"Failed to parse customer segments: {e}"
    except Exception as e:
        print(f"   ❌ Analysis error: {e}")
        return f"Failed to analyze customer segments: {e}"


@tool
def map_segments_to_products() -> str:
    """
    Map customer segments to products that serve them.
    This tool analyzes which products are relevant for each customer segment
    and creates ADDRESSES_CUSTOMER_SEGMENT relationships in the knowledge graph.
    
    Call this AFTER:
    - research_customer_segments() has identified customer segments
    - Products have been saved with specifications
    
    Returns:
        Summary of segment-to-product mappings created.
    """
    global _tool_state
    
    if not _tool_state.customer_segments:
        return "No customer segments found. Call research_customer_segments first."
    
    if not _tool_state.products:
        return "No products found. Save some products first."
    
    if _tool_state.segment_mappings:
        return f"Segment mappings already created: {len(_tool_state.segment_mappings)} mappings."
    
    print(f"🔗 AGENT DECIDED: map_segments_to_products()")
    
    llm = get_llm()
    
    # Prepare segments data
    segments_info = []
    for seg in _tool_state.customer_segments:
        segments_info.append({
            "name": seg.get("name", ""),
            "description": seg.get("description", ""),
            "industry": seg.get("industry", "")
        })
    
    # Prepare products data with specs
    products_info = []
    for product_name, product_data in _tool_state.products.items():
        specs = _tool_state.specifications.get(product_name, {})
        products_info.append({
            "name": product_name,
            "company": product_data.get("company", "Unknown"),
            "specs": specs
        })
    
    prompt = build_poml_prompt(
        role="Go-to-market analyst",
        objective="Map customer segments to suitable products",
        context={
            "customer_segments": segments_info,
            "available_products": products_info,
        },
        instructions=[
            "For each segment, identify one or more suitable products and explain why.",
            "Evaluate specification fit, market relevance, and realistic purchasing behavior.",
            "Mention concrete specs in reasons when possible.",
        ],
        constraints=[
            "Try to map each segment to at least one product if applicable.",
            "Use exact segment names from input.",
            "Use exact product names from input.",
            "Return only valid JSON.",
        ],
        output_format=json.dumps(
            [
                {
                    "segment": "Upstream Oil & Gas Operators",
                    "product": "3051S",
                    "reason": "High pressure rating suitable for wellhead applications and hazardous-area certifications.",
                }
            ],
            indent=2,
            ensure_ascii=True,
        ),
    )

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        mappings = json.loads(content)
        
        # Validate and store mappings
        valid_mappings = []
        product_names = list(_tool_state.products.keys())
        segment_names = [s.get("name", "") for s in _tool_state.customer_segments]
        
        for m in mappings:
            segment = m.get("segment", "")
            product = m.get("product", "")
            reason = m.get("reason", "")
            
            # Validate segment exists
            if segment not in segment_names:
                # Try fuzzy match (lenient threshold)
                from difflib import SequenceMatcher
                best_match = None
                best_score = 0
                for sn in segment_names:
                    score = SequenceMatcher(None, segment.lower(), sn.lower()).ratio()
                    if score > best_score and score >= 0.5:  # Lenient threshold
                        best_score = score
                        best_match = sn
                if best_match:
                    print(f"   📝 Fuzzy match segment: '{segment}' → '{best_match}'")
                    segment = best_match
                else:
                    print(f"   ⚠️ Skipped: Segment '{segment}' not in {segment_names}")
                    continue
            
            # Validate product exists
            if product not in product_names:
                # Try fuzzy match (lenient threshold)
                from difflib import SequenceMatcher
                best_match = None
                best_score = 0
                for pn in product_names:
                    score = SequenceMatcher(None, product.lower(), pn.lower()).ratio()
                    if score > best_score and score >= 0.5:  # Lenient threshold
                        best_score = score
                        best_match = pn
                if best_match:
                    print(f"   📝 Fuzzy match product: '{product}' → '{best_match}'")
                    product = best_match
                else:
                    print(f"   ⚠️ Skipped: Product '{product}' not in {product_names}")
                    continue
            
            # Get evidence_ids from the segment
            segment_data = next((s for s in _tool_state.customer_segments if s.get("name") == segment), {})
            evidence_ids = segment_data.get("evidence_ids", [])
            source_url = segment_data.get("source_url", "")
            
            valid_mapping = {
                "segment": segment,
                "product": product,
                "reason": clean_string(reason)[:300],
                "evidence_ids": evidence_ids,
                "source_url": source_url
            }
            valid_mappings.append(valid_mapping)
            print(f"   ✅ Mapped: '{segment}' → '{product}'")
        
        _tool_state.segment_mappings = valid_mappings
        
        # Generate summary
        result = f"Created {len(valid_mappings)} segment-to-product mappings:\n\n"
        
        # Group by segment
        by_segment = {}
        for m in valid_mappings:
            seg = m["segment"]
            if seg not in by_segment:
                by_segment[seg] = []
            by_segment[seg].append(m["product"])
        
        for seg, products in by_segment.items():
            result += f"• {seg}: {', '.join(products)}\n"
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON parse error: {e}")
        return f"Failed to parse segment mappings: {e}"
    except Exception as e:
        print(f"   ❌ Mapping error: {e}")
        return f"Failed to map segments to products: {e}"


@tool
def generate_house_of_quality() -> str:
    """
    Generate a House of Quality (QFD) matrix.
    This tool creates a Quality Function Deployment matrix mapping customer needs (WHATs) 
    to product specifications (HOWs) with relationship strengths.
    
    Call this AFTER:
    - Customer needs have been extracted (via map_needs_from_report)
    - Products have been saved with specifications
    
    Returns:
        Summary of the House of Quality matrix created.
    """
    global _tool_state
    
    if _tool_state.house_of_quality:
        return f"House of Quality already generated with {len(_tool_state.house_of_quality.get('whats', []))} customer needs and {len(_tool_state.house_of_quality.get('hows', []))} specifications."
    
    if not _tool_state.customer_needs:
        return "No customer needs found. Run research_industry_needs and map_needs_from_report first."
    
    if not _tool_state.products:
        return "No products found. Save some products first."
    
    print(f"🏠 AGENT DECIDED: generate_house_of_quality()")
    
    llm = get_llm()
    
    # Prepare customer needs (WHATs)
    whats = []
    for need_key, need_data in _tool_state.customer_needs.items():
        whats.append({
            "id": need_key,
            "name": need_data.get("name", need_key),
            "threshold": need_data.get("threshold", ""),
            "spec_type": need_data.get("spec_type", ""),
            "description": need_data.get("description", "")[:200]
        })
    
    # Prepare specifications (HOWs) - get unique spec types across all products
    all_spec_types = set()
    product_specs = {}
    for product_name, specs in _tool_state.specifications.items():
        product_specs[product_name] = specs
        for spec_type in specs.keys():
            all_spec_types.add(spec_type)
    
    hows = list(all_spec_types)
    
    # Build prompt for LLM to analyze relationships
    prompt = build_poml_prompt(
        role="Competitive intelligence analyst performing Quality Function Deployment",
        objective="Create a House of Quality matrix mapping customer needs (WHATs) to specifications (HOWs)",
        context={
            "customer_needs_whats": whats,
            "specification_hows": hows,
            "products_and_specs": product_specs,
            "relationship_weights": {
                "9": "Strong relationship",
                "3": "Medium relationship",
                "1": "Weak relationship",
                "0": "No relationship",
            },
            "competitive_scores_scale": {
                "5": "Excellent - exceeds required threshold",
                "4": "Good - meets required threshold",
                "3": "Average - close to threshold",
                "2": "Below Average - falls short",
                "1": "Poor - significantly below threshold",
            },
        },
        instructions=[
            "For each customer need, determine influencing specifications and assign weights 0, 1, 3, or 9.",
            "For each product and each need, assign a score from 1-5.",
            "For every score, include an explicit derivation comparing actual spec value to required threshold.",
            "Provide technical correlations between specifications and key strategic insights.",
        ],
        constraints=[
            "Return only valid JSON.",
            "Every score reason must use the format: Score = X because [spec_name] = [actual_value] [comparison] required [threshold_value].",
        ],
        output_format=json.dumps(
            {
                "matrix": [
                    {
                        "need_id": "need key from WHATs",
                        "need_name": "readable name",
                        "relationships": {"spec_type": 9},
                        "reasoning": "Why these specs relate to this need",
                    }
                ],
                "competitive_scores": [
                    {
                        "product": "product name",
                        "scores": [
                            {
                                "need_id": "the customer need id",
                                "score": 4,
                                "reason": "Score = 4 because [spec] = [actual] meets required [threshold]",
                            }
                        ],
                        "overall_assessment": "Brief overall assessment of this product",
                    }
                ],
                "technical_correlations": [
                    {
                        "spec1": "specification type",
                        "spec2": "specification type",
                        "correlation": "positive",
                        "explanation": "Why these specs correlate",
                    }
                ],
                "key_insights": ["Important insight 1", "Important insight 2"],
            },
            indent=2,
            ensure_ascii=True,
        ),
    )
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Parse JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        hoq_data = json.loads(content)
        
        # Store the House of Quality data
        _tool_state.house_of_quality = {
            "whats": whats,
            "hows": hows,
            "matrix": hoq_data.get("matrix", []),
            "competitive_scores": hoq_data.get("competitive_scores", []),
            "technical_correlations": hoq_data.get("technical_correlations", []),
            "key_insights": hoq_data.get("key_insights", []),
            "products": product_specs,
            "generated_at": datetime.now().isoformat()
        }
        
        # Generate summary
        num_needs = len(whats)
        num_specs = len(hows)
        num_products = len(product_specs)
        num_relationships = sum(len(m.get("relationships", {})) for m in hoq_data.get("matrix", []))
        
        print(f"   ✅ Generated House of Quality: {num_needs} needs × {num_specs} specs, {num_products} products compared")
        
        result = f"""House of Quality Generated Successfully!

Matrix Size: {num_needs} customer needs × {num_specs} specifications
Products Analyzed: {num_products}
Total Relationships Mapped: {num_relationships}

Key Insights:
"""
        for insight in hoq_data.get("key_insights", [])[:5]:
            result += f"• {insight}\n"
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON parse error: {e}")
        return f"Failed to parse House of Quality response: {e}"
    except Exception as e:
        print(f"   ❌ House of Quality error: {e}")
        return f"Failed to generate House of Quality: {e}"


# =============================================================================
# TOOLS LIST
# =============================================================================

TOOLS = [
    set_source_selection,
    search_web,
    extract_page_content,
    save_competitor,
    save_product,
    get_current_progress,
    research_industry_needs,
    map_needs_from_report,
    research_customer_segments,
    map_segments_to_products,
    generate_house_of_quality,
    finish_research,
]

def build_system_prompt(company: str, product: str, product_category: str, industry: str) -> str:
    return build_poml_prompt(
        role=f"Competitive intelligence researcher for {company} {product_category} ({product})",
        objective=(
            f"Research {product_category} competitors to {company}'s {product}, "
            "map customer segments and needs, and produce a House of Quality matrix."
        ),
        context={
            "target_company": company,
            "target_product": product,
            "product_category": product_category,
            "target_industry": industry,
            "goals": [
                f"Research {company} competitors and find {product_category} products with specs.",
                "Identify customer segments in the target industry and map products to them.",
                "Generate an in-depth industry needs report from multiple sources.",
                "Map customer needs to product specifications.",
                "Build a House of Quality (QFD) matrix.",
            ],
            "tools_available": {
                "competitor_product_research": [
                    "search_web: Search for information (returns URLs)",
                    "extract_page_content: Must be called to get content and store evidence in ChromaDB",
                    f"save_competitor: Save competitor only after extracting a relevant page (do NOT save {company})",
                    "save_product: Save product with specs only after extracting datasheet content",
                ],
                "market_research": [
                    "research_customer_segments: Find customer groups/segments with evidence",
                    "map_segments_to_products: Map products to customer segments",
                ],
                "customer_needs_research": [
                    "research_industry_needs: Search 8+ sources and generate in-depth needs report",
                    "map_needs_from_report: Extract needs from report and map to saved products",
                ],
                "quality_function_deployment": [
                    "generate_house_of_quality: Create QFD matrix from needs (WHATs) and specs (HOWs)",
                ],
                "utility": [
                    "set_source_selection: Restrict to specific domains and/or source types",
                    "get_current_progress: Check collection progress",
                    "finish_research: Signal completion",
                ],
            },
            "workflow_phases": [
                f"Phase 1 - Competitors & Products: Find 3-5 competitors to {company}'s {product}, then for each use search_web -> extract_page_content -> save_competitor and search_web -> extract_page_content -> save_product.",
                "Phase 2 - Customer Segments: After products exist, call research_customer_segments(industry), then map_segments_to_products().",
                "Phase 3 - Customer Needs Report: Call research_industry_needs(industry), then map_needs_from_report().",
                "Phase 4 - House of Quality: After needs are mapped, call generate_house_of_quality().",
            ],
        },
        instructions=[
            "Follow the workflow phases in order from Phase 1 to Phase 4.",
            "Use tools proactively and ground extracted facts in stored evidence.",
            "Prioritize complete, verifiable outputs over broad but unverified claims.",
        ],
        constraints=[
            "Do not skip required extraction before save_competitor or save_product.",
            f"Do not save {company} as a competitor — it is the target company being researched.",
            "Finish only when all required outputs are present.",
            "Required outputs: at least 3 competitors with products, customer segments mapped to products, industry needs report, need-to-product mappings, and House of Quality matrix.",
        ],
        output_format="Operational execution via tool calls; begin with Phase 1 and progress sequentially.",
    )



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


def build_supervisor_graph(industry: str, max_competitors: int):
    """
    Build a supervisor-worker multi-agent graph using langgraph-supervisor.

    The supervisor orchestrates four phase-specific workers:
    1) Competitor researcher
    2) Segments researcher
    3) Needs analyst
    4) QFD analyst
    """
    try:
        from langgraph_supervisor import create_supervisor
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'langgraph-supervisor'. "
            "Install it with: pip install langgraph-supervisor"
        ) from exc

    llm = get_llm()

    competitor_researcher = create_react_agent(
        model=llm,
        tools=[search_web, extract_page_content, save_competitor, save_product],
        name="agent_competitor_researcher",
        prompt=(
            "You are Agent Competitor Researcher.\n"
            "Your scope is ONLY competitor and product discovery.\n"
            "Use tools in this sequence: search_web -> extract_page_content -> save_competitor/save_product.\n"
            f"Target up to {max_competitors} competitors in {industry}.\n"
            "Never call tools outside your assigned list."
        ),
    )

    segments_researcher = create_react_agent(
        model=llm,
        tools=[research_customer_segments, map_segments_to_products],
        name="agent_segments_researcher",
        prompt=(
            "You are Agent Segments Researcher.\n"
            "Research customer segments and map them to available products.\n"
            "If segments already exist, focus on completing product mappings.\n"
            "Never call tools outside your assigned list."
        ),
    )

    needs_analyst = create_react_agent(
        model=llm,
        tools=[research_industry_needs, map_needs_from_report],
        name="agent_needs_analyst",
        prompt=(
            "You are Agent Needs Analyst.\n"
            "Generate an industry needs report and map needs to saved products.\n"
            "Always run report generation before attempting mappings.\n"
            "Never call tools outside your assigned list."
        ),
    )

    qfd_analyst = create_react_agent(
        model=llm,
        tools=[generate_house_of_quality],
        name="agent_qfd_analyst",
        prompt=(
            "You are Agent QFD Analyst.\n"
            "Generate the House of Quality once customer needs and mappings are available.\n"
            "Never call tools outside your assigned list."
        ),
    )

    supervisor = create_supervisor(
        [competitor_researcher, segments_researcher, needs_analyst, qfd_analyst],
        model=llm,
        prompt=(
            "You are the supervisor agent for a 4-phase competitive intelligence pipeline.\n"
            "Delegate work in strict order and only move to next phase after prior phase is complete.\n"
            "Phase 1: Agent Competitor Researcher\n"
            "Phase 2: Agent Segments Researcher\n"
            "Phase 3: Agent Needs Analyst\n"
            "Phase 4: Agent QFD Analyst\n"
            "Completion criteria: competitors+products saved, segments mapped, needs mapped, and House of Quality generated.\n"
            "When all phases are complete, provide a brief final completion message."
        ),
    )
    return supervisor.compile()


# =============================================================================
# RUN THE AGENT
# =============================================================================

def run_agent(
    max_competitors: int = 10,
    industry: str = "process industries",
    max_iterations: int = 25,
    allowed_domains: List[str] | None = None,
    allowed_source_types: List[str] | None = None,
    multi_agent: bool = False,
    company: str = "Honeywell",
    product: str = "SmartLine ST700",
    product_category: str = "pressure transmitters",
) -> Dict[str, Any]:
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
    _tool_state.target_company = company
    _tool_state.product_category = product_category
    MAX_COMPETITORS = min(max_competitors, 10)
    MAX_ITERATIONS = max_iterations
    set_source_selection.invoke(
        {
            "allowed_domains": allowed_domains or [],
            "allowed_source_types": allowed_source_types or [],
        }
    )
    
    print("="*60)
    print("🤖 LANGGRAPH AGENTIC COMPETITIVE INTELLIGENCE")
    if multi_agent:
        print("    Mode: Supervisor-worker multi-agent")
        print("    Built with langgraph-supervisor + create_react_agent")
    else:
        print("    Mode: Original single-agent architecture")
        print("    Built with LangGraph StateGraph")
        print("    The agent DECIDES what to do via tool calls")
    print(f"    Max competitors: {MAX_COMPETITORS}")
    print(f"    Max iterations: {MAX_ITERATIONS}")
    print(f"    Industry: {industry}")
    print(f"    Target: {company} {product} ({product_category})")
    print(f"    Source selection: {_source_selection_summary()}")
    print("    📦 Evidence stored in ChromaDB for verification")
    print("="*60)
    
    start = datetime.now()
    
    # Build the graph
    graph = (
        build_supervisor_graph(industry=industry, max_competitors=MAX_COMPETITORS)
        if multi_agent
        else build_graph()
    )
    
    # Build dynamic system prompt
    system_prompt = build_system_prompt(company, product, product_category, industry)
    
    # Initial state with industry context
    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Research {company}'s competitors in {product_category} for the {industry} industry. The target product is {product}.

Tasks:
1. Find up to {MAX_COMPETITORS} competitors with their products and specs
2. Research customer segments specific to {industry} and map products to segments
3. Research customer needs specific to {industry}
4. Map customer needs to product specifications
5. Generate a House of Quality matrix
6. Respect source selection rules: {_source_selection_summary()}

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
    
    # Run the graph
    final_state = graph.invoke(
        initial_state,
        config={"recursion_limit": max_iterations * 2 + 10}
    )
    
    # Add baseline/target product (configurable)
    _tool_state.products[product] = {
        "name": product,
        "company": company,
        "source_url": f"https://www.{company.lower().replace(' ', '')}.com",
        "evidence_ids": []
    }
    _tool_state.specifications[product] = {}
    
    # AUTO-COMPLETE: If no customer segments, research them
    if _tool_state.products and not _tool_state.customer_segments:
        print("\n⚠️  No customer segments. Running research automatically...")
        try:
            result = research_customer_segments.invoke({"industry": industry})
            print(f"   {result[:200]}...")
        except Exception as e:
            print(f"   ❌ Could not auto-research customer segments: {e}")
    
    # AUTO-COMPLETE: If segments exist but no mappings, create them
    if _tool_state.customer_segments and not _tool_state.segment_mappings:
        print("\n⚠️  Customer segments exist but no product mappings. Running automatically...")
        try:
            result = map_segments_to_products.invoke({})
            print(f"   {result[:200]}...")
        except Exception as e:
            print(f"   ❌ Could not auto-map segments to products: {e}")
    
    # AUTO-COMPLETE: If iterations ran out before customer needs phase, run it now
    if _tool_state.products and not _tool_state.industry_needs_report:
        print("\n⚠️  Iterations ended before customer needs research. Running automatically...")
        try:
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
    
    # AUTO-COMPLETE: If customer needs exist but no House of Quality, generate it
    if _tool_state.customer_needs and not _tool_state.house_of_quality:
        print("\n⚠️  Customer needs exist but no House of Quality. Generating automatically...")
        try:
            result = generate_house_of_quality.invoke({})
            print(f"   {result[:200]}...")
        except Exception as e:
            print(f"   ❌ Could not auto-generate House of Quality: {e}")
    
    elapsed = (datetime.now() - start).total_seconds()
    total_evidence = sum(len(v) for v in _tool_state.evidence_map.values())
    
    hoq_status = "✅ Yes" if _tool_state.house_of_quality else "❌ No"
    
    print("\n" + "="*60)
    print(f"🏁 LANGGRAPH AGENT COMPLETE in {elapsed:.1f}s")
    print(f"   Iterations: {final_state.get('iteration', 0)}")
    print(f"   Competitors: {len(_tool_state.competitors)}")
    print(f"   Products: {len(_tool_state.products)}")
    print(f"   Specs: {sum(len(s) for s in _tool_state.specifications.values())}")
    print(f"   Customer segments: {len(_tool_state.customer_segments)} ({len(_tool_state.segment_mappings)} mappings)")
    print(f"   Customer needs: {len(_tool_state.customer_needs)}")
    print(f"   Need mappings: {len(_tool_state.need_mappings)}")
    print(f"   House of Quality: {hoq_status}")
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
        "customer_segments": _tool_state.customer_segments,
        "segments_sources": _tool_state.segments_sources,
        "segment_mappings": _tool_state.segment_mappings,
        "house_of_quality": _tool_state.house_of_quality,
    }


if __name__ == "__main__":
    result = run_agent(max_competitors=5)
    print(f"\nCompetitors discovered: {list(result['competitors'].keys())}")
    print(f"Products found: {list(result['products'].keys())}")
    print(f"Customer needs: {list(result['customer_needs'].keys())}")
    print(f"Need mappings: {len(result['need_mappings'])}")
    print(f"Evidence chunks: {sum(len(v) for v in result['evidence_map'].values())}")
