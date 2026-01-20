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
        self.searched_queries: List[str] = []
        self.extracted_urls: List[str] = []
        self.evidence_map: Dict[str, List[str]] = {}
        self.finished: bool = False
    
    def summary(self) -> str:
        return f"""Current Progress:
- Competitors found: {len(self.competitors)} ({list(self.competitors.keys())[:5]})
- Products found: {len(self.products)}
- Specs collected: {sum(len(s) for s in self.specifications.values())}
- Searches done: {len(self.searched_queries)}
- Pages extracted: {len(self.extracted_urls)}
- Evidence chunks stored: {sum(len(v) for v in self.evidence_map.values())}"""


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
        
        return "Could not extract content from this URL."
    except Exception as e:
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
    Signal that research is complete. ONLY call this when you have products with specs
    for your competitors. Having competitors without products is useless!
    
    Args:
        reason: Why you're finishing (e.g., "Collected 5 competitors with products and specs")
    
    Returns:
        Final summary or warning if not enough products.
    """
    global _tool_state
    
    num_competitors = len(_tool_state.competitors)
    num_products = len(_tool_state.products)
    
    # Check if we have enough products
    if num_products < num_competitors and num_competitors > 0:
        return f"""⚠️ WARNING: You have {num_competitors} competitors but only {num_products} products!
        
You should have at least 1 product per competitor. Please:
1. Search for product datasheets for competitors without products
2. Extract the datasheets and save_product with specs
3. THEN call finish_research

Current status:
{_tool_state.summary()}

DO NOT finish yet - find more products first!"""
    
    _tool_state.finished = True
    print(f"🏁 AGENT DECIDED: finish_research('{reason}')")
    return f"RESEARCH COMPLETE: {reason}\n\n{_tool_state.summary()}"


# =============================================================================
# TOOLS LIST
# =============================================================================

TOOLS = [
    search_web,
    extract_page_content,
    save_competitor,
    save_product,
    get_current_progress,
    finish_research,
]


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a competitive intelligence researcher for Honeywell's pressure transmitter products (SmartLine ST700).

YOUR GOAL: Research Honeywell's competitors and find AT LEAST ONE PRODUCT WITH SPECS for each competitor.

 CRITICAL - DO NOT FINISH UNTIL YOU HAVE PRODUCTS:
- You MUST find at least one product for EACH competitor before calling finish_research
- We need products and specs with each competitor!
- Only call finish_research when you have products with specs for your competitors

TOOLS AVAILABLE:
- search_web: Search for information (returns URLs)
- extract_page_content: MUST CALL THIS to get content AND store evidence in ChromaDB
- save_competitor: Save a competitor (ONLY after extracting a page about them)
- save_product: Save a product with specs (ONLY after extracting the datasheet)
- get_current_progress: Check what you've collected - USE THIS to see if you have enough products
- finish_research: Signal you're done - ONLY call when you have products for competitors!

WORKFLOW - EXTRACT BEFORE SAVING:
1. search_web → get URLs
2. extract_page_content(url) → stores evidence in ChromaDB
3. save_competitor(name, source_url=the_url_you_just_extracted)
4. IMMEDIATELY search for that competitor's products!
5. extract_page_content(product_datasheet_url)
6. save_product(company, model, source_url=..., specs...)

STRATEGY (IMPORTANT - GET PRODUCTS!):
1. Search for 3-5 competitor companies 
2. For EACH competitor:
   a. Extract a page about them → save_competitor
   b. IMMEDIATELY search for their pressure transmitter products
   c. Extract a product datasheet → save_product with specs
3. Only move to next competitor AFTER you have at least 1 product for current one
4. Call finish_research ONLY when you have products for most competitors
5. If you cannot find products for a competitor after 2 tries, skip them and move to the next one, do not hallucinate or make up products.

SPEC REQUIREMENTS:
- Specs MUST have real numbers/units (e.g., "0-6000 psi", "±0.065%", "4-20mA HART")
- Don't save generic specs like "high accuracy" or "wide range"

START by finding 3-5 competitors, then GET PRODUCTS for each one!"""


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
    if state.get("iteration", 0) >= 30:
        print("   Max iterations reached, ending...")
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

def run_agent(max_competitors: int = 10) -> Dict[str, Any]:
    """
    Run the LangGraph agentic pipeline.
    
    The graph:
    1. Agent node calls LLM → LLM decides which tools to call
    2. Conditional edge routes to tools or end
    3. Tools node executes the tool calls
    4. Loop back to agent
    5. Repeat until agent calls finish_research or max iterations
    """
    global _tool_state, MAX_COMPETITORS
    
    # Reset tool state
    _tool_state = ToolState()
    MAX_COMPETITORS = min(max_competitors, 10)
    
    print("="*60)
    print("🤖 LANGGRAPH AGENTIC COMPETITIVE INTELLIGENCE")
    print("    Built with LangGraph StateGraph")
    print("    The agent DECIDES what to do via tool calls")
    print(f"    Max competitors: {MAX_COMPETITORS}")
    print("    📦 Evidence stored in ChromaDB for verification")
    print("="*60)
    
    start = datetime.now()
    
    # Build the graph
    graph = build_graph()
    
    # Initial state
    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Research Honeywell's competitors in pressure transmitters. Find up to {MAX_COMPETITORS} competitors with their products and specs. Start now.")
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
    final_state = graph.invoke(initial_state)
    
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
    
    elapsed = (datetime.now() - start).total_seconds()
    total_evidence = sum(len(v) for v in _tool_state.evidence_map.values())
    
    print("\n" + "="*60)
    print(f"🏁 LANGGRAPH AGENT COMPLETE in {elapsed:.1f}s")
    print(f"   Iterations: {final_state.get('iteration', 0)}")
    print(f"   Competitors: {len(_tool_state.competitors)}")
    print(f"   Products: {len(_tool_state.products)}")
    print(f"   Specs: {sum(len(s) for s in _tool_state.specifications.values())}")
    print(f"   Searches made: {len(_tool_state.searched_queries)}")
    print(f"   Pages extracted: {len(_tool_state.extracted_urls)}")
    print(f"   📦 Evidence chunks in ChromaDB: {total_evidence}")
    print("="*60)
    
    return {
        "competitors": _tool_state.competitors,
        "products": _tool_state.products,
        "specifications": _tool_state.specifications,
        "evidence_map": _tool_state.evidence_map,
    }


if __name__ == "__main__":
    result = run_agent(max_competitors=5)
    print(f"\nCompetitors discovered: {list(result['competitors'].keys())}")
    print(f"Products found: {list(result['products'].keys())}")
    print(f"Evidence chunks: {sum(len(v) for v in result['evidence_map'].values())}")
