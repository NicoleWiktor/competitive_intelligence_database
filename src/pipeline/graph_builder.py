"""
Graph Builder - Main pipeline orchestration using LangGraph.

This module defines the competitive intelligence extraction pipeline:
    START → search → extract → llm → refine → [decision]
              ↑                                   ↓
              └──────────── loop ─────────────────┘
                                                 ↓
                                               write

The pipeline iteratively searches the web, extracts data, and refines queries
until it collects: Honeywell → COMPETES_WITH → OFFERS_PRODUCT → HAS_PRICE → HAS_SPECIFICATION
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Dict, List, TypedDict
from pathlib import Path

from langgraph.graph import START, StateGraph

from src.pipeline.query_node import search_node
from src.pipeline.extract_node import extract_node
from src.pipeline.llm_node import llm_state_node
from src.pipeline.neo4j_write_node import write_node
from src.pipeline.refine_query_node import refine_query_node


class PipelineState(TypedDict, total=False):
    """
    Shared state that flows through all nodes in the LangGraph pipeline.
    
    Fields:
    - query: Current search query (changes each iteration)
    - original_query: Initial query (stays constant)
    - max_results: Number of URLs to fetch per search (default: 1)
    - schema: JSON schema defining extraction structure
    - results: Search results from current iteration (replaced each time)
    - data: Accumulated extracted data (merged across iterations)
    - needs_refinement: Boolean - continue iterating or stop
    - iteration: Current iteration counter
    - max_iterations: Safety limit to prevent infinite loops
    - max_competitors: Max number of competitors to collect (default: 5)
    - max_products_per_company: Max products per competitor (default: 1)
    - phase_attempts: Track attempts per phase {"competitors": 0, "products": {}, "prices": {}}
    """
    query: str
    original_query: str
    max_results: int
    schema: Dict[str, Any]
    results: List[Dict[str, Any]]
    data: Dict[str, Any]
    needs_refinement: bool
    iteration: int
    max_iterations: int
    max_products_per_company: int
    max_competitors: int
    phase_attempts: Dict[str, Any]


def should_continue(state: PipelineState) -> str:
    """
    Decision function: Determine next step after refine node.
    
    The refine node sets needs_refinement based on whether more data is needed.
    
    Returns:
        "search" - Loop back to search for more data
        "write" - All data collected, proceed to Neo4j write
    """
    if state.get("needs_refinement", False):
        return "search"  # Continue iteration
    else:
        return "write"  # Done - write to database


def build_graph() -> Any:
    """
    Build and compile the LangGraph pipeline.
    
    Graph structure:
    1. search: Query web using Tavily API
    2. extract: Get full page content from URLs
    3. llm: Extract structured data using GPT-4o-mini
    4. refine: Analyze what's missing and generate next query
    5. Decision: Loop back to search OR proceed to write
    6. write: Save to Neo4j database (terminal node)
    
    State updates by node:
    - search: Updates 'results' (new search results)
    - extract: Updates 'results' (enriched with raw_content)
    - llm: Updates 'data' (merged extractions)
    - refine: Updates 'query', 'needs_refinement', 'iteration', 'phase_attempts'
    - write: No updates (terminal)
    """
    graph = StateGraph(PipelineState)
    
    # Add nodes to graph
    graph.add_node("search", search_node)
    graph.add_node("extract", extract_node)
    graph.add_node("llm", llm_state_node)
    graph.add_node("refine", refine_query_node)
    graph.add_node("write", write_node)
    
    # Define linear flow through nodes
    graph.add_edge(START, "search")
    graph.add_edge("search", "extract")
    graph.add_edge("extract", "llm")
    graph.add_edge("llm", "refine")
    
    # Add conditional branch after refine
    graph.add_conditional_edges(
        "refine",
        should_continue,
        {
            "search": "search",  # Loop back
            "write": "write"     # Exit to database
        }
    )
    
    return graph.compile()


if __name__ == "__main__":
    """
    Main execution: Configure and run the pipeline.
    
    Configuration:
    - QUERY: Broad initial search query
    - MAX_RESULTS: URLs per search (1 = process one page at a time)
    - MAX_ITERATIONS: Safety limit (Phase 1: 8 + Phase 2: 5×7 + Phase 3: 5×7 + Phase 4: 5×5 ≈ 105)
    - MAX_COMPETITORS: Focus on top 5 competitors
    - MAX_PRODUCTS_PER_COMPANY: 1 product each
    """
    app = build_graph()
    
    # Configuration
    QUERY = "pressure transmitters in process industries with customer reviews and brands"
    MAX_RESULTS = 1  # Process 1 URL per iteration for control
    MAX_ITERATIONS = 110  # Sufficient for all 4 phases
    MAX_COMPETITORS = 5  # Top 5 competitors
    MAX_PRODUCTS_PER_COMPANY = 1  # 1 product each
    
    # Load schema
    schema_path = Path("src/schemas/schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    
    # Initialize state
    state: PipelineState = {
        "query": QUERY,
        "original_query": QUERY,
        "max_results": MAX_RESULTS,
        "schema": schema,
        "iteration": 0,
        "max_iterations": MAX_ITERATIONS,
        "max_competitors": MAX_COMPETITORS,
        "max_products_per_company": MAX_PRODUCTS_PER_COMPANY,
        "phase_attempts": {
            "competitors": 0,
            "products": {},  # Per-competitor tracking
            "prices": {},    # Per-product tracking
            "specs": {},      # Per-product tracking (Phase 4)
            "reviews": {}    # Phase 5
        },
    }
    
    # Run pipeline
    # Recursion limit: 4 nodes per iteration × 110 iterations = 440, set to 500 for safety
    print("="*80)
    print("STARTING COMPETITIVE INTELLIGENCE PIPELINE (5 PHASES)")
    print("="*80)
    
    result = app.invoke(state, {"recursion_limit": 500})
    
    # Print final results
    print("\n" + "="*80)
    print("FINAL EXTRACTED DATA")
    print("="*80)
    print(json.dumps(result.get("data", {}), indent=2))
