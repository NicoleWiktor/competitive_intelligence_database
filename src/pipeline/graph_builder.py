from __future__ import annotations

import json
import operator
from typing import Annotated, Any, Dict, List, TypedDict
from pathlib import Path

from langgraph.graph import START, StateGraph

from src.pipeline.query_node import search_node
from src.pipeline.extract_node import extract_node
from src.pipeline.llm_node import llm_state_node
from src.pipeline.neo4j_write_node import write_node
from src.pipeline.refine_query_node import refine_query_node


class PipelineState(TypedDict, total=False):
    query: str
    original_query: str
    max_results: int
    schema: Dict[str, Any]
    # Results list gets replaced each iteration (new search results)
    results: List[Dict[str, Any]]
    # Data dict gets merged/accumulated across iterations
    data: Dict[str, Any]
    needs_refinement: bool
    iteration: int
    max_iterations: int
    max_products_per_company: int  # Limit products per competitor
    max_competitors: int  # Limit number of competitors to track
    competitor_search_attempts: Dict[str, int]  # Track search attempts per competitor

    


def should_continue(state: PipelineState) -> str:
    """
    LangGraph conditional edge: Decide whether to continue refining or write to Neo4j.
    This runs AFTER refine_query_node has analyzed the data.
    
    Returns:
        "search" if more data is needed (loop back to search)
        "write" if data is complete (proceed to Neo4j)
    """
    if state.get("needs_refinement", False):
        return "search"
    else:
        return "write"


def build_graph() -> Any:
    """
    Build the LangGraph pipeline with iterative query refinement.
    
    Graph Flow:
        START -> search -> extract -> llm -> refine -> [conditional]
                                                        ├─> search (loop)
                                                        └─> write (terminal)
    
    State Updates (following LangGraph conventions):
        - search_node: updates 'results'
        - extract_node: updates 'results' (enriched with raw_content)
        - llm_state_node: updates 'data'
        - refine_query_node: updates 'query', 'needs_refinement', 'iteration'
        - write_node: no updates (terminal node)
    """
    g = StateGraph(PipelineState)
    
    # Add all nodes
    g.add_node("search", search_node)
    g.add_node("extract", extract_node)
    g.add_node("llm", llm_state_node)
    g.add_node("refine", refine_query_node)
    g.add_node("write", write_node)
    
    # Linear flow: START -> search -> extract -> llm -> refine
    g.add_edge(START, "search")
    g.add_edge("search", "extract")
    g.add_edge("extract", "llm")
    g.add_edge("llm", "refine")
    
    # Conditional: after refine analyzes data, either loop back or write
    g.add_conditional_edges(
        "refine",
        should_continue,
        {"search": "search", "write": "write"}
    )
    
    return g.compile()




if __name__ == "__main__":
    app = build_graph()
    # Put your Tavily query here:
    QUERY = "pressure transmitters in process industries with customer reviews and brands"
    # Limit results to avoid context overflow (each result can be large)
    MAX_RESULTS = 2  # Reduced to 2 to stay within context limits
    MAX_ITERATIONS = 18  # Max number of query refinement loops (Phase 1: ~2, Phase 2: ~15 for 5 competitors × 3 attempts)
    MAX_COMPETITORS = 5  # Focus on top 5 competitors only
    MAX_PRODUCTS_PER_COMPANY = 3  # Limit products per competitor to avoid clutter
    
    # Load schema from file
    schema_path = Path("src/schemas/schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    
    state: PipelineState = {
        "query": QUERY,
        "original_query": QUERY,
        "max_results": MAX_RESULTS,
        "schema": schema,
        "iteration": 0,
        "max_iterations": MAX_ITERATIONS,
        "max_competitors": MAX_COMPETITORS,
        "max_products_per_company": MAX_PRODUCTS_PER_COMPANY,
        "competitor_search_attempts": {},  # Track attempts per competitor
    }
    
    # Set recursion limit: each iteration = 4 nodes (search, extract, llm, refine)
    # With 18 max iterations: 4 * 18 = 72, set to 80 for safety
    out = app.invoke(state, {"recursion_limit": 80})
    print("\n" + "="*80)
    print("FINAL EXTRACTED DATA:")
    print("="*80)
    print(json.dumps(out.get("data", {}), indent=2))
