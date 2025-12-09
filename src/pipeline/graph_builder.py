"""
Graph Builder - Main pipeline orchestration using LangGraph.

SUPPORTS TWO MODES:
1. AGENTIC MODE (New!): Uses ReAct agent for autonomous decision-making
2. PIPELINE MODE (Legacy): Uses fixed iteration pipeline

The agentic mode is the recommended approach for better results.
The AI makes decisions about what to search, when to extract specs, and when to stop.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Dict, List, TypedDict
from pathlib import Path
import argparse

from langgraph.graph import START, StateGraph

from src.pipeline.query_node import search_node
from src.pipeline.extract_node import extract_node
from src.pipeline.llm_node import llm_state_node
from src.pipeline.neo4j_write_node import write_node
from src.pipeline.refine_query_node import refine_query_node


class PipelineState(TypedDict, total=False):
    """Shared state for the LangGraph pipeline."""
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
    """Decision function for pipeline mode."""
    if state.get("needs_refinement", False):
        return "search"
    else:
        return "write"


def build_graph() -> Any:
    """Build the LangGraph pipeline for PIPELINE MODE."""
    graph = StateGraph(PipelineState)
    
    graph.add_node("search", search_node)
    graph.add_node("extract", extract_node)
    graph.add_node("llm", llm_state_node)
    graph.add_node("refine", refine_query_node)
    graph.add_node("write", write_node)
    
    graph.add_edge(START, "search")
    graph.add_edge("search", "extract")
    graph.add_edge("extract", "llm")
    graph.add_edge("llm", "refine")
    
    graph.add_conditional_edges(
        "refine",
        should_continue,
        {"search": "search", "write": "write"}
    )
    
    return graph.compile()


def run_pipeline_mode(
    query: str = "pressure transmitters process industries competitors specifications",
    max_results: int = 1,
    max_iterations: int = 50,
    max_competitors: int = 5,
    max_products_per_company: int = 1,
) -> Dict[str, Any]:
    """
    Run the traditional pipeline mode.
    
    This mode uses fixed phases to collect:
    1. Competitors
    2. Products
    3. Prices
    4. Specifications
    """
    app = build_graph()
    
    schema_path = Path("src/schemas/schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    
    state: PipelineState = {
        "query": query,
        "original_query": query,
        "max_results": max_results,
        "schema": schema,
        "iteration": 0,
        "max_iterations": max_iterations,
        "max_competitors": max_competitors,
        "max_products_per_company": max_products_per_company,
        "phase_attempts": {
            "competitors": 0,
            "products": {},
            "prices": {},
            "specs": {}
        },
    }
    
    print("="*80)
    print("STARTING PIPELINE MODE")
    print("="*80)
    
    result = app.invoke(state, {"recursion_limit": 500})
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    
    return result.get("data", {})


def run_agentic_mode(
    target_product: str = "SmartLine ST700",
    target_company: str = "Honeywell",
    max_competitors: int = 5,
    max_iterations: int = 30,
) -> Dict[str, Any]:
    """
    Run the AGENTIC mode using LangGraph agent pipeline.
    
    This is the RECOMMENDED mode for better results!
    
    The LangGraph agent:
    - Uses tool nodes for actions (search, save, extract)
    - Has conditional routing based on decisions
    - Maintains state across the graph
    - Makes autonomous decisions about what to do next
    
    Graph structure:
        __start__ → agent → router → tools → agent (loop) → __end__
    """
    from src.agents.langgraph_agent import run_langgraph_agent
    from src.pipeline.neo4j_write_node import run_neo4j
    from src.pipeline.cypher_node import to_merge_cypher
    
    print("="*80)
    print("🤖 STARTING LANGGRAPH AGENTIC MODE")
    print(f"Target: {target_company} {target_product}")
    print(f"Looking for {max_competitors} competitors")
    print("="*80)
    
    # Reset Neo4j database before running
    print("\n🗑️  Resetting Neo4j database...")
    reset_neo4j()
    print("✓ Database cleared\n")
    
    # Run the LangGraph agent
    data = run_langgraph_agent(
        max_iterations=max_iterations,
        max_competitors=max_competitors,
    )
    
    # Write to Neo4j
    cypher = to_merge_cypher(data)
    print("\n[neo4j] Writing agent results to database...")
    run_neo4j(cypher)
    
    print("\n" + "="*80)
    print("🏁 LANGGRAPH AGENTIC MODE COMPLETE")
    print("="*80)
    
    return data


def reset_neo4j():
    """Clear all nodes and relationships from Neo4j database."""
    from neo4j import GraphDatabase
    from src.config.settings import get_neo4j_config
    
    cfg = get_neo4j_config()
    uri = cfg.get("uri")
    user = cfg.get("user")
    password = cfg.get("password")
    
    if not (uri and user and password):
        print("[neo4j] Neo4j credentials not set - skipping reset")
        return
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            print("[neo4j] All nodes and relationships deleted")
    finally:
        driver.close()


def main():
    """Main entry point with mode selection."""
    parser = argparse.ArgumentParser(
        description="Competitive Intelligence Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in AGENTIC mode (recommended):
  python -m src.pipeline.graph_builder --mode agentic

  # Run in PIPELINE mode:
  python -m src.pipeline.graph_builder --mode pipeline

  # Custom parameters:
  python -m src.pipeline.graph_builder --mode agentic --competitors 3 --iterations 20
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["agentic", "pipeline"],
        default="agentic",
        help="Execution mode: 'agentic' (recommended) or 'pipeline'"
    )
    parser.add_argument(
        "--competitors",
        type=int,
        default=5,
        help="Number of competitors to find (default: 5)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Maximum iterations (default: 30 for agentic, 50 for pipeline)"
    )
    parser.add_argument(
        "--product",
        type=str,
        default="SmartLine ST700",
        help="Target Honeywell product (default: SmartLine ST700)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "agentic":
        result = run_agentic_mode(
            target_product=args.product,
            max_competitors=args.competitors,
            max_iterations=args.iterations,
        )
    else:
        result = run_pipeline_mode(
            max_competitors=args.competitors,
            max_iterations=args.iterations,
        )
    
    print("\n" + "="*80)
    print("FINAL EXTRACTED DATA")
    print("="*80)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
