"""
Neo4j Write Node - Writes extracted data to Neo4j graph database.

This is the terminal node of the pipeline. It converts the accumulated
JSON data into Cypher statements and executes them against Neo4j.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from neo4j import GraphDatabase, Driver

from src.pipeline.cypher_node import to_merge_cypher
from src.config.settings import get_neo4j_config


def _get_driver() -> Optional[Driver]:
    """
    Create Neo4j database driver from environment configuration.
    
    Returns None if credentials not configured (allows running without DB).
    """
    cfg = get_neo4j_config()
    uri = cfg.get("uri")
    user = cfg.get("user")
    password = cfg.get("password")
    
    if not (uri and user and password):
        print("[neo4j] Neo4j credentials not set - skipping DB write")
        return None
    
    return GraphDatabase.driver(uri, auth=(user, password))


def run_neo4j(cypher: str) -> None:
    """
    Execute Cypher statements against Neo4j database.
    
    Splits statements on semicolons and executes each one.
    Shows preview of each statement in console.
    """
    if not cypher or not cypher.strip():
        print("[neo4j] No Cypher statements to execute")
        return
    
    driver = _get_driver()
    if not driver:
        return
    
    try:
        with driver.session() as session:
            print("[neo4j] Executing Cypher statements...")
            
            # Execute each statement
            for stmt in cypher.split(";"):
                stmt = stmt.strip()
                if stmt:
                    # Show preview (truncate long statements)
                    preview = stmt[:120] + ("..." if len(stmt) > 120 else "")
                    print(f"[neo4j] >> {preview}")
                    session.run(stmt)
            
            print("[neo4j] ✓ Write complete")
    finally:
        driver.close()


def write_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph terminal node: Write extracted data to Neo4j.
    
    This is called after the refine node decides all data is collected.
    It marks the end of the pipeline.
    
    Returns empty dict since this is a terminal node (no state updates needed).
    """
    data = state.get("data", {})
    
    # Convert JSON to Cypher
    cypher = to_merge_cypher(data)
    
    # Execute against Neo4j
    print(cypher)  # Full output for debugging
    run_neo4j(cypher)
    
    return {}  # Terminal node - no state updates
