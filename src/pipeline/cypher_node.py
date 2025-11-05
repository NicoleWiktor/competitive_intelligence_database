"""
Cypher Node - Converts extracted JSON data into Neo4j Cypher statements.

This module transforms the structured data (companies, products, prices, relationships)
into Cypher MERGE statements that create the knowledge graph in Neo4j.

Each node and relationship stores source_urls (array) showing which web pages
mentioned that entity, enabling verification and fact-checking.
"""

from __future__ import annotations

from typing import Dict, List


def _esc(s: str) -> str:
    """Escape special characters for Cypher queries."""
    return (s or "").replace("\\", "\\\\").replace("'", "\\'")


def to_merge_cypher(data: Dict) -> str:
    """
    Convert JSON data to Cypher MERGE statements for Neo4j.
    
    Process:
    1. Create constraints (ensure unique node names)
    2. Create nodes (Company, Product, Price) with source_urls
    3. Create relationships (COMPETES_WITH, OFFERS_PRODUCT, HAS_PRICE) with source_urls
    
    Source URL Tracking:
    - Each node has source_urls array (which pages mentioned it)
    - Each relationship has source_urls array (which pages mentioned the connection)
    - ON CREATE: Initialize with current URL
    - ON MATCH: Append current URL if not already present
    
    Args:
        data: Dictionary with Relationships and Doc keys
        
    Returns:
        String of Cypher statements separated by semicolons
    """
    if not data:
        return ""
    
    stmts: List[str] = []
    
    # Step 1: Create constraints for unique node names
    stmts.extend([
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Company) REQUIRE n.name IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Product) REQUIRE n.name IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Price) REQUIRE n.name IS UNIQUE;",
    ])
    
    # Step 2: Extract source URL from data
    raw_url = (data.get("Doc") or {}).get("source_url") or ""
    if isinstance(raw_url, list):
        doc_url = _esc(raw_url[-1] if raw_url else "")  # Use most recent URL
    else:
        doc_url = _esc(raw_url)
    
    # Step 3: Process each relationship to create nodes and edges
    for rel in data.get("Relationships", []) or []:
        # Get relationship details
        relationship_type = _esc((rel.get("relationship") or "").upper().replace(" ", "_"))
        source_label = rel.get("source_type") or "Company"
        target_label = rel.get("target_type") or "Product"
        source_name = _esc(rel.get("source") or "")
        target_name = _esc(rel.get("target") or "")
        
        # Get specific source URL for this relationship (fallback to doc_url)
        rel_url = _esc(rel.get("source_url") or doc_url)
        
        # Skip if missing required fields
        if not (source_name and target_name and relationship_type):
            continue
        
        # Normalize label names
        if source_label == "Brand":
            source_label = "Company"
        if target_label == "Brand":
            target_label = "Company"
        
        # Create source node with source_urls tracking
        stmts.append(
            f"MERGE (s:{source_label} {{name: '{source_name}'}}) "
            f"ON CREATE SET s.source_urls = ['{rel_url}'] "
            f"ON MATCH SET s.source_urls = CASE WHEN '{rel_url}' IN s.source_urls THEN s.source_urls ELSE s.source_urls + '{rel_url}' END;"
        )
        
        # Create target node with source_urls tracking
        stmts.append(
            f"MERGE (t:{target_label} {{name: '{target_name}'}}) "
            f"ON CREATE SET t.source_urls = ['{rel_url}'] "
            f"ON MATCH SET t.source_urls = CASE WHEN '{rel_url}' IN t.source_urls THEN t.source_urls ELSE t.source_urls + '{rel_url}' END;"
        )
        
        # Create relationship with source_urls tracking
        stmts.append(
            f"MATCH (s:{source_label} {{name: '{source_name}'}}), (t:{target_label} {{name: '{target_name}'}}) "
            f"MERGE (s)-[rel:{relationship_type}]->(t) "
            f"ON CREATE SET rel.source_urls = ['{rel_url}'] "
            f"ON MATCH SET rel.source_urls = CASE WHEN '{rel_url}' IN rel.source_urls THEN rel.source_urls ELSE rel.source_urls + '{rel_url}' END;"
        )
    
    # Combine all statements
    cypher = "\n".join(stmts)
    
    # Debug output
    print(f"[cypher] Generated {len(stmts)} statements")
    print(f"[cypher] Preview: {cypher[:200]}...")
    
    return cypher
