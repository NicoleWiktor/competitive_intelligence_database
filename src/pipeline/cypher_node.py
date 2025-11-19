"""
Cypher Node - Converts extracted JSON data into Neo4j Cypher statements.

This module transforms the structured data (companies, products, prices, relationships)
into Cypher MERGE statements that create the knowledge graph in Neo4j.

Each node and relationship stores:
- source_urls (array): Which web pages mentioned this entity
- evidence_ids (array): ChromaDB chunk IDs linking to raw source evidence

This enables full traceability from graph relationships back to original source text.
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
    3. Create relationships with source_urls + evidence_ids (ChromaDB chunk IDs)
    
    Evidence Tracking:
    - Each node has source_urls array (which pages mentioned it)
    - Each relationship has:
        - source_urls array (which pages mentioned the connection)
        - evidence_ids array (ChromaDB chunk IDs for raw source verification)
    - ON CREATE: Initialize with current values
    - ON MATCH: Append if not already present
    
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
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Specification) REQUIRE n.name IS UNIQUE;",
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
        
        # Get evidence IDs (ChromaDB chunk IDs) for this relationship
        evidence_ids = rel.get("evidence_ids") or []
        # Escape and format as Cypher list
        evidence_ids_str = "[" + ", ".join([f"'{_esc(eid)}'" for eid in evidence_ids]) + "]"
        
        # Skip if missing required fields
        if not (source_name and target_name and relationship_type):
            continue
        
        # Normalize label names to ONLY valid Neo4j labels: Company, Product, Price, Specification
        # CRITICAL: Labels must be valid identifiers, NOT entity names
        
        # Normalize source label
        if source_label in ("Company", "Brand"):
            source_label = "Company"
        elif source_label == "Product":
            source_label = "Product"
        elif source_label == "Price":
            source_label = "Price"
        elif source_label == "Specification":
            source_label = "Specification"
        else:
            # Invalid or missing label - infer from relationship type
            if relationship_type == "COMPETES_WITH":
                source_label = "Company"  # Honeywell COMPETES_WITH → source is Company
            elif relationship_type == "OFFERS_PRODUCT":
                source_label = "Company"  # Company OFFERS_PRODUCT → source is Company
            elif relationship_type == "HAS_PRICE":
                source_label = "Product"  # Product HAS_PRICE → source is Product
            elif relationship_type == "HAS_SPECIFICATION":
                source_label = "Product"  # Product HAS_SPECIFICATION → source is Product
            else:
                # Default to Company for unknown relationships
                source_label = "Company"
        
        # Normalize target label
        if target_label in ("Company", "Brand"):
            target_label = "Company"
        elif target_label == "Product":
            target_label = "Product"
        elif target_label == "Price":
            target_label = "Price"
        elif target_label == "Specification":
            target_label = "Specification"
        else:
            # Invalid or missing label - infer from relationship type
            if relationship_type == "COMPETES_WITH":
                target_label = "Company"  # → COMPETES_WITH Company, target is Company
            elif relationship_type == "OFFERS_PRODUCT":
                target_label = "Product"  # → OFFERS_PRODUCT Product, target is Product
            elif relationship_type == "HAS_PRICE":
                target_label = "Price"  # → HAS_PRICE Price, target is Price
            elif relationship_type == "HAS_SPECIFICATION":
                target_label = "Specification"  # → HAS_SPECIFICATION Specification, target is Specification
            else:
                # Default to Product for unknown relationships
                target_label = "Product"
        
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
        
        # Create relationship with source_urls and evidence_ids tracking
        stmts.append(
            f"MATCH (s:{source_label} {{name: '{source_name}'}}), (t:{target_label} {{name: '{target_name}'}}) "
            f"MERGE (s)-[rel:{relationship_type}]->(t) "
            f"ON CREATE SET rel.source_urls = ['{rel_url}'], rel.evidence_ids = {evidence_ids_str} "
            f"ON MATCH SET rel.source_urls = CASE WHEN '{rel_url}' IN rel.source_urls THEN rel.source_urls ELSE rel.source_urls + '{rel_url}' END, "
            f"rel.evidence_ids = CASE WHEN SIZE(rel.evidence_ids) = 0 THEN {evidence_ids_str} ELSE rel.evidence_ids END;"
        )
    
    # Combine all statements
    cypher = "\n".join(stmts)
    
    # Debug output
    print(f"[cypher] Generated {len(stmts)} statements")
    print(f"[cypher] Preview: {cypher[:200]}...")
    
    return cypher
