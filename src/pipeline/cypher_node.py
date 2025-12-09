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
import hashlib


def _esc(s: str) -> str:
    """
    Escape text so it is safe inside a Cypher string literal:
    - Escape single quotes
    - Escape backslashes
    - Replace newlines with \n
    - Replace carriage returns with \r
    """
    if not s:
        return ""

    return (
        s.replace("\\", "\\\\")
         .replace("'", "\\'")
         .replace("\n", "\\n")
         .replace("\r", "\\r")
    )


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
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Specification) REQUIRE n.name IS UNIQUE;"
    ])

    # Build lookup table mapping company → product for long-text node naming
    product_lookup = {}
    for r in data.get("Relationships", []):
        if r.get("relationship") == "OFFERS_PRODUCT":
            comp = r.get("source")
            prod = r.get("target")
            if comp and prod:
                product_lookup[comp] = prod

    # Step 2: Extract source URL from data
    raw_url = (data.get("Doc") or {}).get("source_url") or ""
    if isinstance(raw_url, list):
        doc_url = _esc(raw_url[-1] if raw_url else "")
    else:
        doc_url = _esc(raw_url)

    # Step 3: Process each relationship
    for rel in data.get("Relationships", []) or []:
        raw_rel = rel.get("relationship", "").upper().replace(" ", "_")
        primary_rel = raw_rel.split("|")[0].strip()  # take only the first type

        if not primary_rel:
            continue  # skip malformed relationship

        relationship_type = primary_rel
        VALID_RELATIONSHIPS = {
            "COMPETES_WITH",
            "OFFERS_PRODUCT",
            "HAS_PRICE",
            "HAS_SPECIFICATION",
            "HAS_DESCRIPTION",
            "HAS_REVIEW",
            "HAS_SPEC_SHEET"
        }

        if relationship_type not in VALID_RELATIONSHIPS:
            continue

        source_label = rel.get("source_type") or "Company"
        target_label = rel.get("target_type") or "Product"

        # Fix invalid multi-label strings like "Company|Product"
        if "|" in source_label:
            source_label = source_label.split("|")[0].strip()

        if "|" in target_label:
            target_label = target_label.split("|")[0].strip()

        source_name = _esc(rel.get("source") or "")
        raw_target_value = rel.get("target") or ""
        target_text = _esc(raw_target_value)

        def shorten_text_id(text: str) -> str:
            return hashlib.md5(text.encode()).hexdigest()[:8]

        # Resolve product name for long-text nodes
        product_name = rel.get("source", "")
        if product_name in product_lookup:
            product_name = product_lookup[product_name]
        product_name = _esc(product_name)

        # Determine target_name
        if relationship_type == "HAS_DESCRIPTION":
            short_name = f"{product_name} Description"
            target_name = _esc(short_name)

        elif relationship_type == "HAS_SPEC_SHEET":
            short_name = f"{product_name} Spec Sheet"
            target_name = _esc(short_name)

        elif relationship_type == "HAS_REVIEW":
            text_hash = shorten_text_id(target_text)
            short_name = f"{product_name} Review {text_hash}"
            target_name = _esc(short_name)

        else:
            target_name = target_text

        # Reject garbage names
        if source_name.lower() in ("", "entity name"):
            continue
        if target_name.lower() in ("", "entity name"):
            continue

        # URL for this relationship
        rel_url = _esc(rel.get("source_url") or doc_url)

        # Evidence IDs
        evidence_ids = rel.get("evidence_ids") or []
        evidence_ids_str = "[" + ", ".join([f"'{_esc(eid)}'" for eid in evidence_ids]) + "]"

        # Skip incomplete relationships
        if not (source_name and target_name and relationship_type):
            continue

        # Normalize source label
        if relationship_type in ("HAS_DESCRIPTION", "HAS_SPEC_SHEET", "HAS_REVIEW"):
            source_label = "Product"
        elif relationship_type == "HAS_PRICE":
            source_label = "Product"
        elif relationship_type == "OFFERS_PRODUCT":
            source_label = "Company"
        elif relationship_type == "COMPETES_WITH":
            source_label = "Company"
        else:
            source_label = "Company"

        # Normalize target label
        if relationship_type == "HAS_DESCRIPTION":
            target_label = "Description"
        elif relationship_type == "HAS_SPEC_SHEET":
            target_label = "SpecSheet"
        elif relationship_type == "HAS_REVIEW":
            target_label = "Review"
        elif relationship_type == "HAS_PRICE":
            target_label = "Price"
        elif relationship_type == "HAS_SPECIFICATION":
            target_label = "Specification"
        elif relationship_type == "OFFERS_PRODUCT":
            target_label = "Product"
        elif relationship_type == "COMPETES_WITH":
            target_label = "Company"
        else:
            target_label = "Product"

        # Create source node
        stmts.append(
            f"MERGE (s:{source_label} {{name: '{source_name}'}}) "
            f"ON CREATE SET s.source_urls = ['{rel_url}'] "
            f"ON MATCH SET s.source_urls = CASE WHEN '{rel_url}' IN s.source_urls "
            f"THEN s.source_urls ELSE s.source_urls + '{rel_url}' END;"
        )

        # Create target node
        if relationship_type in ("HAS_SPEC_SHEET", "HAS_DESCRIPTION", "HAS_REVIEW"):
            stmts.append(
                f"MERGE (t:{target_label} {{name: '{target_name}'}}) "
                f"ON CREATE SET t.source_urls = ['{rel_url}'], t.text = '{target_text}' "
                f"ON MATCH SET "
                f"t.source_urls = CASE WHEN '{rel_url}' IN t.source_urls "
                f"THEN t.source_urls ELSE t.source_urls + '{rel_url}' END, "
                f"t.text = CASE WHEN t.text IS NULL THEN '{target_text}' ELSE t.text END;"
            )
        else:
            stmts.append(
                f"MERGE (t:{target_label} {{name: '{target_name}'}}) "
                f"ON CREATE SET t.source_urls = ['{rel_url}'] "
                f"ON MATCH SET t.source_urls = CASE WHEN '{rel_url}' IN t.source_urls "
                f"THEN t.source_urls ELSE t.source_urls + '{rel_url}' END;"
            )

        # Create relationship
        stmts.append(
            f"MATCH (s:{source_label} {{name: '{source_name}'}}), "
            f"(t:{target_label} {{name: '{target_name}'}}) "
            f"MERGE (s)-[rel:{relationship_type}]->(t) "
            f"ON CREATE SET rel.source_urls = ['{rel_url}'], rel.evidence_ids = {evidence_ids_str} "
            f"ON MATCH SET rel.source_urls = CASE WHEN '{rel_url}' IN rel.source_urls "
            f"THEN rel.source_urls ELSE rel.source_urls + '{rel_url}' END, "
            f"rel.evidence_ids = CASE WHEN SIZE(rel.evidence_ids) = 0 "
            f"THEN {evidence_ids_str} ELSE rel.evidence_ids END;"
        )

    # Combine statements
    cypher = "\n".join(stmts)

    print(f"[cypher] Generated {len(stmts)} statements")
    print(f"[cypher] Preview: {cypher[:200]}...")

    return cypher
