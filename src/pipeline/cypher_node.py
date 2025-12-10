"""
Cypher Node - Converts extracted JSON data into Neo4j Cypher statements.

This module transforms the structured data (companies, products, prices, SPECIFICATIONS)
into Cypher MERGE statements that create the knowledge graph in Neo4j.

ENHANCED FOR STRUCTURED SPECIFICATIONS:
- Each specification is a separate node with: name, value, unit, normalized_value
- Products connect to specs via HAS_SPEC relationship
- Enables head-to-head comparison queries in Neo4j

Each node and relationship stores:
- source_urls (array): Which web pages mentioned this entity
- evidence_ids (array): ChromaDB chunk IDs linking to raw source evidence
"""

from __future__ import annotations

from typing import Dict, List, Any
import re


def _esc(s: str) -> str:
    """Escape special characters for Cypher queries."""
    return (
        (s or "")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace('"', '\\"')
    )


def _normalize_spec_name(name: str) -> str:
    """Normalize specification names for consistent querying."""
    # Convert to title case and clean up
    name = name.replace("_", " ").title()
    return name


# Known company patterns for product name inference
# Order matters - more specific patterns first
COMPANY_PATTERNS = [
    ("Emerson Rosemount", ["rosemount"]),
    ("Emerson", ["emerson", "fisher"]),
    ("ABB", ["abb ", "abb-", "abb266", "266 pressure"]),
    ("Siemens", ["siemens", "sitrans"]),
    ("Endress+Hauser", ["endress", "e+h", "cerabar", "deltabar", "promass"]),
    ("Yokogawa", ["yokogawa", "ejx", "eja", "dpharp"]),
    ("WIKA", ["wika"]),
    ("Honeywell", ["honeywell", "smartline", "st700", "st800", "stg700", "std700"]),
    ("Schneider Electric", ["schneider", "foxboro"]),
    ("Danfoss", ["danfoss"]),
    ("Fuji Electric", ["fuji"]),
    ("Krohne", ["krohne"]),
    ("Vega", ["vega"]),
]


def _infer_company_from_product(product_name: str) -> str:
    """
    Infer company name from product name using known patterns.
    Returns empty string if no match found.
    """
    product_lower = product_name.lower()
    
    for company, patterns in COMPANY_PATTERNS:
        for pattern in patterns:
            if pattern.lower() in product_lower:
                return company
    
    return ""


def to_merge_cypher(data: Dict) -> str:
    """
    Convert JSON data to Cypher MERGE statements for Neo4j.
    
    ENHANCED: Now creates structured Specification nodes with:
    - name: The specification type (e.g., "Pressure Range")
    - value: The actual value (e.g., "0-6000 psi")
    - spec_type: The ontology key (e.g., "pressure_range")
    - unit: The unit if applicable (e.g., "psi")
    
    This enables queries like:
    - "Compare accuracy across all products"
    - "Find products with pressure range > 3000 psi"
    - "Which products support HART protocol?"
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
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Review) REQUIRE n.name IS UNIQUE;",
    ])
    
    # Step 2: ALWAYS create Honeywell as the baseline company FIRST
    stmts.append(
        "MERGE (h:Company {name: 'Honeywell'}) "
        "ON CREATE SET h.is_baseline = true, h.source_urls = [] "
        "ON MATCH SET h.is_baseline = true;"
    )
    
    # Step 3: Extract source URL from data
    raw_url = (data.get("Doc") or {}).get("source_url") or ""
    if isinstance(raw_url, list):
        doc_url = _esc(raw_url[-1] if raw_url else "")
    else:
        doc_url = _esc(raw_url)
    
    # Track created nodes to avoid duplicates
    created_companies = set(["Honeywell"])  # Honeywell already created above
    created_products = set()
    created_specs = set()
    
    # Step 4: Get all relationships
    relationships = data.get("relationships", []) or data.get("Relationships", []) or []
    
    # Step 5: FIRST PASS - Create all competitor companies and COMPETES_WITH relationships
    # This ensures all companies exist before we try to link products to them
    for rel in relationships:
        relationship_type = (rel.get("relationship") or "").upper().replace(" ", "_")
        if relationship_type == "COMPETES_WITH":
            competitor_name = _esc(rel.get("target") or "")
            rel_url = _esc(rel.get("source_url") or doc_url)
            
            if competitor_name and competitor_name.lower() != "honeywell":
                # Create the competitor company node
                stmts.append(
                    f"MERGE (c:Company {{name: '{competitor_name}'}}) "
                    f"ON CREATE SET c.source_urls = ['{rel_url}'] "
                    f"ON MATCH SET c.source_urls = CASE WHEN '{rel_url}' IN c.source_urls THEN c.source_urls ELSE c.source_urls + '{rel_url}' END;"
                )
                created_companies.add(competitor_name)
                
                # Create the COMPETES_WITH relationship from Honeywell
                stmts.append(
                    f"MATCH (h:Company {{name: 'Honeywell'}}), (c:Company {{name: '{competitor_name}'}}) "
                    f"MERGE (h)-[rel:COMPETES_WITH]->(c) "
                    f"ON CREATE SET rel.source_urls = ['{rel_url}'] "
                    f"ON MATCH SET rel.source_urls = CASE WHEN '{rel_url}' IN rel.source_urls THEN rel.source_urls ELSE rel.source_urls + '{rel_url}' END;"
                )
                print(f"[cypher] Created COMPETES_WITH: Honeywell → {competitor_name}")
    
    # Step 6: Process remaining relationships
    for rel in relationships:
        relationship_type = _esc((rel.get("relationship") or "").upper().replace(" ", "_"))
        source_label = rel.get("source_type") or "Company"
        target_label = rel.get("target_type") or "Product"
        source_name = _esc(rel.get("source") or "")
        target_name = _esc(rel.get("target") or "")
        
        # Skip COMPETES_WITH - already handled in first pass
        if relationship_type == "COMPETES_WITH":
            continue
        
        # Get source URL and evidence
        rel_url = _esc(rel.get("source_url") or doc_url)
        evidence_ids = rel.get("evidence_ids") or []
        evidence_ids_str = "[" + ", ".join([f"'{_esc(eid)}'" for eid in evidence_ids]) + "]"
        
        if not (source_name and target_name and relationship_type):
            continue
        
        # Normalize labels
        source_label = _normalize_label(source_label, relationship_type, is_source=True)
        target_label = _normalize_label(target_label, relationship_type, is_source=False)
        
        # === ENSURE COMPANY EXISTS FOR OFFERS_PRODUCT ===
        if relationship_type == "OFFERS_PRODUCT" and source_label == "Company":
            if source_name not in created_companies:
                stmts.append(
                    f"MERGE (c:Company {{name: '{source_name}'}}) "
                    f"ON CREATE SET c.source_urls = ['{rel_url}'] "
                    f"ON MATCH SET c.source_urls = CASE WHEN '{rel_url}' IN c.source_urls THEN c.source_urls ELSE c.source_urls + '{rel_url}' END;"
                )
                created_companies.add(source_name)
                
                # Also create COMPETES_WITH if not Honeywell
                if source_name.lower() != "honeywell":
                    stmts.append(
                        f"MATCH (h:Company {{name: 'Honeywell'}}), (c:Company {{name: '{source_name}'}}) "
                        f"MERGE (h)-[rel:COMPETES_WITH]->(c) "
                        f"ON CREATE SET rel.source_urls = ['{rel_url}'] "
                        f"ON MATCH SET rel.source_urls = CASE WHEN '{rel_url}' IN rel.source_urls THEN rel.source_urls ELSE rel.source_urls + '{rel_url}' END;"
                    )
                    print(f"[cypher] Auto-created COMPETES_WITH: Honeywell → {source_name}")
        
        # === SPECIAL HANDLING FOR HAS_SPEC RELATIONSHIPS ===
        if relationship_type == "HAS_SPEC":
            # Support both "spec_type" (from agent) and "spec_name" (legacy)
            spec_name = rel.get("spec_type") or rel.get("spec_name", "")
            spec_value = rel.get("spec_value", target_name)
            normalized_value = rel.get("normalized_value", spec_value)  # Converted value
            spec_unit = rel.get("unit", "")  # Original unit
            rel_snippet = _esc(rel.get("snippet", ""))
            
            # Create specification node with rich metadata
            spec_display_name = _normalize_spec_name(spec_name)
            # Use a unique ID internally but display the value
            spec_node_id = f"{source_name}_{spec_name}"
            # The displayed name shows the spec type and value
            spec_display_value = f"{spec_display_name}: {spec_value}"
            # Make the specification node name unique per product to avoid cross-links
            spec_node_name = f"{source_name} | {spec_display_value}"
            
            if spec_node_id not in created_specs:
                created_specs.add(spec_node_id)
                
                # Specification node: name shows the value for display in graph
                # Store both original value and normalized value for comparison
                stmts.append(
                    f"MERGE (spec:Specification {{name: '{_esc(spec_node_name)}'}}) "
                    f"ON CREATE SET spec.spec_type = '{_esc(spec_name)}', "
                    f"spec.display_name = '{_esc(spec_display_name)}', "
                    f"spec.value = '{_esc(spec_value)}', "
                    f"spec.normalized_value = '{_esc(str(normalized_value))}', "
                    f"spec.unit = '{_esc(spec_unit)}', "
                    f"spec.product = '{source_name}', "
                    f"spec.source_urls = ['{rel_url}'] "
                    f"ON MATCH SET spec.value = '{_esc(spec_value)}', "
                    f"spec.normalized_value = '{_esc(str(normalized_value))}', "
                    f"spec.unit = '{_esc(spec_unit)}', "
                    f"spec.source_urls = CASE WHEN '{rel_url}' IN spec.source_urls THEN spec.source_urls ELSE spec.source_urls + '{rel_url}' END;"
                )
            
            # Create product node if not exists
            if source_name not in created_products:
                created_products.add(source_name)
                stmts.append(
                    f"MERGE (p:Product {{name: '{source_name}'}}) "
                    f"ON CREATE SET p.source_urls = ['{rel_url}'] "
                    f"ON MATCH SET p.source_urls = CASE WHEN '{rel_url}' IN p.source_urls THEN p.source_urls ELSE p.source_urls + '{rel_url}' END;"
                )
                
                # AUTO-INFER company from product name and create OFFERS_PRODUCT
                inferred_company = _infer_company_from_product(source_name)
                if inferred_company:
                    # Create the company if it doesn't exist
                    if inferred_company not in created_companies:
                        stmts.append(
                            f"MERGE (c:Company {{name: '{inferred_company}'}}) "
                            f"ON CREATE SET c.source_urls = ['{rel_url}'] "
                            f"ON MATCH SET c.source_urls = CASE WHEN '{rel_url}' IN c.source_urls THEN c.source_urls ELSE c.source_urls + '{rel_url}' END;"
                        )
                        created_companies.add(inferred_company)
                        
                        # Create COMPETES_WITH if not Honeywell
                        if inferred_company.lower() != "honeywell":
                            stmts.append(
                                f"MATCH (h:Company {{name: 'Honeywell'}}), (c:Company {{name: '{inferred_company}'}}) "
                                f"MERGE (h)-[rel:COMPETES_WITH]->(c) "
                                f"ON CREATE SET rel.source_urls = ['{rel_url}'] "
                                f"ON MATCH SET rel.source_urls = CASE WHEN '{rel_url}' IN rel.source_urls THEN rel.source_urls ELSE rel.source_urls + '{rel_url}' END;"
                            )
                            print(f"[cypher] Auto-inferred COMPETES_WITH: Honeywell → {inferred_company}")
                    
                    # Create OFFERS_PRODUCT relationship
                    stmts.append(
                        f"MATCH (c:Company {{name: '{inferred_company}'}}), (p:Product {{name: '{source_name}'}}) "
                        f"MERGE (c)-[rel:OFFERS_PRODUCT]->(p) "
                        f"ON CREATE SET rel.source_urls = ['{rel_url}'] "
                        f"ON MATCH SET rel.source_urls = CASE WHEN '{rel_url}' IN rel.source_urls THEN rel.source_urls ELSE rel.source_urls + '{rel_url}' END;"
                    )
                    print(f"[cypher] Auto-inferred OFFERS_PRODUCT: {inferred_company} → {source_name}")
            
            # Create HAS_SPEC relationship
            stmts.append(
                f"MATCH (p:Product {{name: '{source_name}'}}), "
                f"(spec:Specification {{name: '{_esc(spec_node_name)}'}}) "
                f"MERGE (p)-[rel:HAS_SPEC]->(spec) "
                f"ON CREATE SET rel.source_urls = ['{rel_url}'], rel.evidence_ids = {evidence_ids_str}, rel.snippet = '{rel_snippet}' "
                f"ON MATCH SET rel.source_urls = CASE WHEN '{rel_url}' IN rel.source_urls THEN rel.source_urls ELSE rel.source_urls + '{rel_url}' END, "
                f"rel.snippet = CASE WHEN rel.snippet IS NULL OR rel.snippet = '' THEN '{rel_snippet}' ELSE rel.snippet END;"
            )
            continue
        
        # === SPECIAL HANDLING FOR HAS_REVIEW RELATIONSHIPS ===
        if relationship_type == "HAS_REVIEW":
            review_text = rel.get("review_text", target_name)[:200]  # Limit text length
            rating = rel.get("rating", "")
            review_source = rel.get("review_source", "")
            rel_url = _esc(rel.get("source_url") or doc_url)
            rel_snippet = _esc(rel.get("snippet", ""))
            
            # Create a unique review node name
            review_display = f"{rating}: {review_text[:50]}..." if rating else f"{review_text[:50]}..."
            review_node_id = f"{source_name}_{hash(review_text) % 10000}"
            
            # Create review node
            stmts.append(
                f"MERGE (review:Review {{name: '{_esc(review_display)}'}}) "
                f"ON CREATE SET review.text = '{_esc(review_text)}', "
                f"review.rating = '{_esc(rating)}', "
                f"review.source = '{_esc(review_source)}', "
                f"review.product = '{source_name}', "
                f"review.source_urls = ['{rel_url}'] "
                f"ON MATCH SET review.text = '{_esc(review_text)}', "
                f"review.source_urls = CASE WHEN '{rel_url}' IN review.source_urls THEN review.source_urls ELSE review.source_urls + '{rel_url}' END;"
            )
            
            # Create product node if not exists
            if source_name not in created_products:
                created_products.add(source_name)
                stmts.append(
                    f"MERGE (p:Product {{name: '{source_name}'}}) "
                    f"ON CREATE SET p.source_urls = ['{rel_url}'] "
                    f"ON MATCH SET p.source_urls = CASE WHEN '{rel_url}' IN p.source_urls THEN p.source_urls ELSE p.source_urls + '{rel_url}' END;"
                )
            
            # Create HAS_REVIEW relationship
            stmts.append(
                f"MATCH (p:Product {{name: '{source_name}'}}), "
                f"(review:Review {{name: '{_esc(review_display)}'}}) "
                f"MERGE (p)-[rel:HAS_REVIEW]->(review) "
                f"ON CREATE SET rel.source_urls = ['{rel_url}'], rel.snippet = '{rel_snippet}', rel.evidence_ids = {evidence_ids_str} "
                f"ON MATCH SET rel.source_urls = CASE WHEN '{rel_url}' IN rel.source_urls THEN rel.source_urls ELSE rel.source_urls + '{rel_url}' END, "
                f"rel.snippet = CASE WHEN rel.snippet IS NULL OR rel.snippet = '' THEN '{rel_snippet}' ELSE rel.snippet END, "
                f"rel.evidence_ids = CASE WHEN SIZE(rel.evidence_ids) = 0 THEN {evidence_ids_str} ELSE rel.evidence_ids END;"
            )
            continue
        
        # === SPECIAL HANDLING FOR HAS_PRICE RELATIONSHIPS ===
        if relationship_type == "HAS_PRICE":
            price_value = rel.get("price_value", target_name)
            rel_snippet = _esc(rel.get("snippet", ""))
            price_node_name = f"{source_name} | {price_value}"
            
            # Price node unique per product
            stmts.append(
                f"MERGE (price:Price {{name: '{_esc(price_node_name)}'}}) "
                f"ON CREATE SET price.value = '{_esc(price_value)}', price.source_urls = ['{rel_url}'] "
                f"ON MATCH SET price.value = '{_esc(price_value)}', "
                f"price.source_urls = CASE WHEN '{rel_url}' IN price.source_urls THEN price.source_urls ELSE price.source_urls + '{rel_url}' END;"
            )
            
            # Create product node if not exists
            if source_name not in created_products:
                created_products.add(source_name)
                stmts.append(
                    f"MERGE (p:Product {{name: '{source_name}'}}) "
                    f"ON CREATE SET p.source_urls = ['{rel_url}'] "
                    f"ON MATCH SET p.source_urls = CASE WHEN '{rel_url}' IN p.source_urls THEN p.source_urls ELSE p.source_urls + '{rel_url}' END;"
                )
            
            # Create HAS_PRICE relationship
            stmts.append(
                f"MATCH (p:Product {{name: '{source_name}'}}), "
                f"(price:Price {{name: '{_esc(price_node_name)}'}}) "
                f"MERGE (p)-[rel:HAS_PRICE]->(price) "
                f"ON CREATE SET rel.source_urls = ['{rel_url}'], rel.evidence_ids = {evidence_ids_str}, rel.snippet = '{rel_snippet}' "
                f"ON MATCH SET rel.source_urls = CASE WHEN '{rel_url}' IN rel.source_urls THEN rel.source_urls ELSE rel.source_urls + '{rel_url}' END, "
                f"rel.evidence_ids = CASE WHEN SIZE(rel.evidence_ids) = 0 THEN {evidence_ids_str} ELSE rel.evidence_ids END, "
                f"rel.snippet = CASE WHEN rel.snippet IS NULL OR rel.snippet = '' THEN '{rel_snippet}' ELSE rel.snippet END;"
            )
            continue
        
        # === STANDARD RELATIONSHIP HANDLING ===
        
        # Create source node
        stmts.append(
            f"MERGE (s:{source_label} {{name: '{source_name}'}}) "
            f"ON CREATE SET s.source_urls = ['{rel_url}'] "
            f"ON MATCH SET s.source_urls = CASE WHEN '{rel_url}' IN s.source_urls THEN s.source_urls ELSE s.source_urls + '{rel_url}' END;"
        )
        
        # Create target node
        stmts.append(
            f"MERGE (t:{target_label} {{name: '{target_name}'}}) "
            f"ON CREATE SET t.source_urls = ['{rel_url}'] "
            f"ON MATCH SET t.source_urls = CASE WHEN '{rel_url}' IN t.source_urls THEN t.source_urls ELSE t.source_urls + '{rel_url}' END;"
        )
        
        # Create relationship
        rel_snippet = _esc(rel.get("snippet", ""))
        stmts.append(
            f"MATCH (s:{source_label} {{name: '{source_name}'}}), (t:{target_label} {{name: '{target_name}'}}) "
            f"MERGE (s)-[rel:{relationship_type}]->(t) "
            f"ON CREATE SET rel.source_urls = ['{rel_url}'], rel.evidence_ids = {evidence_ids_str}, rel.snippet = '{rel_snippet}' "
            f"ON MATCH SET rel.source_urls = CASE WHEN '{rel_url}' IN rel.source_urls THEN rel.source_urls ELSE rel.source_urls + '{rel_url}' END, "
            f"rel.evidence_ids = CASE WHEN SIZE(rel.evidence_ids) = 0 THEN {evidence_ids_str} ELSE rel.evidence_ids END, "
            f"rel.snippet = CASE WHEN rel.snippet IS NULL OR rel.snippet = '' THEN '{rel_snippet}' ELSE rel.snippet END;"
        )
    
    # Combine all statements
    cypher = "\n".join(stmts)
    
    print(f"[cypher] Generated {len(stmts)} statements")
    print(f"[cypher] Preview: {cypher[:300]}...")
    
    return cypher


def _normalize_label(label: str, relationship_type: str, is_source: bool) -> str:
    """Normalize node labels based on context."""
    if label in ("Company", "Brand"):
        return "Company"
    elif label == "Product":
        return "Product"
    elif label == "Price":
        return "Price"
    elif label == "Specification":
        return "Specification"
    else:
        # Infer from relationship type
        if relationship_type == "COMPETES_WITH":
            return "Company"
        elif relationship_type == "OFFERS_PRODUCT":
            return "Company" if is_source else "Product"
        elif relationship_type == "HAS_PRICE":
            return "Product" if is_source else "Price"
        elif relationship_type in ("HAS_SPECIFICATION", "HAS_SPEC"):
            return "Product" if is_source else "Specification"
        elif relationship_type == "HAS_REVIEW":
            return "Product" if is_source else "Review"
        else:
            return "Company" if is_source else "Product"


# =============================================================================
# ENHANCED QUERIES FOR SPECIFICATIONS
# =============================================================================

def get_product_comparison_query(product_names: List[str]) -> str:
    """
    Generate a Cypher query to compare specifications across products.
    
    Example usage:
    MATCH (p:Product)-[:HAS_SPEC]->(s:Specification)
    WHERE p.name IN ['ST700', 'Rosemount 3051', 'A-10']
    RETURN p.name as product, s.spec_type, s.value
    ORDER BY s.spec_type, p.name
    """
    products_str = ", ".join([f"'{_esc(p)}'" for p in product_names])
    return f"""
MATCH (p:Product)-[:HAS_SPEC]->(s:Specification)
WHERE p.name IN [{products_str}]
RETURN p.name as product, s.spec_type as spec, s.value as value
ORDER BY s.spec_type, p.name
"""


def get_spec_pivot_query() -> str:
    """
    Generate a query to get all products with their specs in a pivotable format.
    """
    return """
MATCH (c:Company)-[:OFFERS_PRODUCT]->(p:Product)
OPTIONAL MATCH (p)-[:HAS_SPEC]->(s:Specification)
OPTIONAL MATCH (p)-[:HAS_PRICE]->(price:Price)
RETURN 
    c.name as company,
    p.name as product,
    collect(DISTINCT {spec: s.spec_type, value: s.value}) as specifications,
    price.name as price
ORDER BY c.name, p.name
"""


def get_spec_search_query(spec_type: str, condition: str) -> str:
    """
    Generate a query to find products matching a specification condition.
    
    Example: Find products with accuracy better than 0.1%
    """
    return f"""
MATCH (p:Product)-[:HAS_SPEC]->(s:Specification)
WHERE s.spec_type = '{_esc(spec_type)}' AND {condition}
RETURN p.name as product, s.value as {spec_type}
ORDER BY s.value
"""
