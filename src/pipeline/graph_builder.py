"""
Graph Builder - Simple, direct Neo4j pipeline.

STRICT LIMITS:
- 10 competitors max
- 10 products per competitor  
- 10 specs per product

Graph Structure:
    Honeywell (center)
        ├── OFFERS_PRODUCT → SmartLine ST700 → HAS_SPEC → specs
        ├── COMPETES_WITH → Competitor → OFFERS_PRODUCT → Product → HAS_SPEC → specs
"""

from __future__ import annotations

import json
from typing import Any, Dict
from neo4j import GraphDatabase

from src.config.settings import get_neo4j_config


def get_driver():
    """Get Neo4j driver."""
    cfg = get_neo4j_config()
    return GraphDatabase.driver(
        cfg.get("uri"),
        auth=(cfg.get("user"), cfg.get("password"))
    )


def reset_neo4j():
    """Clear ALL nodes, relationships, and constraints."""
    driver = get_driver()
    try:
        with driver.session() as session:
            # Drop all constraints
            try:
                constraints = session.run("SHOW CONSTRAINTS").data()
                for c in constraints:
                    name = c.get('name', '')
                    if name:
                        try:
                            session.run(f"DROP CONSTRAINT {name}")
                            print(f"[neo4j] Dropped constraint: {name}")
                        except:
                            pass
            except:
                pass
            
            # Delete everything
            session.run("MATCH (n) DETACH DELETE n")
            
            # Verify deletion
            count = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
            print(f"[neo4j] Database cleared. Node count: {count}")
    finally:
        driver.close()


def count_nodes():
    """Count nodes by type."""
    driver = get_driver()
    try:
        with driver.session() as session:
            query = """
            MATCH (n) 
            RETURN labels(n)[0] as label, count(n) as count
            ORDER BY label
            """
            results = session.run(query)
            counts = {r["label"]: r["count"] for r in results}
            return counts
    finally:
        driver.close()


def write_to_neo4j(data: Dict[str, Any]):
    """
    Write data directly to Neo4j with simple Cypher.
    
    Includes evidence_ids and source_urls on relationships for human verification.
    The Streamlit "Verify Data" tab uses these to show source evidence from ChromaDB.
    """
    driver = get_driver()
    
    competitors = data.get("competitors", {})
    products = data.get("products", {})
    specifications = data.get("specifications", {})
    
    try:
        with driver.session() as session:
            # 1. Create Honeywell (center node)
            session.run("""
                MERGE (h:Company {name: 'Honeywell'})
                SET h.is_baseline = true
            """)
            print("[neo4j] Created Honeywell")
            
            # 2. Create competitors and COMPETES_WITH relationships (with evidence)
            for comp_name, comp_data in competitors.items():
                safe_name = comp_name.replace("'", "").replace('"', '')
                source_url = (comp_data.get("source_url", "") or "").replace("'", "")[:200]
                evidence_ids = comp_data.get("evidence_ids", [])
                evidence_str = json.dumps(evidence_ids[:10]) if evidence_ids else "[]"
                
                session.run(f"""
                    MERGE (c:Company {{name: '{safe_name}'}})
                    WITH c
                    MATCH (h:Company {{name: 'Honeywell'}})
                    MERGE (h)-[r:COMPETES_WITH]->(c)
                    SET r.source_urls = ['{source_url}'],
                        r.evidence_ids = {evidence_str}
                """)
            print(f"[neo4j] Created {len(competitors)} competitors with evidence links")
            
            # 3. Create products and OFFERS_PRODUCT relationships (with evidence)
            product_count = 0
            for prod_name, prod_data in products.items():
                safe_prod = prod_name.replace("'", "").replace('"', '')[:100]
                company = prod_data.get("company", "").replace("'", "").replace('"', '')
                source_url = (prod_data.get("source_url", "") or "").replace("'", "")[:200]
                evidence_ids = prod_data.get("evidence_ids", [])
                evidence_str = json.dumps(evidence_ids[:10]) if evidence_ids else "[]"
                
                if company:
                    session.run(f"""
                        MERGE (p:Product {{name: '{safe_prod}'}})
                        SET p.source_urls = ['{source_url}']
                        WITH p
                        MATCH (c:Company {{name: '{company}'}})
                        MERGE (c)-[r:OFFERS_PRODUCT]->(p)
                        SET r.source_urls = ['{source_url}'],
                            r.evidence_ids = {evidence_str}
                    """)
                    product_count += 1
            print(f"[neo4j] Created {product_count} products with evidence links")
            
            # 4. Create specs and HAS_SPEC relationships (with evidence)
            spec_count = 0
            for prod_name, specs in specifications.items():
                safe_prod = prod_name.replace("'", "").replace('"', '')[:100]
                
                # Get evidence from the product
                prod_data = products.get(prod_name, {})
                source_url = (prod_data.get("source_url", "") or "").replace("'", "")[:200]
                evidence_ids = prod_data.get("evidence_ids", [])
                evidence_str = json.dumps(evidence_ids[:10]) if evidence_ids else "[]"
                
                for spec_type, spec_value in specs.items():
                    safe_type = spec_type.replace("'", "").replace('"', '').replace('_', ' ').title()[:50]
                    safe_value = str(spec_value).replace("'", "").replace('"', '').replace('\n', ' ')[:100]
                    
                    # SHOW THE VALUE in the name! e.g., "Accuracy: ±0.065%"
                    spec_display = f"{safe_type}: {safe_value}"
                    # Unique key per product to avoid cross-links
                    spec_key = f"{safe_prod}|{spec_type}"
                    
                    session.run(f"""
                        MERGE (s:Specification {{key: '{spec_key}'}})
                        SET s.name = '{spec_display}',
                            s.spec_type = '{safe_type}',
                            s.value = '{safe_value}',
                            s.product = '{safe_prod}'
                        WITH s
                        MATCH (p:Product {{name: '{safe_prod}'}})
                        MERGE (p)-[r:HAS_SPEC]->(s)
                        SET r.source_urls = ['{source_url}'],
                            r.evidence_ids = {evidence_str}
                    """)
                    spec_count += 1
            print(f"[neo4j] Created {spec_count} specifications with evidence links")
            
            # Verify final counts
            counts = count_nodes()
            print(f"\n[neo4j] FINAL COUNTS:")
            for label, count in counts.items():
                print(f"   {label}: {count}")
                
    finally:
        driver.close()


def run_pipeline(
    target_product: str = "SmartLine ST700",
    target_company: str = "Honeywell",
    max_competitors: int = 10,
    incremental: bool = False,
) -> Dict[str, Any]:
    """
    Run the AGENTIC pipeline.
    
    The agent DECIDES what to do:
    - Which searches to run
    - Which pages to extract
    - What data to save
    - When to stop
    """
    from src.agents.agentic_agent import run_agent
    
    print("="*60)
    print("🚀 COMPETITIVE INTELLIGENCE PIPELINE")
    print(f"   Target: {target_company} {target_product}")
    print(f"   Max competitors: {max_competitors}")
    print("="*60)
    
    # Always reset unless incremental
    if not incremental:
        print("\n🗑️  Resetting Neo4j...")
        reset_neo4j()
    
    # Run research
    print("\n📊 Researching competitors...")
    data = run_agent(max_competitors=max_competitors)
    
    # Write directly to Neo4j
    print("\n📝 Writing to Neo4j...")
    write_to_neo4j(data)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    
    return data


if __name__ == "__main__":
    run_pipeline(max_competitors=5)
