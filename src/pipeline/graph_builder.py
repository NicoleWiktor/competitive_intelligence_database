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
            
            # 5. Create CustomerNeed nodes PER PRODUCT (each product gets its own need node)
            # This avoids multiple products pointing to the same need node
            customer_needs = data.get("customer_needs", {})
            need_mappings = data.get("need_mappings", [])
            
            # Create CustomerNeed nodes only for products that have mappings
            # Each product gets its own copy of the need
            mapping_count = 0
            created_needs = set()
            
            for mapping in need_mappings:
                need_name = mapping.get("need", "").replace("'", "").replace('"', '')
                product = mapping.get("product", "").replace("'", "").replace('"', '')
                spec = mapping.get("spec", "").replace("'", "")
                spec_value = mapping.get("spec_value", "").replace("'", "").replace('"', '')[:100]
                need_threshold = mapping.get("need_threshold", "").replace("'", "")[:50]
                explanation = mapping.get("explanation", "").replace("'", "").replace('"', '')[:200]
                
                if not need_name or not product:
                    continue
                
                # Get need data
                need_data = customer_needs.get(need_name, {})
                spec_type = (need_data.get("spec_type", "") or "").replace("'", "")[:50]
                threshold = (need_data.get("threshold", "") or "").replace("'", "")[:50]
                
                # Handle both source_urls (list) and source_url (string) for backwards compat
                source_urls = need_data.get("source_urls", [])
                if not source_urls:
                    single_url = need_data.get("source_url", "")
                    source_urls = [single_url] if single_url else []
                # Clean and format URLs for Neo4j
                clean_urls = [u.replace("'", "")[:200] for u in source_urls[:5]]  # Max 5 URLs
                source_urls_str = json.dumps(clean_urls)
                
                evidence_ids = need_data.get("evidence_ids", [])
                evidence_str = json.dumps(evidence_ids[:20]) if evidence_ids else "[]"  # More evidence
                
                # Create a UNIQUE CustomerNeed node for this product
                # Key includes product name to make it unique per product
                need_key = f"{product}|{need_name}"
                safe_need_display = need_name[:100]
                
                session.run(f"""
                    MERGE (n:CustomerNeed {{key: '{need_key}'}})
                    SET n.name = '{safe_need_display}',
                        n.spec_type = '{spec_type}',
                        n.threshold = '{threshold}',
                        n.source_urls = {source_urls_str},
                        n.evidence_ids = {evidence_str}
                """)
                created_needs.add(need_key)
                
                # Create ADDRESSES_NEED relationship (with evidence for verification!)
                session.run(f"""
                    MATCH (p:Product {{name: '{product}'}})
                    MATCH (n:CustomerNeed {{key: '{need_key}'}})
                    MERGE (p)-[r:ADDRESSES_NEED]->(n)
                    SET r.via_spec = '{spec}',
                        r.spec_value = '{spec_value}',
                        r.need_threshold = '{need_threshold}',
                        r.explanation = '{explanation}',
                        r.source_urls = {source_urls_str},
                        r.evidence_ids = {evidence_str}
                """)
                mapping_count += 1
            
            print(f"[neo4j] Created {len(created_needs)} customer needs (one per product)")
            print(f"[neo4j] Created {mapping_count} ADDRESSES_NEED relationships")
            
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
    industry: str = "process industries",
    max_iterations: int = 25,
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
    print(f"   Industry: {industry}")
    print(f"   Max competitors: {max_competitors}")
    print("="*60)
    
    # Always reset unless incremental
    if not incremental:
        print("\n🗑️  Resetting Neo4j...")
        reset_neo4j()
    
    # Run research
    print("\n📊 Researching competitors...")
    data = run_agent(max_competitors=max_competitors, industry=industry, max_iterations=max_iterations)
    
    # Write directly to Neo4j
    print("\n📝 Writing to Neo4j...")
    write_to_neo4j(data)
    
    # Save report to file for Streamlit
    if data.get("industry_needs_report"):
        report_data = {
            "report": data.get("industry_needs_report", ""),
            "sources": data.get("report_sources", []),
            "industry": industry,
            "needs_count": len(data.get("customer_needs", {})),
            "mappings_count": len(data.get("need_mappings", []))
        }
        import os
        report_path = os.path.join(os.path.dirname(__file__), "..", "..", "industry_report.json")
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"📄 Report saved to industry_report.json")
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    
    return data


if __name__ == "__main__":
    run_pipeline(max_competitors=5)
