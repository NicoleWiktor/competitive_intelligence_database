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
from typing import Any, Dict, List
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


def write_to_neo4j(data: Dict[str, Any], company: str = "Honeywell"):
    driver = get_driver()
    
    competitors = data.get("competitors", {})
    products = data.get("products", {})
    specifications = data.get("specifications", {})
    
    safe_company = company.replace("'", "").replace('"', '')
    
    try:
        with driver.session() as session:
            # 1. Create target company (center node)
            session.run(f"""
                MERGE (h:Company {{name: '{safe_company}'}})
                SET h.is_baseline = true
            """)
            print(f"[neo4j] Created {safe_company}")

            # 2. Create competitors and COMPETES_WITH relationships
            for comp_name, comp_data in competitors.items():
                safe_name = comp_name.replace("'", "").replace('"', '')
                source_url = (comp_data.get("source_url", "") or "").replace("'", "")[:200]
                evidence_ids = comp_data.get("evidence_ids", [])
                evidence_str = json.dumps(evidence_ids[:10]) if evidence_ids else "[]"
                
                session.run(f"""
                    MERGE (c:Company {{name: '{safe_name}'}})
                    WITH c
                    MATCH (h:Company {{name: '{safe_company}'}})
                    MERGE (h)-[r:COMPETES_WITH]->(c)
                    SET r.source_urls = ['{source_url}'],
                        r.evidence_ids = {evidence_str}
                """)
            print(f"[neo4j] Created {len(competitors)} competitors with evidence links")

            # 3. Create products and OFFERS_PRODUCT relationships
            product_count = 0
            for prod_name, prod_data in products.items():
                safe_prod = prod_name.replace("'", "").replace('"', '')[:100]
                company_name = prod_data.get("company", "").replace("'", "").replace('"', '')
                source_url = (prod_data.get("source_url", "") or "").replace("'", "")[:200]
                evidence_ids = prod_data.get("evidence_ids", [])
                evidence_str = json.dumps(evidence_ids[:10]) if evidence_ids else "[]"
                
                if company_name:
                    session.run(f"""
                        MERGE (p:Product {{name: '{safe_prod}'}})
                        SET p.source_urls = ['{source_url}']
                        WITH p
                        MATCH (c:Company {{name: '{company_name}'}})
                        MERGE (c)-[r:OFFERS_PRODUCT]->(p)
                        SET r.source_urls = ['{source_url}'],
                            r.evidence_ids = {evidence_str}
                    """)
                    product_count += 1
            print(f"[neo4j] Created {product_count} products with evidence links")

            # 4. Create specs and HAS_SPEC relationships
            spec_count = 0
            for prod_name, specs in specifications.items():
                safe_prod = prod_name.replace("'", "").replace('"', '')[:100]
                
                prod_data = products.get(prod_name, {})
                source_url = (prod_data.get("source_url", "") or "").replace("'", "")[:200]
                evidence_ids = prod_data.get("evidence_ids", [])
                evidence_str = json.dumps(evidence_ids[:10]) if evidence_ids else "[]"
                
                for spec_type, spec_value in specs.items():
                    safe_type = spec_type.replace("'", "").replace('"', '').replace('_', ' ').title()[:50]
                    safe_value = str(spec_value).replace("'", "").replace('"', '').replace('\n', ' ')[:100]
                    
                    spec_display = f"{safe_type}: {safe_value}"
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

            # 5. Create CustomerNeed nodes and ADDRESSES_NEED relationships
            customer_needs = data.get("customer_needs", {})
            need_mappings = data.get("need_mappings", [])
            
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
                
                need_data = customer_needs.get(need_name, {})
                spec_type = (need_data.get("spec_type", "") or "").replace("'", "")[:50]
                threshold = (need_data.get("threshold", "") or "").replace("'", "")[:50]
                
                source_urls = need_data.get("source_urls", [])
                if not source_urls:
                    single_url = need_data.get("source_url", "")
                    source_urls = [single_url] if single_url else []
                clean_urls = [u.replace("'", "")[:200] for u in source_urls[:5]]
                source_urls_str = json.dumps(clean_urls)
                
                evidence_ids = need_data.get("evidence_ids", [])
                evidence_str = json.dumps(evidence_ids[:20]) if evidence_ids else "[]"
                
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
            
            print(f"[neo4j] Created {len(created_needs)} customer needs")
            print(f"[neo4j] Created {mapping_count} ADDRESSES_NEED relationships")

            # 6. Create CustomerSegment nodes and ADDRESSES_CUSTOMER_SEGMENT relationships
            segments = data.get("customer_segments", [])
            segment_mappings = data.get("segment_mappings", [])
            segment_details = {seg.get("name", ""): seg for seg in segments}
            
            if segment_mappings:
                created_segments = set()
                seg_mapping_count = 0
                
                for m in segment_mappings:
                    segment_name = m.get("segment", "")[:100].replace("'", "").replace('"', '')
                    product = m.get("product", "")[:100].replace("'", "").replace('"', '')
                    reason = m.get("reason", "")[:300].replace("'", "").replace('"', '')
                    mapping_source_url = m.get("source_url", "")[:200].replace("'", "")
                    mapping_evidence_ids = json.dumps(m.get("evidence_ids", [])[:10])
                    
                    seg_data = segment_details.get(segment_name, {})
                    seg_desc = seg_data.get("description", "")[:200].replace("'", "").replace('"', '')
                    seg_industry = seg_data.get("industry", "")[:50].replace("'", "")
                    seg_source_url = seg_data.get("source_url", "")[:200].replace("'", "")
                    seg_evidence_text = seg_data.get("evidence_text", "")[:500].replace("'", "").replace('"', '')
                    seg_evidence_ids = json.dumps(seg_data.get("evidence_ids", [])[:10])
                    
                    segment_key = f"{product}|{segment_name}"
                    
                    session.run(f"""
                        MERGE (s:CustomerSegment {{key: '{segment_key}'}})
                        SET s.name = '{segment_name}',
                            s.description = '{seg_desc}',
                            s.industry = '{seg_industry}',
                            s.source_url = '{seg_source_url}',
                            s.evidence_text = '{seg_evidence_text}',
                            s.evidence_ids = {seg_evidence_ids}
                    """)
                    created_segments.add(segment_key)
                    
                    session.run(f"""
                        MATCH (p:Product {{name: '{product}'}})
                        MATCH (s:CustomerSegment {{key: '{segment_key}'}})
                        MERGE (p)-[r:ADDRESSES_CUSTOMER_SEGMENT]->(s)
                        SET r.reason = '{reason}',
                            r.source_url = '{mapping_source_url}',
                            r.evidence_ids = {mapping_evidence_ids}
                    """)
                    seg_mapping_count += 1
                
                print(f"[neo4j] Created {len(created_segments)} CustomerSegment nodes")
                print(f"[neo4j] Created {seg_mapping_count} ADDRESSES_CUSTOMER_SEGMENT relationships")

            # Verify final counts
            counts = count_nodes()
            print(f"\n[neo4j] FINAL COUNTS:")
            for label, count in counts.items():
                print(f"   {label}: {count}")
                
    finally:
        driver.close()

def run_pipeline(
    company: str = "Honeywell",
    target_product: str = "SmartLine ST700",
    product_category: str = "pressure transmitters",
    max_competitors: int = 10,
    industry: str = "process industries",
    max_iterations: int = 25,
    allowed_domains: List[str] | None = None,
    allowed_source_types: List[str] | None = None,
    incremental: bool = False,
    multi_agent: bool = False,
) -> Dict[str, Any]:
    from src.agents.agentic_agent import run_agent

    print("="*60)
    print("🚀 COMPETITIVE INTELLIGENCE PIPELINE")
    print(f"   Target: {company} {target_product}")
    print(f"   Category: {product_category}")
    print(f"   Industry: {industry}")
    print(f"   Max competitors: {max_competitors}")
    print(f"   Allowed domains: {allowed_domains or 'Any'}")
    print(f"   Allowed source types: {allowed_source_types or 'Any'}")
    print("="*60)

    if not incremental:
        print("\n🗑️  Resetting Neo4j...")
        reset_neo4j()

    print("\n📊 Researching competitors...")
    data = run_agent(
        company=company,
        product=target_product,
        product_category=product_category,
        max_competitors=max_competitors,
        industry=industry,
        max_iterations=max_iterations,
        allowed_domains=allowed_domains,
        allowed_source_types=allowed_source_types,
        multi_agent=multi_agent,
    )

    print("\n📝 Writing to Neo4j...")
    write_to_neo4j(data, company=company)

    # Save report
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

    if data.get("customer_segments"):
        import os
        segments_data = {
            "segments": data.get("customer_segments", []),
            "sources": data.get("segments_sources", []),
            "segment_mappings": data.get("segment_mappings", []),
            "industry": industry
        }
        segments_path = os.path.join(os.path.dirname(__file__), "..", "..", "customer_segments.json")
        with open(segments_path, "w") as f:
            json.dump(segments_data, f, indent=2)
        print(f"👥 Customer segments saved")

    if data.get("house_of_quality"):
        import os
        hoq_path = os.path.join(os.path.dirname(__file__), "..", "..", "house_of_quality.json")
        with open(hoq_path, "w") as f:
            json.dump(data.get("house_of_quality"), f, indent=2)
        hoq = data.get("house_of_quality", {})
        print(f"🏠 House of Quality saved ({len(hoq.get('whats', []))} needs × {len(hoq.get('hows', []))} specs)")

    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)

    return data


if __name__ == "__main__":
    run_pipeline(max_competitors=5)
