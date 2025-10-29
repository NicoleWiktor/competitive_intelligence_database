from __future__ import annotations

from typing import Dict, List


def _esc(s: str) -> str:
    return (s or "").replace("\\", "\\\\").replace("'", "\\'")


def to_merge_cypher(data: Dict) -> str:
    """Convert one schema-shaped JSON object into Cypher MERGE statements.

    Expected keys: Industry, CustomerSegment, CustomerNeed[], HoneywellProduct,
    Competitor, CompetitiveProduct, Relationships[] with
    {source_type, source, relationship, target_type, target}.
    """
    if not data:
        return ""

    stmts: List[str] = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Industry)        REQUIRE n.name IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:CustomerSegment) REQUIRE n.name IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:CustomerNeed)    REQUIRE n.name IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Company)         REQUIRE n.name IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Product)         REQUIRE n.name IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Brand)           REQUIRE n.name IS UNIQUE;",
    ]

    # Nodes
    # Handle source_url as either string or list of strings
    raw_url = (data.get("Doc") or {}).get("source_url") or ""
    if isinstance(raw_url, list):
        doc_url = _esc(", ".join(raw_url))  # Join multiple URLs with comma
    else:
        doc_url = _esc(raw_url)
    
    if data.get("Industry"):
        stmts.append(
            f"MERGE (i:Industry {{name: '{_esc(data['Industry'])}'}}) SET i.source_url = '{doc_url}';"
        )
    if data.get("CustomerSegment"):
        stmts.append(
            f"MERGE (s:CustomerSegment {{name: '{_esc(data['CustomerSegment'])}'}}) SET s.source_url = '{doc_url}';"
        )
    for need in data.get("CustomerNeed", []) or []:
        stmts.append(
            f"MERGE (n:CustomerNeed {{name: '{_esc(need)}'}}) SET n.source_url = '{doc_url}';"
        )

    if data.get("HoneywellProduct"):
        stmts.append(
            f"MERGE (p:Product {{name: '{_esc(data['HoneywellProduct'])}'}}) "
            f"ON CREATE SET p.manufacturer = 'Honeywell' SET p.source_url = '{doc_url}';"
        )

    if data.get("Competitor"):
        stmts.append(
            f"MERGE (c:Company {{name: '{_esc(data['Competitor'])}'}}) SET c.source_url = '{doc_url}';"
        )

    if data.get("CompetitiveProduct"):
        manufacturer = _esc(data.get("Competitor", "")) if data.get("Competitor") else ""
        if manufacturer:
            stmts.append(
                f"MERGE (p:Product {{name: '{_esc(data['CompetitiveProduct'])}'}}) ON CREATE SET p.manufacturer = '{manufacturer}' SET p.source_url = '{doc_url}';"
            )
        else:
            stmts.append(
                f"MERGE (p:Product {{name: '{_esc(data['CompetitiveProduct'])}'}}) SET p.source_url = '{doc_url}';"
            )

    # Ensure nodes for any labels referenced only in Relationships
    for r in data.get("Relationships", []) or []:
        for label, name in ((r.get("source_type"), r.get("source")), (r.get("target_type"), r.get("target"))):
            if not name:
                continue
            raw = (label or "").strip()
            real_label = "Company" if raw == "Brand" else raw
            if not real_label:
                real_label = "Product"
            n = _esc(name)
            if real_label in ("Industry", "CustomerSegment", "CustomerNeed"):
                stmts.append(f"MERGE (x:{real_label} {{name: '{n}'}}) SET x.source_url = '{doc_url}';")
            elif real_label in ("Company",):
                stmts.append(f"MERGE (x:{real_label} {{name: '{n}'}}) SET x.source_url = '{doc_url}';")
            elif real_label == "Product":
                stmts.append(f"MERGE (x:Product {{name: '{n}'}}) SET x.source_url = '{doc_url}';")

    # Relationships
    for r in data.get("Relationships", []) or []:
        # Normalize label values: treat Brand as Company
        s_label_raw = (r.get("source_type") or "").strip()
        t_label_raw = (r.get("target_type") or "").strip()
        s_label = "Company" if s_label_raw == "Brand" else (s_label_raw or "Product")
        t_label = "Company" if t_label_raw == "Brand" else (t_label_raw or "Product")
        s_name = _esc(r.get("source") or "")
        t_name = _esc(r.get("target") or "")
        rel = _esc((r.get("relationship") or "").upper().replace(" ", "_"))
        if not (s_label and t_label and s_name and t_name and rel):
            continue
        stmts.append(
            "MATCH (s:" + s_label + " {name: '" + s_name + "'}), "
            "(t:" + t_label + " {name: '" + t_name + "'}) "
            "MERGE (s)-[r:" + rel + "]->(t) SET r.source_url = '" + doc_url + "';"
        )

    # Seed MUST-HAVE edge from top-level fields: CustomerSegment -> HAS_NEED -> CustomerNeed
    # No additional inferred relationships; mirror only JSON Relationships


    cypher = "\n".join(stmts)
    # Debug print
    try:
        print(f"[cypher_node] statements={len(stmts)}")
        print("[cypher_node] preview=", cypher[:300].replace("\n", " ") + ("..." if len(cypher) > 300 else ""))
    except Exception:
        pass
    return cypher


