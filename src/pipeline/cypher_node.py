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
    if data.get("Industry"):
        stmts.append(f"MERGE (:Industry {{name: '{_esc(data['Industry'])}'}});")
    if data.get("CustomerSegment"):
        stmts.append(f"MERGE (:CustomerSegment {{name: '{_esc(data['CustomerSegment'])}'}});")
    for need in data.get("CustomerNeed", []) or []:
        stmts.append(f"MERGE (:CustomerNeed {{name: '{_esc(need)}'}});")

    if data.get("HoneywellProduct"):
        stmts.append(
            f"MERGE (p:Product {{name: '{_esc(data['HoneywellProduct'])}'}}) "
            f"ON CREATE SET p.manufacturer = 'Honeywell';"
        )

    if data.get("Competitor"):
        stmts.append(f"MERGE (:Company {{name: '{_esc(data['Competitor'])}'}});")

    if data.get("CompetitiveProduct"):
        manufacturer = _esc(data.get("Competitor", "")) if data.get("Competitor") else ""
        if manufacturer:
            stmts.append(
                f"MERGE (p:Product {{name: '{_esc(data['CompetitiveProduct'])}'}}) ON CREATE SET p.manufacturer = '{manufacturer}';"
            )
        else:
            stmts.append(f"MERGE (:Product {{name: '{_esc(data['CompetitiveProduct'])}'}});")

    # Ensure nodes for any labels referenced only in Relationships
    for r in data.get("Relationships", []) or []:
        for label, name in ((r.get("source_type"), r.get("source")), (r.get("target_type"), r.get("target"))):
            if not name:
                continue
            label = (label or "").strip()
            n = _esc(name)
            if label in ("Industry", "CustomerSegment", "CustomerNeed"):
                stmts.append(f"MERGE (:{label} {{name: '{n}'}});")
            elif label in ("Company", "Brand"):
                stmts.append(f"MERGE (:{label} {{name: '{n}'}});")
            elif label == "Product":
                stmts.append(f"MERGE (:Product {{name: '{n}'}});")

    # Relationships
    for r in data.get("Relationships", []) or []:
        s_label = (r.get("source_type") or "").strip() or "Product"
        t_label = (r.get("target_type") or "").strip() or "Product"
        s_name = _esc(r.get("source") or "")
        t_name = _esc(r.get("target") or "")
        rel = _esc((r.get("relationship") or "").upper().replace(" ", "_"))
        if not (s_label and t_label and s_name and t_name and rel):
            continue
        stmts.append(
            "MATCH (s:" + s_label + " {name: '" + s_name + "'}), "
            "(t:" + t_label + " {name: '" + t_name + "'}) "
            "MERGE (s)-[:" + rel + "]->(t);"
        )

    cypher = "\n".join(stmts)
    # Debug print
    try:
        print(f"[cypher_node] statements={len(stmts)}")
        print("[cypher_node] preview=", cypher[:300].replace("\n", " ") + ("..." if len(cypher) > 300 else ""))
    except Exception:
        pass
    return cypher


