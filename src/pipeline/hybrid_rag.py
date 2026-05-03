"""
Hybrid Neo4j + ChromaDB retrieval for interactive chat.

This module combines:
1. Structured retrieval from Neo4j for product/company/spec relationships
2. Evidence retrieval from ChromaDB for supporting source text
3. Optional LLM synthesis with explicit citations
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase

from src.config.settings import get_neo4j_config, get_openai_api_key
from src.pipeline.chroma_store import get_chunk_by_id, query_evidence


def _get_driver():
    cfg = get_neo4j_config()
    if not cfg["uri"] or not cfg["user"] or not cfg["password"]:
        raise RuntimeError(
            "Neo4j is not configured. Add NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD to your .env file."
        )
    return GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9][a-z0-9_./+-]*", (text or "").lower())


def _classify_question(question: str) -> str:
    q = question.lower()
    if any(term in q for term in ["compare", "vs", "versus"]):
        return "comparison"
    if any(term in q for term in ["customer need", "pain point", "requirement"]):
        return "customer_needs"
    if any(term in q for term in ["segment", "customer group", "who buys", "buyer"]):
        return "customer_segments"
    if any(term in q for term in ["competitor", "competes", "competition"]):
        return "competitors"
    if any(term in q for term in ["spec", "accuracy", "pressure", "temperature", "voltage"]):
        return "product_specs"
    return "general"


def _match_products(question: str, limit: int = 4) -> List[str]:
    question_lower = question.lower()
    driver = _get_driver()
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (:Company)-[:OFFERS_PRODUCT]->(p:Product)
                RETURN DISTINCT p.name AS product
                ORDER BY p.name
            """)
            products = [record["product"] for record in result if record["product"]]
    finally:
        driver.close()

    matches = [product for product in products if product.lower() in question_lower]
    if matches:
        return matches[:limit]

    question_tokens = set(_tokenize(question))
    scored = []
    for product in products:
        tokens = set(_tokenize(product))
        overlap = len(tokens.intersection(question_tokens))
        if overlap:
            scored.append((overlap, product))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [product for _, product in scored[:limit]]


def _extract_comparison_products(question: str, known_products: List[str]) -> List[str]:
    """Extract both product names from a comparison question using regex, then match against known products."""
    q = question.lower().strip()
    segments: List[str] = []

    patterns = [
        r"compare\s+(.+?)\s+(?:vs\.?|versus|with|and|to)\s+(.+?)(?:\?|$)",
        r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$)",
    ]
    for pattern in patterns:
        m = re.search(pattern, q)
        if m:
            segments = [m.group(1).strip(), m.group(2).strip()]
            break

    if not segments:
        return []

    results: List[str] = []
    seen: set = set()
    for segment in segments:
        seg_tokens = set(_tokenize(segment))
        best_product = None
        best_score = 0
        for product in known_products:
            if product in seen:
                continue
            prod_tokens = set(_tokenize(product))
            score = len(prod_tokens.intersection(seg_tokens))
            if score > best_score:
                best_score = score
                best_product = product
        if best_product and best_score > 0:
            results.append(best_product)
            seen.add(best_product)

    return results


def _fetch_product_details(products: List[str]) -> List[Dict[str, Any]]:
    if not products:
        return []

    driver = _get_driver()
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (c:Company)-[:OFFERS_PRODUCT]->(p:Product)
                WHERE p.name IN $products
                OPTIONAL MATCH (p)-[spec_rel:HAS_SPEC]->(s:Specification)
                OPTIONAL MATCH (p)-[need_rel:ADDRESSES_NEED]->(n:CustomerNeed)
                OPTIONAL MATCH (p)-[segment_rel:ADDRESSES_CUSTOMER_SEGMENT]->(seg:CustomerSegment)
                RETURN
                    c.name AS company,
                    p.name AS product,
                    collect(DISTINCT {
                        spec_type: s.spec_type,
                        value: s.value,
                        source_urls: spec_rel.source_urls,
                        evidence_ids: spec_rel.evidence_ids
                    }) AS specs,
                    collect(DISTINCT {
                        name: n.name,
                        threshold: n.threshold,
                        source_urls: need_rel.source_urls,
                        evidence_ids: need_rel.evidence_ids
                    }) AS needs,
                    collect(DISTINCT {
                        name: seg.name,
                        source_url: seg.source_url,
                        evidence_ids: segment_rel.evidence_ids
                    }) AS segments,
                    p.source_urls AS product_sources
                ORDER BY c.name, p.name
                """,
                products=products,
            )

            rows = []
            for record in result:
                specs = [
                    spec for spec in (record["specs"] or [])
                    if spec.get("spec_type") and spec.get("value")
                ]
                needs = [need for need in (record["needs"] or []) if need.get("name")]
                segments = [seg for seg in (record["segments"] or []) if seg.get("name")]
                rows.append(
                    {
                        "source_id": f"product::{record['product']}",
                        "kind": "product",
                        "company": record["company"],
                        "product": record["product"],
                        "product_sources": record["product_sources"] or [],
                        "specs": specs,
                        "needs": needs,
                        "segments": segments,
                    }
                )
            return rows
    finally:
        driver.close()


def _fetch_competitors() -> List[Dict[str, Any]]:
    driver = _get_driver()
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (h:Company {is_baseline: true})-[r:COMPETES_WITH]->(c:Company)
                OPTIONAL MATCH (c)-[:OFFERS_PRODUCT]->(p:Product)
                RETURN
                    c.name AS company,
                    collect(DISTINCT p.name) AS products,
                    r.source_urls AS source_urls,
                    r.evidence_ids AS evidence_ids
                ORDER BY c.name
            """)
            return [
                {
                    "source_id": f"competitor::{record['company']}",
                    "kind": "competitor",
                    "company": record["company"],
                    "products": [p for p in (record["products"] or []) if p],
                    "source_urls": record["source_urls"] or [],
                    "evidence_ids": record["evidence_ids"] or [],
                }
                for record in result
                if record["company"]
            ]
    finally:
        driver.close()


def _fetch_customer_needs(limit: int = 8) -> List[Dict[str, Any]]:
    driver = _get_driver()
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (p:Product)-[r:ADDRESSES_NEED]->(n:CustomerNeed)
                RETURN
                    n.name AS need,
                    n.threshold AS threshold,
                    collect(DISTINCT p.name) AS products,
                    r.source_urls AS source_urls,
                    r.evidence_ids AS evidence_ids
                ORDER BY n.name
                LIMIT $limit
                """,
                limit=limit,
            )
            return [
                {
                    "source_id": f"need::{record['need']}",
                    "kind": "customer_need",
                    "need": record["need"],
                    "threshold": record["threshold"],
                    "products": [p for p in (record["products"] or []) if p],
                    "source_urls": record["source_urls"] or [],
                    "evidence_ids": record["evidence_ids"] or [],
                }
                for record in result
                if record["need"]
            ]
    finally:
        driver.close()


def _fetch_customer_segments(limit: int = 8) -> List[Dict[str, Any]]:
    driver = _get_driver()
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (p:Product)-[r:ADDRESSES_CUSTOMER_SEGMENT]->(s:CustomerSegment)
                RETURN
                    s.name AS segment,
                    collect(DISTINCT p.name) AS products,
                    s.source_url AS source_url,
                    r.evidence_ids AS evidence_ids
                ORDER BY s.name
                LIMIT $limit
                """,
                limit=limit,
            )
            return [
                {
                    "source_id": f"segment::{record['segment']}",
                    "kind": "customer_segment",
                    "segment": record["segment"],
                    "products": [p for p in (record["products"] or []) if p],
                    "source_url": record["source_url"] or "",
                    "evidence_ids": record["evidence_ids"] or [],
                }
                for record in result
                if record["segment"]
            ]
    finally:
        driver.close()


def _retrieve_neo4j_context(question: str) -> List[Dict[str, Any]]:
    question_type = _classify_question(question)
    products = _match_products(question)

    if question_type == "comparison":
        comparison_products = _extract_comparison_products(question, products or [])
        target = comparison_products if len(comparison_products) >= 2 else (products[:2] if products else [])
        if target:
            return _fetch_product_details(target)
    if question_type == "product_specs" and products:
        return _fetch_product_details(products[:2])
    if question_type == "competitors":
        return _fetch_competitors()
    if question_type == "customer_needs":
        return _fetch_customer_needs()
    if question_type == "customer_segments":
        return _fetch_customer_segments()

    if products:
        return _fetch_product_details(products[:2])

    competitors = _fetch_competitors()[:4]
    needs = _fetch_customer_needs(limit=4)[:2]
    return competitors + needs


def _retrieve_chroma_context(question: str, neo4j_rows: List[Dict[str, Any]], limit: int = 6) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    seen_ids = set()

    for row in neo4j_rows:
        evidence_ids = row.get("evidence_ids") or []
        for spec in row.get("specs", []):
            evidence_ids.extend(spec.get("evidence_ids") or [])
        for need in row.get("needs", []):
            evidence_ids.extend(need.get("evidence_ids") or [])
        for seg in row.get("segments", []):
            evidence_ids.extend(seg.get("evidence_ids") or [])

        for evidence_id in evidence_ids[:6]:
            if evidence_id in seen_ids:
                continue
            chunk = get_chunk_by_id(evidence_id)
            if chunk:
                chunks.append(
                    {
                        "id": chunk["id"],
                        "document": chunk["document"],
                        "metadata": chunk.get("metadata", {}),
                        "retrieval_method": "linked_evidence",
                    }
                )
                seen_ids.add(evidence_id)
            if len(chunks) >= limit:
                return chunks[:limit]

    for item in query_evidence(question, n_results=limit):
        if item["id"] in seen_ids:
            continue
        chunks.append(
            {
                "id": item["id"],
                "document": item["document"],
                "metadata": item.get("metadata", {}),
                "distance": item.get("distance"),
                "retrieval_method": "semantic_search",
            }
        )
        seen_ids.add(item["id"])
        if len(chunks) >= limit:
            break

    return chunks[:limit]


def _build_structured_context(rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for index, row in enumerate(rows, start=1):
        label = f"S{index}"
        if row["kind"] == "product":
            spec_lines = [f"{spec['spec_type']}: {spec['value']}" for spec in row.get("specs", [])[:8]]
            need_lines = [need["name"] for need in row.get("needs", [])[:5]]
            segment_lines = [seg["name"] for seg in row.get("segments", [])[:5]]
            lines.append(
                "\n".join(
                    [
                        f"[{label}] Product record",
                        f"Company: {row.get('company', 'Unknown')}",
                        f"Product: {row.get('product', 'Unknown')}",
                        f"Specifications: {'; '.join(spec_lines) if spec_lines else 'None'}",
                        f"Customer needs linked: {', '.join(need_lines) if need_lines else 'None'}",
                        f"Customer segments linked: {', '.join(segment_lines) if segment_lines else 'None'}",
                    ]
                )
            )
        elif row["kind"] == "competitor":
            lines.append(
                "\n".join(
                    [
                        f"[{label}] Competitor record",
                        f"Company: {row.get('company', 'Unknown')}",
                        f"Products: {', '.join(row.get('products', [])) or 'None'}",
                    ]
                )
            )
        elif row["kind"] == "customer_need":
            lines.append(
                "\n".join(
                    [
                        f"[{label}] Customer need record",
                        f"Need: {row.get('need', 'Unknown')}",
                        f"Threshold: {row.get('threshold', '') or 'Not specified'}",
                        f"Products: {', '.join(row.get('products', [])) or 'None'}",
                    ]
                )
            )
        elif row["kind"] == "customer_segment":
            lines.append(
                "\n".join(
                    [
                        f"[{label}] Customer segment record",
                        f"Segment: {row.get('segment', 'Unknown')}",
                        f"Products: {', '.join(row.get('products', [])) or 'None'}",
                    ]
                )
            )
    return "\n\n".join(lines)


def _build_chroma_context(chunks: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        lines.append(
            "\n".join(
                [
                    f"[C{index}] Evidence chunk",
                    f"Chunk ID: {chunk.get('id', '')}",
                    f"Source URL: {metadata.get('source_url', 'n/a')}",
                    f"Text: {chunk.get('document', '')[:700]}",
                ]
            )
        )
    return "\n\n".join(lines)


def _fallback_answer(question: str, question_type: str, neo4j_rows: List[Dict[str, Any]], chroma_chunks: List[Dict[str, Any]]) -> str:
    if question_type == "comparison" and neo4j_rows:
        parts = []
        for index, row in enumerate(neo4j_rows[:2], start=1):
            specs = row.get("specs", [])[:4]
            spec_summary = ", ".join(f"{spec['spec_type']}={spec['value']}" for spec in specs) or "no specs found"
            parts.append(f"{row.get('product')} from {row.get('company')} has {spec_summary} [Neo4j S{index}]")
        if chroma_chunks:
            parts.append("Supporting source text is available in " + ", ".join(f"[Chroma C{i}]" for i in range(1, min(len(chroma_chunks), 3) + 1)) + ".")
        return " ".join(parts)

    if question_type == "competitors" and neo4j_rows:
        names = [f"{row.get('company')} [Neo4j S{index}]" for index, row in enumerate(neo4j_rows[:5], start=1)]
        return "Honeywell competitors in the graph include " + ", ".join(names) + "."

    if question_type == "customer_needs" and neo4j_rows:
        needs = [f"{row.get('need')} [Neo4j S{index}]" for index, row in enumerate(neo4j_rows[:5], start=1)]
        return "Customer needs captured in the graph include " + ", ".join(needs) + "."

    if neo4j_rows and neo4j_rows[0]["kind"] == "product":
        row = neo4j_rows[0]
        specs = ", ".join(f"{spec['spec_type']}={spec['value']}" for spec in row.get("specs", [])[:6]) or "no specs found"
        return f"{row.get('product')} from {row.get('company')} has the following specs in Neo4j: {specs} [Neo4j S1]."

    if chroma_chunks:
        return f"I could not find enough structured graph data, but Chroma evidence suggests: {chroma_chunks[0].get('document', '')[:280]} [Chroma C1]"

    return "I could not find enough data in Neo4j or ChromaDB to answer that question yet."


def _answer_with_llm(question: str, question_type: str, neo4j_rows: List[Dict[str, Any]], chroma_chunks: List[Dict[str, Any]]) -> str:
    llm = ChatOpenAI(
        api_key=get_openai_api_key(),
        model="gpt-4.1-mini",
        temperature=0,
        timeout=60,
        max_retries=2,
    )

    prompt = f"""Answer the user's question using only the context below.

Rules:
- Use structured Neo4j facts when available.
- Use Chroma evidence as supporting proof.
- Cite every factual claim with citations like [Neo4j S1] or [Chroma C2].
- If the data is incomplete, say so.
- Keep the answer concise and helpful.

Question type: {question_type}
Question: {question}

Neo4j context:
{_build_structured_context(neo4j_rows) or 'None'}

Chroma context:
{_build_chroma_context(chroma_chunks) or 'None'}
"""

    response = llm.invoke(prompt)
    return response.content.strip()


def answer_hybrid_question(question: str) -> Dict[str, Any]:
    question_type = _classify_question(question)

    neo4j_error: str = ""
    neo4j_rows: List[Dict[str, Any]] = []
    try:
        neo4j_rows = _retrieve_neo4j_context(question)
    except Exception as exc:
        neo4j_error = str(exc)

    chroma_chunks = _retrieve_chroma_context(question, neo4j_rows)

    llm_used = True
    try:
        answer = _answer_with_llm(question, question_type, neo4j_rows, chroma_chunks)
    except Exception:
        llm_used = False
        answer = _fallback_answer(question, question_type, neo4j_rows, chroma_chunks)

    result: Dict[str, Any] = {
        "question": question,
        "question_type": question_type,
        "answer": answer,
        "llm_used": llm_used,
        "neo4j_sources": neo4j_rows,
        "chroma_sources": chroma_chunks,
    }
    if neo4j_error:
        result["neo4j_error"] = neo4j_error
    return result


__all__ = ["answer_hybrid_question"]
