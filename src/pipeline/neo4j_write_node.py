from __future__ import annotations

import os
from typing import Optional, Dict, Any
from neo4j import GraphDatabase, Driver

from src.pipeline.cypher_node import to_merge_cypher
from src.config.settings import get_neo4j_config


def _get_driver() -> Optional[Driver]:
    cfg = get_neo4j_config()
    uri = cfg.get("uri")
    user = cfg.get("user")
    pwd = cfg.get("password")
    if not (uri and user and pwd):
        print("[neo4j] NEO4J_URI/USER/PASSWORD not set; skipping DB write")
        return None
    return GraphDatabase.driver(uri, auth=(user, pwd))


def run_neo4j(cypher: str) -> None:
    if not cypher or not cypher.strip():
        print("[neo4j] empty cypher; nothing to run")
        return
    driver = _get_driver()
    if not driver:
        return
    with driver.session() as session:
        print("[neo4j] executing statements...")
        for stmt in cypher.split(";"):
            stmt = stmt.strip()
            if stmt:
                print("[neo4j] >>", (stmt[:120] + ("..." if len(stmt) > 120 else "")))
                session.run(stmt)
    driver.close()
    print("[neo4j] write complete")


def write_node(state: Dict[str, Any]) -> Dict[str, Any]:
    data = state.get("data", {})
    cypher = to_merge_cypher(data)
    print(cypher)  # optional for inspection
    run_neo4j(cypher)
    return state


