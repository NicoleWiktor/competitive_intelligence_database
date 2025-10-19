from __future__ import annotations

import json
from typing import Any, Dict, List, TypedDict
from pathlib import Path

from langgraph.graph import START, StateGraph

from src.pipeline.query_node import search_node
from src.pipeline.extract_node import extract_node
from src.pipeline.llm_node import llm_state_node
from src.pipeline.neo4j_write_node import write_node


class PipelineState(TypedDict, total=False):
    query: str
    max_results: int
    schema: Dict[str, Any]
    results: List[Dict[str, Any]]
    data: Dict[str, Any]

    


def build_graph() -> Any:
    g = StateGraph(PipelineState)
    g.add_node("search", search_node)
    g.add_node("extract", extract_node)
    g.add_node("llm", llm_state_node)
    g.add_node("write", write_node)
    g.add_edge(START, "search")
    g.add_edge("search", "extract")
    g.add_edge("extract", "llm")
    g.add_edge("llm", "write")
    return g.compile()




if __name__ == "__main__":
    app = build_graph()
    # Put your Tavily query here:
    QUERY = "pressure transmitters in process industries with customer reviews and brands"
    # Optional: change number of results
    MAX_RESULTS = 3
    # Load schema from file
    schema_path = Path("src/schemas/schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    state: PipelineState = {"query": QUERY, "max_results": MAX_RESULTS, "schema": schema}
    out = app.invoke(state)
    print(json.dumps(out.get("data", {}), indent=2))
