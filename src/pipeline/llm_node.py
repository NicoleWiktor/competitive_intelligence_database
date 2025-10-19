from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
import re

from src.config.settings import get_openai_api_key


def _load_schema() -> Dict[str, Any]:
    return json.loads(Path("src/schemas/schema.json").read_text(encoding="utf-8"))


def _make_llm() -> ChatOpenAI:
    # Enforce JSON object responses to reduce parsing errors
    return ChatOpenAI(
        api_key=get_openai_api_key(),
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
    )


def _build_prompt(chunk_text: str, schema: Dict[str, Any]) -> str:
    return (
        "Analyze the text. Then, extract structured data as JSON in EXACTLY this schema (keys only, no extras).\n"
        + json.dumps(schema)
        + "\n\nText to analyze:\n"
        + chunk_text
        + "\n\nReturn ONLY the filled in JSON based on the text, no other text or commentary. If content does not exist in the text, return a empty stinge in the schema."
    )


def extract_with_schema(schema: Dict[str, Any], raw_content: str, source_url: str) -> Dict[str, Any]:
    """Primary entry: receive schema dict and raw text; return schema-shaped JSON."""
    llm = _make_llm()
    prompt = _build_prompt(raw_content, schema)
    msg = llm.invoke(prompt)
    txt = getattr(msg, "content", str(msg))
    # Debug: show a small portion of the raw LLM text
    print("[llm_state_node] raw_llm_preview=", (txt or "")[:200].replace("\n", " "))

    # Robust JSON coercion
    data = _coerce_to_json(txt, schema)
    if isinstance(data, dict):
        data["Doc"] = {"source_url": source_url}
    return data


def _coerce_to_json(text: str, fallback_schema: Dict[str, Any]) -> Dict[str, Any]:
    if not text:
        return {**fallback_schema}
    # 1) direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) fenced code block ```json ... ```
    m = re.search(r"```json\s*([\s\S]*?)```", text)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # 3) first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return {**fallback_schema}


def llm_state_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph-friendly node: take schema + results, run LLM once, return data."""
    schema = state.get("schema") 
    results = state.get("results", []) or []
    if not results:
        return {"data": schema}
    # Concatenate all Tavily outputs (raw_content or content) into one bundle
    parts: List[str] = []
    urls: List[str] = []
    for r in results:
        text = r.get("raw_content") or r.get("content") or ""
        if text:
            url = r.get("url", "")
            urls.append(url)
            parts.append(f"SOURCE: {url}\n{text}")
    raw_bundle = "\n\n---\n\n".join(parts)
    source_url = urls[0] if urls else ""
    data = extract_with_schema(schema, raw_bundle, source_url)
    # Brief debug only (final JSON is printed by the runner)
    print(f"[llm_state_node] inputs={len(results)} bundle_len={len(raw_bundle)}")
    return {"data": data}





