"""POML-style prompt construction helpers."""

from __future__ import annotations

import json
from html import escape
from typing import Any, Dict, List


def _render_value(value: Any) -> str:
    """Render structured values in a model-friendly way."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=True)
    return str(value)


def build_poml_prompt(
    *,
    role: str,
    objective: str,
    context: Dict[str, Any] | None = None,
    instructions: List[str] | None = None,
    constraints: List[str] | None = None,
    output_format: str | None = None,
) -> str:
    """
    Build a structured POML document as plain text.

    POML here is a lightweight XML schema that makes prompt sections explicit.
    """
    blocks: List[str] = [
        "<poml>",
        f"  <role>{escape(role)}</role>",
        f"  <objective>{escape(objective)}</objective>",
    ]

    if context:
        blocks.append("  <context>")
        for key, value in context.items():
            rendered = escape(_render_value(value))
            blocks.append(f'    <item name="{escape(str(key))}">{rendered}</item>')
        blocks.append("  </context>")

    if instructions:
        blocks.append("  <instructions>")
        for item in instructions:
            blocks.append(f"    <step>{escape(item)}</step>")
        blocks.append("  </instructions>")

    if constraints:
        blocks.append("  <constraints>")
        for item in constraints:
            blocks.append(f"    <rule>{escape(item)}</rule>")
        blocks.append("  </constraints>")

    if output_format:
        blocks.append(f"  <output_format>{escape(output_format)}</output_format>")

    blocks.append("</poml>")
    return "\n".join(blocks)
