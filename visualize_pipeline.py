#!/usr/bin/env python3
"""
Visualize the LangGraph pipeline without running the data pipeline.

How to run:
  python visualize_pipeline.py

If PNG rendering isn't available, this script will print an ASCII or Mermaid diagram
and tell you how to preview it (e.g., paste Mermaid into https://mermaid.live).
"""

import sys

try:
    from IPython.display import Image, display  # type: ignore
except Exception:  # IPython may not exist in non-notebook envs; we handle gracefully
    Image = None
    display = None

try:
    from src.pipeline.graph_builder import build_graph
except Exception as e:
    print("Failed to import build_graph from src.pipeline.graph_builder:", e)
    sys.exit(1)


def main() -> None:
    app = build_graph()

    # Try to render a PNG using LangGraph's Mermaid renderer
    try:
        graph = app.get_graph()
        if hasattr(graph, "draw_mermaid_png"):
            png_bytes = graph.draw_mermaid_png()
            if Image and display:
                display(Image(png_bytes))
                return
            else:
                # Save to file if we're not in a notebook
                out_path = "langgraph_pipeline.png"
                with open(out_path, "wb") as f:
                    f.write(png_bytes)
                print(f"Saved pipeline PNG to {out_path}")
                return
        else:
            raise AttributeError("draw_mermaid_png not available on this graph")
    except Exception as e:
        print(f"Graph PNG visualization not available: {e}")

    # Fallback 1: ASCII
    try:
        ascii_diagram = app.get_graph().draw_ascii()
        print("\nASCII pipeline diagram:\n")
        print(ascii_diagram)
        return
    except Exception as e:
        print(f"ASCII visualization not available: {e}")

    # Fallback 2: Mermaid text
    try:
        mermaid_diagram = app.get_graph().draw_mermaid()
        print("\nMermaid diagram (paste into https://mermaid.live):\n")
        print(mermaid_diagram)
        return
    except Exception as e:
        print(f"Mermaid diagram not available: {e}")
        print("No visualization method available.")


if __name__ == "__main__":
    main()
