# Competitive Intelligence Database

Agentic (LangGraph) AI that discovers competitors, products, specs, prices, and reviews for pressure transmitters; the LLM chooses tool calls based on current state (competitors/products/specs/prices/reviews) and stops when thresholds are met. Results are written to Neo4j and visualized in Streamlit.

## LangGraph Overview
```
__start__ → agent (LLM) → router → tool nodes → agent (loop) → __end__

Core tools (agent chooses):
- search_web / search_with_variations           → find sources
- extract_page_content / extract_multiple_pages → pull pages + recurse relevant links
- save_competitor / save_product                → build companies/products
- save_multiple_specs                           → bulk specs (with unit normalization)
- save_price / save_review                      → commerce + sentiment
- search_evidence_store / get_search_suggestions→ recall evidence, fill gaps
- mark_complete                                 → graceful finish
```
![LangGraph Pipeline](langgraph_agentic_pipeline.png)

## Setup & Run
1) Create and activate env, then install deps:
```
# conda
conda create -n ci_db python=3.11 -y
conda activate ci_db

# OR venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

2) Create `.env` with keys and Neo4j creds:
```
OPENAI_API_KEY=...
TAVILY_API_KEY=...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

3) Run the agentic pipeline (recommended):
```
python -m src.pipeline.graph_builder --mode agentic --iterations 50
```

4) Launch the Streamlit dashboard:
```
streamlit run streamlit_app.py
```

5) Optional quick test:
```
python -m src.pipeline.graph_builder --mode agentic --iterations 10 --competitors 3
```

## Data Model & Ontology
- Specifications follow the pressure-transmitter ontology (`src/ontology/specifications.py`)
- Fuzzy matching on spec aliases (similarity > 0.6)
- Unit normalization stored alongside raw (pressure psi/bar/kPa, temp °C/°F/K, etc.)
- AI-derived attributes allowed when no canonical slot fits
- Neo4j graph:
  - Company ─[:OFFERS_PRODUCT]→ Product
  - Product ─[:HAS_SPEC]→ Specification (`value`, `normalized_value`, `unit`)
  - Product ─[:HAS_PRICE]→ Price
  - Product ─[:HAS_REVIEW]→ Review

## Evidence (ChromaDB)
- All extracted pages are chunked (800 chars, 100 overlap) and stored with metadata (URL, query, title, timestamp)
- Chunk IDs are attached to Neo4j relationships for traceability
- `search_evidence_store` can recall prior evidence

## Streamlit UI Highlights
- Knowledge graph view (larger canvas)
- Ontology tab explains specs, fuzzy matching, unit conversions, AI-derived attrs
- Ontology tab table shows raw vs normalized values from actual data
- Spec table & comparison matrix made Arrow-safe (string casting for list columns)






