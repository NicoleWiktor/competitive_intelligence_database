# Competitive Intelligence Database

An agentic AI system built with LangGraph that autonomously researches Honeywell's competitors in the pressure transmitter market. The LLM decides which tools to call, what to search, and when to stop. Results are stored in Neo4j (structured graph) and ChromaDB (evidence chunks for human in the loop verification).

## Architecture

The system uses a LangGraph StateGraph with two nodes (`agent` and `tools`) in a loop:

1. `agent` node calls the LLM with bound tools. LLM decides which tools to call.
2. `should_continue` checks if LLM returned tool calls. If yes → go to tools, if no → end.
3. `tools` node executes the tool calls, results go back to agent, repeat until done.

**Tools (LLM chooses which to call):**

| Tool | Purpose |
|------|---------|
| `search_web` | Tavily web search |
| `extract_page_content` | Tavily page extraction + stores chunks in ChromaDB |
| `save_competitor` | Saves company with evidence link |
| `save_product` | Saves product + specs with evidence link |
| `research_industry_needs` | Searches 8+ sources, generates comprehensive needs report |
| `map_needs_from_report` | Extracts needs from report and maps to products |
| `get_current_progress` | Returns current research status |
| `finish_research` | Signals completion |

**Data Storage:**

| Store | Purpose |
|-------|---------|
| **ChromaDB** | Raw text chunks from web pages (evidence for verification) |
| **Neo4j** | Structured knowledge graph (Companies, Products, Specifications, CustomerNeeds) |

## How It Works

**Agent Loop:**
1. LLM receives the conversation history and decides which tools to call
2. If LLM returns tool calls → execute them, add results to conversation, go back to step 1
3. If LLM returns no tool calls (or calls `finish_research`) → end
4. Final data written to Neo4j

**Research Strategy (Two Phases):**
- **Phase 1**: Find competitors and their products with specs
- **Phase 2**: Generate comprehensive industry needs report (from 8+ sources), then map needs to product specs

**Graph Structure:**
```
Honeywell ─COMPETES_WITH→ Competitor ─OFFERS_PRODUCT→ Product ─HAS_SPEC→ Specification
                                                          │
                                                          └─ADDRESSES_NEED→ CustomerNeed
```


## Setup

### 1. Create Environment

```bash
# Using conda (recommended)
conda create -n ci_db python=3.11 -y
conda activate ci_db

# OR using venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### 4. Start Neo4j

## Usage

### Run the Pipeline

```bash
# Default: 5 competitors, 25 iterations, process industries
python main.py

# Specify number of competitors (max 10)
python main.py --competitors 10

# Specify target industry
python main.py --industry "oil and gas"
python main.py --industry "chemical processing"

# Limit agent iterations (default 25)
python main.py --iterations 15

# Keep existing data (incremental mode)
python main.py --competitors 5 --incremental

# Combine options
python main.py --competitors 5 --industry "oil and gas" --iterations 20
```

### Launch Dashboard

```bash
python main.py --streamlit

# Or directly:
streamlit run streamlit_app.py
```

### Verify Evidence

Evidence verification is done through the Streamlit dashboard's **"✅ Verify Data"** tab, which:
- Shows all relationships from Neo4j
- Retrieves original source evidence from ChromaDB
- Displays the exact text and source URL for human verification


## Streamlit Dashboard Features

| Tab | Description |
|-----|-------------|
| 📊 Knowledge Graph | Interactive visualization of the Neo4j graph |
| 🔄 Pipeline Architecture | Shows how LangGraph agent works |
| 📚 Ontology | Spec definitions and normalization rules |
| 📋 Specification Table | All products and their specs in a table |
| 🔍 Compare Products | Side-by-side product comparison |
| ✅ Verify Data | Human verification with ChromaDB evidence |
| 📊 Industry Needs Report | Full report generated from 8+ sources |
| 🎯 Customer Needs | Extracted needs and their product mappings |


