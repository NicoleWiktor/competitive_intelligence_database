"""
Streamlit App - Competitive Intelligence Dashboard

A beautiful, agentic-powered dashboard for Honeywell competitive intelligence.

Features:
1. 📊 Interactive Knowledge Graph Visualization
2. 📋 Product Specification Comparison Table
3. 🔍 Head-to-Head Product Comparison
4. ✅ Human-in-the-Loop Verification
5. 🤖 Run Agentic Pipeline from UI
"""

import streamlit as st
import streamlit.components.v1 as components
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from pyvis.network import Network
import tempfile
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import get_neo4j_config
from src.pipeline.chroma_store import find_best_evidence_for_relationship, get_chunk_by_id
from src.ontology.specifications import PRESSURE_TRANSMITTER_ONTOLOGY

# =============================================================================
# PAGE CONFIG & STYLES
# =============================================================================

st.set_page_config(
    page_title="Competitive Intelligence Database",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling (neutral light greys)
st.markdown("""
<style>
    .main { background: linear-gradient(180deg, #f7f7f8 0%, #f1f3f5 100%); }
    .main-header {
        background: linear-gradient(135deg, #f5f5f6 0%, #e5e7eb 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.10);
        border: 1px solid #e5e7eb;
    }
    .main-header h1 { font-family: 'Inter', sans-serif; letter-spacing: 0.5px; color: #111827; margin: 0; }
    .main-header p { color: #374151; margin: 0.2rem 0 0; font-size: 1rem; }
    .section-header {
        background: linear-gradient(90deg, #f5f6f7 0%, #e8ebef 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 2rem 0 1rem 0;
        border-left: 4px solid #9ca3af;
    }
    .section-header h2 { color: #111827; margin: 0; font-size: 1.3rem; font-weight: 600; }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        color: #111827;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #111827; }
    .metric-label { font-size: 0.9rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }
    .dataframe { background: #ffffff !important; border-radius: 8px; overflow: hidden; }
    .dataframe th { background: #f3f4f6 !important; color: #111827 !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.5px; padding: 12px 16px !important; }
    .dataframe td { background: #ffffff !important; color: #1f2937 !important; padding: 10px 16px !important; border-bottom: 1px solid #e5e7eb !important; }
    .dataframe tr:hover td { background: #f3f4f6 !important; }
    .comparison-winner { background: #e5e7eb !important; color: #111827 !important; font-weight: 600; }
    .comparison-loser { background: #f9fafb !important; color: #6b7280 !important; }
    .streamlit-expanderHeader { background: #f3f4f6; border-radius: 8px; }
    .css-1d391kg { background: linear-gradient(180deg, #f7f7f8 0%, #f1f3f5 100%); }
    .css-1d391kg p, .css-1d391kg label { color: #111827 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #e5e7eb 0%, #d1d5db 100%);
        color: #111827;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #d1d5db 0%, #9ca3af 100%);
        box-shadow: 0 4px 15px rgba(156, 163, 175, 0.35);
        transform: translateY(-1px);
    }
    .stDataFrame { background: #ffffff; border-radius: 10px; border: 1px solid #e5e7eb; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #f3f4f6;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        color: #4b5563;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #111827; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #e5e7eb 0%, #d1d5db 100%);
        color: #111827;
    }
</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def get_neo4j_driver():
    """Create Neo4j connection."""
    cfg = get_neo4j_config()
    return GraphDatabase.driver(cfg['uri'], auth=(cfg['user'], cfg['password']))


def fetch_all_products_with_specs() -> pd.DataFrame:
    """Fetch all products with their specifications for the comparison table."""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Company)-[:OFFERS_PRODUCT]->(p:Product)
            OPTIONAL MATCH (p)-[:HAS_SPEC]->(s:Specification)
            OPTIONAL MATCH (p)-[:HAS_PRICE]->(price:Price)
            OPTIONAL MATCH (p)-[:HAS_REVIEW]->(r:Review)
            RETURN 
                c.name as company,
                p.name as product,
                collect(DISTINCT {spec_type: s.spec_type, value: s.value}) as specifications,
                price.name as price,
                p.source_urls as sources,
                collect(DISTINCT {text: r.text, rating: r.rating, source: r.source}) as reviews
            ORDER BY c.name, p.name
        """)
        
        rows = []
        for record in result:
            row = {
                'Company': record['company'],
                'Product': record['product'],
                'Price': record['price'] or '-',
                'Sources': record['sources'] or [],
                'Review Count': len([rv for rv in record['reviews'] or [] if rv.get('text')]),
            }
            
            # First review snippet
            first_review = None
            if record['reviews']:
                for rv in record['reviews']:
                    if rv.get('text'):
                        first_review = rv
                        break
            if first_review:
                snippet = (first_review.get('text', '') or '')[:120]
                rating = first_review.get('rating', '')
                source = first_review.get('source', '')
                row['Review Snippet'] = f"{rating} - {snippet} ({source})" if rating else f"{snippet} ({source})"
            else:
                row['Review Snippet'] = ''
            
            # Flatten specifications
            for spec in record['specifications']:
                if spec['spec_type'] and spec['value']:
                    spec_display = spec['spec_type'].replace('_', ' ').title()
                    row[spec_display] = spec['value']
            
            rows.append(row)
    
    driver.close()
    
    if rows:
        df = pd.DataFrame(rows)
        return df
    return pd.DataFrame()


def fetch_graph_data():
    """Fetch all nodes and relationships for visualization."""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (source)-[rel]->(target)
            RETURN 
                elementId(source) as source_id,
                labels(source)[0] as source_label,
                source.name as source_name,
                type(rel) as relationship_type,
                elementId(target) as target_id,
                labels(target)[0] as target_label,
                target.name as target_name,
                rel.source_urls as rel_sources,
                rel.evidence_ids as rel_evidence,
                rel.snippet as rel_snippet
        """)
        
        nodes = {}
        edges = []
        
        for record in result:
            source_id = record['source_id']
            if source_id not in nodes:
                source_name = record['source_name']
                source_group = record['source_label']
                # For specs/prices/reviews, strip the product prefix before the pipe for a cleaner label
                if source_group in ["Specification", "Price", "Review"] and "|" in source_name:
                    clean_source_label = source_name.split("|", 1)[1].strip()
                else:
                    clean_source_label = source_name
                nodes[source_id] = {
                    'id': source_id,
                    'label': clean_source_label[:30],
                    'title': clean_source_label,
                    'group': source_group,
                    'full_name': source_name
                }
            
            target_id = record['target_id']
            if target_id not in nodes:
                target_name = record['target_name']
                target_group = record['target_label']
                if target_group in ["Specification", "Price", "Review"] and "|" in target_name:
                    clean_target_label = target_name.split("|", 1)[1].strip()
                else:
                    clean_target_label = target_name
                nodes[target_id] = {
                    'id': target_id,
                    'label': clean_target_label[:30],
                    'title': clean_target_label,
                    'group': target_group,
                    'full_name': target_name
                }
            
            edges.append({
                'from': source_id,
                'to': target_id,
                'label': record['relationship_type'],
                'title': record['relationship_type'],
                'sources': record.get('rel_sources') or [],
                'evidence_ids': record.get('rel_evidence') or [],
                'snippet': record.get('rel_snippet', "") or "",
            })
    
    driver.close()
    return list(nodes.values()), edges


def fetch_graph_stats() -> Dict[str, int]:
    """Get statistics about the knowledge graph."""
    driver = get_neo4j_driver()
    with driver.session() as session:
        stats = {}
        
        result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] as label, count(n) as count
        """)
        stats['nodes'] = {record['label']: record['count'] for record in result}
        
        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) as type, count(r) as count
        """)
        stats['relationships'] = {record['type']: record['count'] for record in result}
    
    driver.close()
    return stats


def _is_valid_http_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    return url.startswith("http://") or url.startswith("https://")


def fetch_all_relationships():
    """Fetch all relationships for verification."""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (source)-[rel]->(target)
            RETURN 
                elementId(rel) as rel_id,
                labels(source)[0] as source_label,
                source.name as source_name,
                type(rel) as relationship_type,
                labels(target)[0] as target_label,
                target.name as target_name,
                rel.source_urls as source_urls,
                rel.evidence_ids as evidence_ids,
                rel.snippet as snippet
            ORDER BY relationship_type, source_name
        """)
        
        relationships = []
        for record in result:
            relationships.append({
                'rel_id': record['rel_id'],
                'source_label': record['source_label'],
                'source_name': record['source_name'],
                'relationship_type': record['relationship_type'],
                'target_label': record['target_label'],
                'target_name': record['target_name'],
                'source_urls': record['source_urls'] or [],
                'evidence_ids': record['evidence_ids'] or [],
                'snippet': record['snippet'] or ""
            })
    
    driver.close()
    return relationships


def delete_relationships(rel_ids: List[int]):
    """Delete relationships from Neo4j."""
    driver = get_neo4j_driver()
    with driver.session() as session:
        for rel_id in rel_ids:
            session.run("""
                MATCH ()-[r]->()
                WHERE id(r) = $rel_id
                DELETE r
            """, rel_id=rel_id)
        driver.close()


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_network_graph(nodes, edges):
    """Create beautiful interactive network graph with white background."""
    net = Network(
        height="820px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#1f2937",
        directed=True
    )
    
    net.set_options("""
    {
        "layout": {
            "hierarchical": {
                "enabled": false
            }
        },
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -25000,
                "centralGravity": 0.3,
                "springLength": 200,
                "springConstant": 0.04,
                "damping": 0.6
            },
            "stabilization": {
                "enabled": true,
                "iterations": 200
            }
        },
        "nodes": {
            "font": {"size": 14, "face": "Inter, sans-serif", "color": "#1f2937"},
            "borderWidth": 3,
            "borderWidthSelected": 4,
            "shadow": {
                "enabled": true,
                "color": "rgba(0,0,0,0.15)",
                "size": 10
            }
        },
        "edges": {
            "font": {"size": 11, "align": "middle", "color": "#6b7280"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.6}},
            "smooth": {"type": "continuous"},
            "width": 2,
            "color": {"color": "#9ca3af", "highlight": "#3b82f6"}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true
        }
    }
    """)
    
    # Neutral color scheme
    colors = {
        'Company': {'background': '#1f2937', 'border': '#111827'},
        'Product': {'background': '#4b5563', 'border': '#374151'},
        'Price': {'background': '#6b7280', 'border': '#4b5563'},
        'Specification': {'background': '#9ca3af', 'border': '#6b7280'},
        'Review': {'background': '#d1d5db', 'border': '#9ca3af'}
    }
    
    # Special color for Honeywell (center node)
    honeywell_color = {'background': '#111827', 'border': '#030712'}
    
    for node in nodes:
        group = node['group']
        # Only treat the Honeywell COMPANY as the fixed center node to avoid overlap
        label_text = node.get('label', '') or ''
        is_honeywell = (group == 'Company') and ('Honeywell' in label_text)
        
        # Uniform node sizing for readability
        size = 50 if is_honeywell else 35
        color = honeywell_color if is_honeywell else colors.get(group, {'background': '#6b7280', 'border': '#9ca3af'})
        font_cfg = {'size': 18, 'color': '#1f2937', 'bold': True} if is_honeywell else {'size': 13, 'color': '#1f2937'}
        
        net.add_node(
            node['id'],
            label=node['label'],
            title=node['title'],
            color=color,
            group=group,
            size=size,
            font=font_cfg,
            x=0 if is_honeywell else None,
            y=0 if is_honeywell else None,
            fixed={'x': True, 'y': True} if is_honeywell else None,
            physics=not is_honeywell
        )
    
    for edge in edges:
        net.add_edge(
            edge['from'],
            edge['to'],
            label=edge['label'].replace('_', ' '),
            title=edge['title']
        )
    
    return net


def create_comparison_table(products_df: pd.DataFrame, selected_products: List[str]) -> pd.DataFrame:
    """Create a head-to-head comparison table for selected products."""
    if not selected_products or len(selected_products) < 2:
        return pd.DataFrame()
    
    # Filter to selected products
    comparison_df = products_df[products_df['Product'].isin(selected_products)].copy()
    
    if comparison_df.empty:
        return pd.DataFrame()
    
    # Transpose for comparison view
    comparison_df = comparison_df.set_index('Product').T
    
    return comparison_df


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Initialize session state
    if 'selected_items' not in st.session_state:
        st.session_state.selected_items = set()
    if 'verified_relationships' not in st.session_state:
        st.session_state.verified_relationships = set()
    if 'rejected_relationships' not in st.session_state:
        st.session_state.rejected_relationships = set()
    if 'selected_products' not in st.session_state:
        st.session_state.selected_products = []
    
    # === HEADER ===
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Competitive Intelligence Database</h1>
        <p>Competitive Intelligence for Pressure Transmitter Markets | Powered by Agentic AI</p>
    </div>
    """, unsafe_allow_html=True)
        
    # === SIDEBAR ===
    with st.sidebar:
        st.markdown("### 🤖 Agentic Pipeline")
        st.markdown("Run the AI agent to collect competitive intelligence.")
        
        target_product = st.text_input("Target Product", "SmartLine ST700")
        max_competitors = st.slider("Max Competitors", 1, 10, 5)
        max_iterations = st.slider("Max Iterations", 10, 50, 30)
        
        if st.button("🚀 Run Agentic Pipeline", width="stretch"):
            with st.spinner("🤖 Agent is researching..."):
                try:
                    from src.pipeline.graph_builder import run_pipeline
                    result = run_pipeline(
                        max_competitors=max_competitors,
                    )
                    st.success("✅ Pipeline complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        
        try:
            stats = fetch_graph_stats()
            for label, count in stats.get('nodes', {}).items():
                st.metric(label, count)
        except:
            st.info("Connect to Neo4j to see stats")
    
    # === MAIN CONTENT TABS ===
    tabs = st.tabs([
        "📊 Knowledge Graph", 
        "🔄 Pipeline Architecture",
        "📚 Ontology",
        "📋 Specification Table", 
        "🔍 Compare Products",
        "✅ Verify Data",
        "🎯 Customer Needs"
    ])
    
    # === TAB 1: KNOWLEDGE GRAPH ===
    with tabs[0]:
        st.markdown("""
        <div class="section-header">
            <h2>📊 Knowledge Graph Visualization</h2>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            nodes, edges = fetch_graph_data()
            
            if nodes and edges:
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    companies = len([n for n in nodes if n['group'] == 'Company'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{companies}</div>
                        <div class="metric-label">Companies</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    products = len([n for n in nodes if n['group'] == 'Product'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{products}</div>
                        <div class="metric-label">Products</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    specs = len([n for n in nodes if n['group'] == 'Specification'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{specs}</div>
                        <div class="metric-label">Specifications</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(edges)}</div>
                        <div class="metric-label">Relationships</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
            
                # Legend
                with st.expander("🎨 Graph Legend & Controls"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("""
                        **Node Types:**
                        - 🔵 **Company** - Competitors
                        - 🟡 **Product** - Product models
                        - 🟢 **Price** - Pricing data
                        - 🔴 **Specification** - Technical specs
                        - 🟣 **Review** - Customer reviews
                        """)
                    with col2:
                        st.markdown("""
                        **Controls:**
                        - **Drag** nodes to rearrange
                        - **Scroll** to zoom
                        - **Click** to highlight connections
                        - **Double-click** to focus
                        """)
                
                # Render graph
                net = create_network_graph(nodes, edges)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
                    net.save_graph(f.name)
                    with open(f.name, 'r', encoding='utf-8') as f2:
                        html_content = f2.read()
                
                components.html(html_content, height=850, scrolling=False)
            
            else:
                st.info("📊 No graph data yet. Run the pipeline to generate data.")
    
        except Exception as e:
            st.warning(f"⚠️ Could not load graph: {str(e)}")
    
    # === TAB 2: PIPELINE ARCHITECTURE ===
    with tabs[1]:
        st.markdown("""
        <div class="section-header">
            <h2>🔄 LangGraph Agentic Pipeline</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("This shows the architecture of the agentic AI pipeline that collects competitive intelligence.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Pipeline Visualization")
            
            # Try to load and display the pipeline image
            try:
                # Generate pipeline diagram from the LangGraph agent
                try:
                    from src.agents.agentic_agent import build_graph
                    graph = build_graph()
                    png_bytes = graph.get_graph().draw_mermaid_png()
                    st.image(png_bytes, caption="LangGraph Agentic Pipeline", width="stretch")
                except Exception as e:
                    # Fallback to text diagram
                    st.code("""
__start__
    │
    ▼
┌─────────┐
│  agent  │ ←── LLM decides tool calls
└────┬────┘
     │
     ▼ (conditional)
┌─────────┐
│  tools  │ ←── Executes: search_web, extract_page_content, etc.
└────┬────┘
     │
     └──────→ back to agent (loop)
                    """, language="text")
                    st.caption(f"Live diagram unavailable: {e}")
            except Exception as e:
                st.warning(f"Could not load pipeline visualization: {e}")
        
        with col2:
            st.markdown("### How It Works")
            
            st.markdown("""
            **The LangGraph Agent Loop:**
            
            ```
            __start__
                │
                ▼
            ┌─────────┐
            │  agent  │◄────────┐
            │ (LLM)   │         │
            └────┬────┘         │
                 │              │
                 ▼              │
            ┌─────────┐         │
            │  tools  │─────────┘
            └────┬────┘
                 │ (when complete)
                 ▼
            ┌─────────┐
            │ __end__ │ → Neo4j
            └─────────┘
            ```
            """)
            
        st.markdown("---")
                
        st.markdown("### Available Tools")
        
        tools_info = {
            "🔍 search_web": "Search for competitors, products, specs",
            "📄 extract_page": "Get full content from a URL",
            "🏢 save_competitor": "Store a competitor company",
            "📦 save_product": "Store a product model",
            "📊 save_specification": "Store a technical spec",
            "💰 save_price": "Store a price",
            "📝 save_review": "Store a customer review",
            "✅ mark_complete": "Signal mission complete",
        }
        
        for tool, desc in tools_info.items():
            st.markdown(f"- **{tool}**: {desc}")
    
        st.markdown("---")
        
        # Show agent decision flow
        st.markdown("### Agent Decision Flow")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="background: #fee2e2; padding: 1rem; border-radius: 8px; text-align: center;">
                <strong>1️⃣ OBSERVE</strong><br>
                <small>Check current state</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: #fef3c7; padding: 1rem; border-radius: 8px; text-align: center;">
                <strong>2️⃣ THINK</strong><br>
                <small>What's missing?</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: #d1fae5; padding: 1rem; border-radius: 8px; text-align: center;">
                <strong>3️⃣ ACT</strong><br>
                <small>Call a tool</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="background: #dbeafe; padding: 1rem; border-radius: 8px; text-align: center;">
                <strong>4️⃣ UPDATE</strong><br>
                <small>Save to state</small>
            </div>
            """, unsafe_allow_html=True)
    
    # === TAB 3: ONTOLOGY ===
    with tabs[2]:
        st.markdown("""
        <div class="section-header">
            <h2>📚 Specification Ontology</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        The **ontology** defines what specifications matter for pressure transmitters and how to normalize them 
        for head-to-head comparison. It's a hybrid human + AI approach:
        
        - **Human-defined**: The specification categories, units, and importance levels
        - **AI-extracted**: Values are extracted from datasheets and mapped to the ontology
        - **AI-derived**: New specs found by AI that aren't in the ontology are tagged separately
        """)
        
        st.markdown("---")
        
        # Display ontology categories
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🎯 High-Priority Specifications")
            st.markdown("*These are critical for competitive comparison (★★★★★)*")
            
            high_priority = [
                ("**Pressure Range**", "Operating pressure span", "psi (normalized from bar, kPa, MPa)"),
                ("**Accuracy**", "Measurement precision", "% of full scale"),
                ("**Output Signal**", "Communication protocol", "4-20mA, HART, Profibus, etc."),
                ("**Measurement Type**", "Gauge, Absolute, Differential", "Enum values"),
            ]
            
            for name, desc, unit in high_priority:
                st.markdown(f"- {name}: {desc} → *{unit}*")
            
            st.markdown("### 🔧 Physical Specifications")
            st.markdown("*Connection and material specs (★★★★)*")
            
            physical = [
                ("**Process Connection**", "1/4 NPT, 1/2 NPT, G1/2, Tri-Clamp, etc."),
                ("**Wetted Materials**", "316 SS, Hastelloy, Monel, Titanium"),
                ("**IP Rating**", "IP65, IP66, IP67, IP68, NEMA 4X"),
                ("**Hazardous Area**", "ATEX, IECEx, FM, Class I Div 1/2"),
            ]
            
            for name, values in physical:
                st.markdown(f"- {name}: *{values}*")
        
        with col2:
            st.markdown("### 🌡️ Environmental Specifications")
            st.markdown("*Temperature ranges and certifications (★★★★)*")
            
            environmental = [
                ("**Operating Temp**", "Ambient temperature range", "°C (normalized from °F)"),
                ("**Process Temp**", "Media temperature range", "°C"),
                ("**SIL Rating**", "Safety Integrity Level", "SIL1, SIL2, SIL3"),
            ]
            
            for name, desc, unit in environmental:
                st.markdown(f"- {name}: {desc} → *{unit}*")
            
            st.markdown("### ⚡ Electrical Specifications")
            st.markdown("*Power and signal specs (★★★)*")
            
            electrical = [
                ("**Supply Voltage**", "DC power input", "V DC"),
                ("**Response Time**", "Measurement update speed", "ms"),
                ("**Load Resistance**", "Maximum loop resistance", "ohm"),
            ]
            
            for name, desc, unit in electrical:
                st.markdown(f"- {name}: {desc} → *{unit}*")
        
        st.markdown("---")
        
        # Unit conversion explanation
        st.markdown("### 🔄 Automatic Unit Normalization")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Pressure → PSI**")
            st.code("""
1 bar = 14.5038 psi
1 kPa = 0.145 psi
1 MPa = 145.038 psi
1 mbar = 0.0145 psi
            """)
        
        with col2:
            st.markdown("**Temperature → Celsius**")
            st.code("""
°F → °C: (F - 32) × 5/9
K → °C: K - 273.15
            """)
        
        with col3:
            st.markdown("**Length → mm**")
            st.code("""
1 inch = 25.4 mm
1 ft = 304.8 mm
1 cm = 10 mm
            """)
        
        st.markdown("---")
        
        # Fuzzy matching explanation
        st.markdown("### 🔍 Fuzzy Matching & Aliases")
        
        st.markdown("""
        The ontology uses **fuzzy matching** (similarity > 0.6) to recognize specs even when named differently:
        """)
        
        fuzzy_examples = {
            "pressure_range": ["measuring range", "pressure span", "span", "range of measurement"],
            "accuracy": ["reference accuracy", "measurement error", "max error", "precision"],
            "output_signal": ["signal output", "communication protocol", "fieldbus", "analog output"],
            "wetted_materials": ["wetted parts", "media contact materials", "diaphragm material"],
        }
        
        for canonical, aliases in fuzzy_examples.items():
            st.markdown(f"- `{canonical}` ← *{', '.join(aliases)}*")
        
        st.markdown("---")
        
        # ACTUAL DATA TABLE - Show real extracted specs with normalization
        st.markdown("### 📊 Extracted Specifications (Actual Data)")
        st.markdown("""
        This table shows the **actual specifications** extracted from datasheets, including:
        - **Original Value**: What was extracted from the source
        - **Normalized Value**: What it was converted to (if unit conversion applied)
        - **Original Unit**: The unit found in the source
        - **Target Unit**: The canonical unit from the ontology
        """)
        
        try:
            # Fetch specs from Neo4j including normalized values
            driver = get_neo4j_driver()
            with driver.session() as session:
                result = session.run("""
                    MATCH (p:Product)-[:HAS_SPEC]->(s:Specification)
                    RETURN 
                        p.name as product,
                        s.spec_type as spec_type,
                        s.display_name as display_name,
                        s.value as original_value,
                        s.normalized_value as normalized_value,
                        s.unit as original_unit,
                        s.source_urls as sources
                    ORDER BY p.name, s.spec_type
                    LIMIT 100
                """)
                
                spec_rows = []
                for record in result:
                    spec_type = record['spec_type'] or ''
                    original_value = record['original_value'] or ''
                    normalized_value = record['normalized_value'] or ''
                    original_unit = record['original_unit'] or ''
                    display_name = record['display_name'] or spec_type.replace('_', ' ').title()
                    
                    # Determine if this is an ontology match
                    is_ontology = spec_type in PRESSURE_TRANSMITTER_ONTOLOGY
                    ontology_status = "✅ Ontology" if is_ontology else "🤖 AI-Derived"
                    
                    # Get canonical unit if in ontology
                    target_unit = ""
                    if is_ontology:
                        target_unit = PRESSURE_TRANSMITTER_ONTOLOGY[spec_type].canonical_unit or ""
                    
                    # Check if conversion happened
                    was_converted = (normalized_value and normalized_value != original_value and 
                                    normalized_value != '' and original_value != '')
                    conversion_indicator = "🔄" if was_converted else ""
                    
                    spec_rows.append({
                        'Product': record['product'],
                        'Spec Type': display_name,
                        'Original Value': original_value,
                        'Original Unit': original_unit,
                        'Normalized Value': normalized_value if was_converted else '-',
                        'Target Unit': target_unit,
                        'Converted': conversion_indicator,
                        'Status': ontology_status,
                    })
            
            driver.close()
            
            if spec_rows:
                spec_df = pd.DataFrame(spec_rows)
                
                # Filters
                col1, col2, col3 = st.columns(3)
                with col1:
                    product_filter = st.multiselect(
                        "Filter by Product",
                        options=sorted(spec_df['Product'].unique()),
                        default=[],
                        key="ontology_product_filter"
                    )
                with col2:
                    status_filter = st.multiselect(
                        "Filter by Status",
                        options=["✅ Ontology", "🤖 AI-Derived"],
                        default=[],
                        key="ontology_status_filter"
                    )
                with col3:
                    show_converted_only = st.checkbox(
                        "Show only converted specs 🔄",
                        value=False,
                        key="ontology_converted_filter"
                    )
                
                # Apply filters
                filtered_spec_df = spec_df.copy()
                if product_filter:
                    filtered_spec_df = filtered_spec_df[filtered_spec_df['Product'].isin(product_filter)]
                if status_filter:
                    filtered_spec_df = filtered_spec_df[filtered_spec_df['Status'].isin(status_filter)]
                if show_converted_only:
                    filtered_spec_df = filtered_spec_df[filtered_spec_df['Converted'] == '🔄']
                
                # Stats
                ontology_count = len(filtered_spec_df[filtered_spec_df['Status'] == '✅ Ontology'])
                ai_count = len(filtered_spec_df[filtered_spec_df['Status'] == '🤖 AI-Derived'])
                converted_count = len(filtered_spec_df[filtered_spec_df['Converted'] == '🔄'])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Specs", len(filtered_spec_df))
                with col2:
                    st.metric("Ontology Matches", ontology_count)
                with col3:
                    st.metric("AI-Derived", ai_count)
                with col4:
                    st.metric("Unit Conversions 🔄", converted_count)
                
                # Display table
                st.dataframe(
                    filtered_spec_df,
                    width="stretch",
                    height=450,
                    hide_index=True,
                    column_config={
                        "Product": st.column_config.TextColumn("Product", width="small"),
                        "Spec Type": st.column_config.TextColumn("Spec Type", width="small"),
                        "Original Value": st.column_config.TextColumn("Original Value", width="medium"),
                        "Original Unit": st.column_config.TextColumn("Orig Unit", width="small"),
                        "Normalized Value": st.column_config.TextColumn("→ Normalized", width="medium"),
                        "Target Unit": st.column_config.TextColumn("Target Unit", width="small"),
                        "Converted": st.column_config.TextColumn("🔄", width="small"),
                        "Status": st.column_config.TextColumn("Status", width="small"),
                    }
                )
                
                # Show spec type distribution
                st.markdown("#### Spec Type Distribution")
                spec_counts = filtered_spec_df['Spec Type'].value_counts().head(15)
                st.bar_chart(spec_counts)
                
            else:
                st.info("No specifications extracted yet. Run the pipeline first!")
                
        except Exception as e:
            st.warning(f"Could not load specifications: {str(e)}")
        
        st.markdown("---")
        
        # AI-derived attributes
        st.markdown("### 🤖 AI-Derived Attributes")
        
        st.markdown("""
        When the AI finds specifications **not in the ontology**, it:
        1. Saves them with a special `AI_DERIVED` tag
        2. Tracks how often each new spec appears
        3. Specs seen 3+ times become candidates for ontology expansion
        
        This allows the system to **learn new specification types** automatically!
        """)
        
        # Show AI-derived specs if any exist
        try:
            from src.ontology.specifications import get_ai_derived_attributes
            ai_derived = get_ai_derived_attributes()
            if ai_derived:
                st.markdown("**Recently discovered specs:**")
                for key, data in list(ai_derived.items())[:10]:
                    count = data.get('occurrence_count', 1)
                    st.markdown(f"- `{key}`: seen {count}x")
            else:
                st.info("No AI-derived attributes yet. Run the pipeline to discover new specs!")
        except:
            st.info("Run the pipeline to see AI-derived attributes.")
    
    # === TAB 4: SPECIFICATION TABLE ===
    with tabs[3]:
        st.markdown("""
        <div class="section-header">
            <h2>📋 Product Specification Database</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("View all products with their extracted specifications. Click column headers to sort.")
        
        try:
            products_df = fetch_all_products_with_specs()
            
            if not products_df.empty:
                # Column configuration
                column_order = ['Company', 'Product', 'Price', 'Review Count', 'Review Snippet', 'Sources']
                spec_columns = [col for col in products_df.columns if col not in column_order]
                column_order.extend(sorted(spec_columns))
                
                # Reorder columns
                display_df = products_df[[col for col in column_order if col in products_df.columns]].copy()
                
                # Convert list columns to strings to avoid PyArrow errors
                for col in display_df.columns:
                    if display_df[col].apply(lambda x: isinstance(x, list)).any():
                        display_df[col] = display_df[col].apply(
                            lambda x: ', '.join(str(i) for i in x) if isinstance(x, list) else str(x) if x else ''
                        )
                    # Also handle None/NaN values
                    display_df[col] = display_df[col].fillna('')
                
                # Show count
                st.markdown(f"**{len(display_df)} products** with specifications")
                
                # Filter options
                col1, col2 = st.columns([2, 1])
                with col1:
                    company_filter = st.multiselect(
                        "Filter by Company",
                        options=sorted(display_df['Company'].unique()),
                        default=[]
                    )
                
                with col2:
                    search = st.text_input("🔍 Search products", "")
                
                # Apply filters
                filtered_df = display_df.copy()
                if company_filter:
                    filtered_df = filtered_df[filtered_df['Company'].isin(company_filter)]
                if search:
                    mask = filtered_df.apply(lambda x: x.astype(str).str.contains(search, case=False).any(), axis=1)
                    filtered_df = filtered_df[mask]
                
                # Display table
                st.dataframe(
                    filtered_df,
                    width="stretch",
                    height=500,
                    hide_index=True,
                    column_config={
                        "Company": st.column_config.TextColumn("Company", width="medium"),
                        "Product": st.column_config.TextColumn("Product", width="medium"),
                        "Price": st.column_config.TextColumn("Price", width="small"),
                    }
                )
                
                # Export option
                if st.button("📥 Export to CSV"):
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="product_specifications.csv",
                        mime="text/csv"
                    )
            else:
                st.info("📋 No products found. Run the pipeline to extract product data.")
        
        except Exception as e:
            st.warning(f"⚠️ Could not load specifications: {str(e)}")
    
    # === TAB 5: PRODUCT COMPARISON ===
    with tabs[4]:
        st.markdown("""
        <div class="section-header">
            <h2>🔍 Head-to-Head Product Comparison</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("Select 2 or more products to compare their specifications side-by-side.")
        
        try:
            products_df = fetch_all_products_with_specs()
            
            if not products_df.empty:
                # Product selection
                all_products = products_df['Product'].unique().tolist()
                
                selected = st.multiselect(
                    "Select products to compare",
                    options=all_products,
                    default=all_products[:2] if len(all_products) >= 2 else all_products,
                    max_selections=5
                )
                
                if len(selected) >= 2:
                    # Create comparison view
                    comparison_df = products_df[products_df['Product'].isin(selected)].copy()
                    
                    # Convert ALL columns to strings to avoid PyArrow errors
                    for col in comparison_df.columns:
                        comparison_df[col] = comparison_df[col].apply(
                            lambda x: ', '.join(str(i) for i in x) if isinstance(x, list) 
                            else str(x) if pd.notna(x) and x != '' else ''
                        )
                    
                    # Transpose for side-by-side view
                    comparison_df = comparison_df.set_index('Product')
                    if 'Sources' in comparison_df.columns:
                        comparison_df = comparison_df.drop(columns=['Sources'])
                    comparison_df = comparison_df.T
                    
                    # Ensure all values are strings after transpose
                    comparison_df = comparison_df.astype(str).replace('nan', '').replace('None', '')
                    
                    st.markdown("### 📊 Comparison Matrix")
                    
                    # Display with highlighting
                    st.dataframe(
                        comparison_df,
                        width="stretch",
                        height=600,
                    )
                    
                    # Ontology-based analysis
                    st.markdown("### 🎯 Key Insights")
                    
                    insights = []
                    
                    # Check for important specs
                    for spec_name, spec_def in PRESSURE_TRANSMITTER_ONTOLOGY.items():
                        if spec_def.importance >= 4:  # High importance specs
                            display_name = spec_name.replace('_', ' ').title()
                            if display_name in comparison_df.index:
                                values = comparison_df.loc[display_name].dropna()
                                if len(values) > 0:
                                    unique_values = values.unique()
                                    if len(unique_values) > 1:
                                        insights.append(f"**{display_name}** varies: {', '.join(str(v) for v in unique_values)}")
                    
                    if insights:
                        for insight in insights:
                            st.markdown(f"- {insight}")
                    else:
                        st.info("No significant differences detected in high-priority specifications.")
                
                elif len(selected) == 1:
                    st.info("Select at least one more product to compare.")
                else:
                    st.info("Select products from the list above to start comparing.")
            
            else:
                st.info("🔍 No products available for comparison. Run the pipeline first.")
        
        except Exception as e:
            st.warning(f"⚠️ Could not load comparison data: {str(e)}")
    
    # === TAB 6: DATA VERIFICATION ===
    with tabs[5]:
        st.markdown("""
        <div class="section-header">
            <h2>✅ Human-in-the-Loop Verification</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("Review extracted data and approve or reject based on evidence quality.")
        
        try:
            all_relationships = fetch_all_relationships()
            
            pending_relationships = [
                r for r in all_relationships 
                if r['rel_id'] not in st.session_state.verified_relationships 
                and r['rel_id'] not in st.session_state.rejected_relationships
            ]
            
            # Stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", len(all_relationships))
            with col2:
                st.metric("Verified ✅", len(st.session_state.verified_relationships))
            with col3:
                st.metric("Rejected ❌", len(st.session_state.rejected_relationships))
            with col4:
                st.metric("Pending", len(pending_relationships))
            
            # Evidence coverage breakdown
            with_evidence = sum(1 for r in all_relationships if r['evidence_ids'])
            with_sources = sum(1 for r in all_relationships if r['source_urls'])
            
            st.markdown("#### 📊 Evidence Coverage")
            col1, col2, col3 = st.columns(3)
            with col1:
                pct_evidence = (with_evidence / len(all_relationships) * 100) if all_relationships else 0
                st.metric("With Evidence IDs", f"{with_evidence}/{len(all_relationships)}", f"{pct_evidence:.0f}%")
            with col2:
                pct_sources = (with_sources / len(all_relationships) * 100) if all_relationships else 0
                st.metric("With Source URLs", f"{with_sources}/{len(all_relationships)}", f"{pct_sources:.0f}%")
            with col3:
                # Count by relationship type
                by_type = {}
                for r in all_relationships:
                    t = r['relationship_type']
                    by_type[t] = by_type.get(t, 0) + 1
                st.write("**By Type:**")
                for t, count in sorted(by_type.items()):
                    st.caption(f"• {t}: {count}")
            
            st.markdown("---")
            
            if not pending_relationships:
                st.success("🎉 All relationships have been verified!")
                if st.button("🔄 Reset Verification"):
                    st.session_state.verified_relationships = set()
                    st.session_state.rejected_relationships = set()
                    st.session_state.selected_items = set()
                    st.rerun()
            else:
                # Batch actions
                col1, col2, col3, col4 = st.columns([2, 2, 1, 2])
                
                with col1:
                    if st.button(f"✅ Approve Selected ({len(st.session_state.selected_items)})", disabled=len(st.session_state.selected_items) == 0):
                        st.session_state.verified_relationships.update(st.session_state.selected_items)
                        st.session_state.selected_items = set()
                        st.rerun()
                
                with col2:
                    if st.button(f"❌ Reject Selected ({len(st.session_state.selected_items)})", disabled=len(st.session_state.selected_items) == 0):
                        delete_relationships(list(st.session_state.selected_items))
                        st.session_state.rejected_relationships.update(st.session_state.selected_items)
                        st.session_state.selected_items = set()
                        st.rerun()
                
                with col3:
                    if st.button("☑️ Select All"):
                        st.session_state.selected_items = {r['rel_id'] for r in pending_relationships}
                        st.rerun()
                
                with col4:
                    filter_type = st.selectbox(
                        "Filter",
                        ["All", "COMPETES_WITH", "OFFERS_PRODUCT", "HAS_SPEC", "ADDRESSES_NEED"],
                        label_visibility="collapsed"
                    )
                
                st.markdown("---")
                
                # Filter relationships
                if filter_type != "All":
                    filtered = [r for r in pending_relationships if r['relationship_type'] == filter_type]
                else:
                    filtered = pending_relationships
                
                # Display relationships
                for rel in filtered[:20]:  # Limit to 20 for performance
                    rel_id = rel['rel_id']
                    is_selected = rel_id in st.session_state.selected_items
                    
                    col1, col2, col3 = st.columns([0.5, 7, 2])
                    
                    with col1:
                        if st.checkbox("Select", value=is_selected, key=f"sel_{rel_id}", label_visibility="collapsed"):
                            st.session_state.selected_items.add(rel_id)
                        elif rel_id in st.session_state.selected_items:
                            st.session_state.selected_items.remove(rel_id)
                    
                    with col2:
                        st.markdown(f"**{rel['source_name']}** → {rel['relationship_type'].replace('_', ' ')} → **{rel['target_name']}**")
                    
                    with col3:
                        if rel['source_urls']:
                            domain = rel['source_urls'][0].split('/')[2] if len(rel['source_urls'][0].split('/')) > 2 else 'source'
                            st.caption(f"📎 {domain[:20]}")
                    
                    # Evidence expander - show exact text + ONE source link
                    with st.expander("📄 View Evidence"):
                        chunk = None
                        
                        # Strategy 1: Direct lookup by evidence_ids
                        if rel['evidence_ids']:
                            for eid in rel['evidence_ids'][:3]:
                                direct_chunk = get_chunk_by_id(eid)
                                if direct_chunk:
                                    chunk = direct_chunk
                                    break
                        
                        # Strategy 2: Fall back to semantic search
                        if not chunk:
                            chunk = find_best_evidence_for_relationship(
                                source=rel['source_name'],
                                relationship=rel['relationship_type'],
                                target=rel['target_name'],
                                evidence_ids=rel['evidence_ids']
                            )
                        
                        if chunk:
                            # Show the EXACT TEXT from the source
                            st.markdown("**📝 Evidence text:**")
                            evidence_text = chunk['document']
                            if len(evidence_text) > 500:
                                st.info(evidence_text[:500] + "...")
                            else:
                                st.info(evidence_text)
                            
                            # Show ONE source link
                            chunk_url = chunk.get('metadata', {}).get('source_url', '')
                            if _is_valid_http_url(chunk_url):
                                st.markdown(f"**🔗 Source:** [{chunk_url[:60]}...]({chunk_url})")
                        else:
                            # Fallback to stored URL if no ChromaDB evidence
                            if rel.get('source_urls') and rel['source_urls'][0]:
                                st.warning("No text evidence in ChromaDB")
                                st.markdown(f"**🔗 Source:** [{rel['source_urls'][0][:60]}...]({rel['source_urls'][0]})")
                            else:
                                st.error("⚠️ No evidence found. Run pipeline to regenerate.")
                
            st.markdown("---")
        
        except Exception as e:
            st.warning(f"⚠️ Could not load verification data: {str(e)}")
    
    # === TAB 7: CUSTOMER NEEDS ===
    with tabs[6]:
        st.markdown("""
        <div class="section-header">
            <h2>🎯 Customer Needs & Mapping</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("Comprehensive industry needs research from multiple sources.")
        
        # Load and display the industry report
        try:
            import os
            report_path = os.path.join(os.path.dirname(__file__), "industry_report.json")
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    report_data = json.load(f)
                
                # Report header
                st.markdown("### 📊 Industry Needs Report")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sources Analyzed", len(report_data.get("sources", [])))
                with col2:
                    st.metric("Industry", report_data.get("industry", "N/A"))
                with col3:
                    st.metric("Needs Extracted", report_data.get("needs_count", 0))
                with col4:
                    st.metric("Mappings Created", report_data.get("mappings_count", 0))
                
                # Full report in expandable section
                with st.expander("📄 View Full Report", expanded=True):
                    report_text = report_data.get("report", "No report available.")
                    st.markdown(report_text)
                
                # Sources
                with st.expander("🔗 Sources Used"):
                    for i, src in enumerate(report_data.get("sources", []), 1):
                        st.markdown(f"{i}. [{src[:60]}...]({src})")
                
                st.markdown("---")
            else:
                st.info("📝 No industry report generated yet. Run the pipeline to generate a comprehensive report.")
        except Exception as e:
            st.warning(f"Could not load report: {e}")
        
        st.markdown("### 🎯 Extracted Needs & Mappings")
        
        try:
            driver = get_neo4j_driver()
            with driver.session() as session:
                # Fetch customer needs with evidence IDs
                needs_result = session.run("""
                    MATCH (n:CustomerNeed)
                    OPTIONAL MATCH (p:Product)-[r:ADDRESSES_NEED]->(n)
                    RETURN 
                        n.name as need,
                        n.description as description,
                        n.industry as industry,
                        n.source_urls as sources,
                        n.evidence_ids as evidence_ids,
                        collect(DISTINCT {product: p.name, spec: r.via_spec, explanation: r.explanation}) as mappings
                    ORDER BY n.name
                """)
                
                needs = list(needs_result)
                
                if needs:
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{len(needs)}</div>
                            <div class="metric-label">Customer Needs</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        total_mappings = sum(len([m for m in n['mappings'] if m.get('product')]) for n in needs)
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{total_mappings}</div>
                            <div class="metric-label">Need-Product Mappings</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        industries = set(n['industry'] for n in needs if n['industry'])
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{len(industries) if industries else 1}</div>
                            <div class="metric-label">Industries</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Display each need
                    for need in needs:
                        with st.expander(f"🎯 {need['need']}", expanded=False):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(f"**Description:** {need['description'] or 'No description'}")
                                
                                # Show mappings
                                mappings = [m for m in need['mappings'] if m.get('product')]
                                if mappings:
                                    st.markdown("**Products addressing this need:**")
                                    for m in mappings:
                                        st.markdown(f"- **{m['product']}** via `{m['spec']}`: {m['explanation'] or ''}")
                                else:
                                    st.info("No products mapped to this need yet.")
                            
                            with col2:
                                st.markdown(f"**Industry:** {need['industry'] or 'General'}")
                                
                                # Show WHERE the customer need requirement came from
                                # (the industry research pages, NOT product pages)
                                need_name = need.get('need', '')
                                report_sources = need.get('sources') or []  # URLs from industry research
                                
                                # Search for evidence ONLY within the report source URLs
                                chunk = None
                                if report_sources:
                                    # Semantic search with need name as query
                                    chunk = find_best_evidence_for_relationship(
                                        source="industry requirement",
                                        relationship="states",
                                        target=need_name,
                                        evidence_ids=need.get('evidence_ids') or []
                                    )
                                    
                                    # Verify the chunk is from a report source, not a product page
                                    if chunk:
                                        chunk_url = chunk.get('metadata', {}).get('source_url', '')
                                        # Check if this URL is from our report sources
                                        is_from_report = any(
                                            chunk_url and src and (chunk_url in src or src in chunk_url)
                                            for src in report_sources
                                        )
                                        if not is_from_report:
                                            chunk = None  # Reject - it's from a product page
                                
                                if chunk:
                                    st.markdown("**📄 Source Evidence (Industry Research):**")
                                    evidence_text = chunk.get('document', '')
                                    if len(evidence_text) > 400:
                                        st.info(evidence_text[:400] + "...")
                                    else:
                                        st.info(evidence_text)
                                    src_url = chunk.get('metadata', {}).get('source_url', '')
                                    if src_url and _is_valid_http_url(src_url):
                                        st.markdown(f"[View Source]({src_url})")
                                elif report_sources:
                                    # Fall back to just showing the report source URLs
                                    st.markdown("**📄 Sources (Industry Research):**")
                                    for src in report_sources[:2]:
                                        if src and _is_valid_http_url(src):
                                            st.markdown(f"- [{src[:50]}...]({src})")
                    
                    # Summary table
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 📊 Needs Summary Table")
                    
                    table_data = []
                    for need in needs:
                        mappings = [m for m in need['mappings'] if m.get('product')]
                        table_data.append({
                            'Need': need['need'],
                            'Industry': need['industry'] or 'General',
                            'Description': (need['description'] or '')[:100],
                            'Products Mapped': len(mappings),
                            'Products': ', '.join([m['product'] for m in mappings][:3]) or '-'
                        })
                    
                    df = pd.DataFrame(table_data)
                    st.dataframe(df, use_container_width=True)
                    
                else:
                    st.info("No customer needs found. Run the pipeline with customer needs research enabled.")
            
            driver.close()
            
        except Exception as e:
            st.warning(f"⚠️ Could not load customer needs: {str(e)}")


if __name__ == "__main__":
    main()
