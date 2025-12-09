"""
Graph Visualization Page - Interactive knowledge graph viewer
"""

import streamlit as st
from neo4j import GraphDatabase
import sys
from pathlib import Path
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config.settings import get_neo4j_config

st.set_page_config(page_title="Graph Visualization", page_icon="📊", layout="wide")


def get_neo4j_driver():
    """Create Neo4j connection."""
    cfg = get_neo4j_config()
    return GraphDatabase.driver(cfg['uri'], auth=(cfg['user'], cfg['password']))


def fetch_graph_data():
    """Fetch all nodes and relationships for visualization."""
    driver = get_neo4j_driver()
    with driver.session() as session:
        # Get all relationships with node info
        result = session.run("""
            MATCH (source)-[rel]->(target)
            RETURN 
                id(source) as source_id,
                labels(source)[0] as source_label,
                source.name as source_name,
                type(rel) as relationship_type,
                id(target) as target_id,
                labels(target)[0] as target_label,
                target.name as target_name
        """)
        
        nodes = {}
        edges = []
        
        for record in result:
            # Add source node
            source_id = record['source_id']
            if source_id not in nodes:
                nodes[source_id] = {
                    'id': source_id,
                    'label': record['source_name'],
                    'group': record['source_label'],
                    'title': f"{record['source_label']}: {record['source_name']}"
                }
            
            # Add target node
            target_id = record['target_id']
            if target_id not in nodes:
                nodes[target_id] = {
                    'id': target_id,
                    'label': record['target_name'],
                    'group': record['target_label'],
                    'title': f"{record['target_label']}: {record['target_name']}"
                }
            
            # Add edge
            edges.append({
                'from': source_id,
                'to': target_id,
                'label': record['relationship_type'],
                'title': record['relationship_type']
            })
    
    driver.close()
    return list(nodes.values()), edges


def create_graph_viz(nodes, edges):
    """Create interactive network visualization using PyVis."""
    net = Network(
        height="700px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True
    )
    
    # Configure physics
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -30000,
                "centralGravity": 0.3,
                "springLength": 200,
                "springConstant": 0.04
            },
            "stabilization": {
                "enabled": true,
                "iterations": 100
            }
        },
        "nodes": {
            "font": {"size": 16, "face": "arial"},
            "borderWidth": 2,
            "size": 25
        },
        "edges": {
            "font": {"size": 12, "align": "middle"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
            "smooth": {"type": "continuous"}
        }
    }
    """)
    
    # Define colors for different node types
    colors = {
        'Company': '#e74c3c',      # Red
        'Product': '#3498db',       # Blue
        'Price': '#2ecc71',         # Green
        'Specification': '#f39c12' # Orange
    }
    
    # Add nodes
    for node in nodes:
        net.add_node(
            node['id'],
            label=node['label'],
            title=node['title'],
            color=colors.get(node['group'], '#95a5a6'),
            group=node['group']
        )
    
    # Add edges
    for edge in edges:
        net.add_edge(
            edge['from'],
            edge['to'],
            label=edge['label'],
            title=edge['title']
        )
    
    return net


def main():
    # Professional header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0;">📊 Knowledge Graph Visualization</h1>
        <p style="color: #f0f0f0; margin: 0.5rem 0 0 0; font-size: 1.1em;">
            Honeywell Competitive Intelligence Network
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Overview
    with st.expander("ℹ️ How to Read This Graph", expanded=False):
        st.markdown("""
        ### Understanding the Network
        This interactive graph shows the competitive landscape for Honeywell pressure transmitters:
        
        - **Honeywell** (center) connects to all competitors
        - **Competitors** (red) link to their specific products
        - **Products** (blue) connect to their prices and specifications
        - **Prices** (green) and **Specifications** (orange) provide detailed product data
        
        ### Navigation
        - **Drag** any node to rearrange the layout
        - **Zoom** with your mouse wheel or trackpad
        - **Hover** over nodes/edges to see source information
        - **Click** nodes to highlight their connections
        """)
    
    st.markdown("---")
    
    # Sidebar - Legend
    with st.sidebar:
        st.header("🎨 Graph Legend")
        
        st.markdown("### Entity Types")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("🔴")
            st.markdown("🔵")
            st.markdown("🟢")
            st.markdown("🟠")
        with col2:
            st.markdown("**Company**")
            st.markdown("**Product**")
            st.markdown("**Price**")
            st.markdown("**Specification**")
        
        st.markdown("---")
        st.markdown("### Relationship Types")
        st.markdown("""
        **COMPETES_WITH**  
        Links Honeywell to competitor companies
        
        **OFFERS_PRODUCT**  
        Links companies to their product models
        
        **HAS_PRICE**  
        Links products to market pricing
        
        **HAS_SPECIFICATION**  
        Links products to technical specs
        """)
        
        st.markdown("---")
        st.markdown("### 🖱️ Controls")
        st.markdown("""
        **Drag** - Rearrange nodes  
        **Scroll** - Zoom in/out  
        **Hover** - View details  
        **Click** - Highlight connections
        """)
    
    try:
        # Fetch graph data
        with st.spinner("Loading graph data..."):
            nodes, edges = fetch_graph_data()
        
        if not nodes:
            st.warning("No data found in the knowledge graph. Run the pipeline first!")
            return
        
        # Display stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Nodes", len(nodes))
        with col2:
            st.metric("Total Relationships", len(edges))
        with col3:
            node_types = {}
            for node in nodes:
                node_types[node['group']] = node_types.get(node['group'], 0) + 1
            st.metric("Node Types", len(node_types))
        
        st.markdown("---")
        
        # Create and display graph
        with st.spinner("Generating visualization..."):
            net = create_graph_viz(nodes, edges)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
                net.save_graph(f.name)
                with open(f.name, 'r') as html_file:
                    html_content = html_file.read()
            
            # Display in Streamlit
            components.html(html_content, height=750, scrolling=True)
        
        st.markdown("---")
        
        # Node list
        with st.expander("📋 View Node List"):
            st.dataframe([
                {
                    'ID': n['id'],
                    'Type': n['group'],
                    'Name': n['label']
                } for n in nodes
            ], use_container_width=True)
        
        # Relationship list
        with st.expander("🔗 View Relationship List"):
            st.dataframe([
                {
                    'From': nodes[[n['id'] for n in nodes].index(e['from'])]['label'],
                    'Relationship': e['label'],
                    'To': nodes[[n['id'] for n in nodes].index(e['to'])]['label']
                } for e in edges
            ], use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading graph: {e}")
        st.info("Make sure Neo4j is running and the pipeline has generated data.")


if __name__ == "__main__":
    main()

