#!/bin/bash

# Launch Streamlit Human-in-the-Loop Verification App

echo "🚀 Starting Competitive Intelligence Verifier..."
echo ""
echo "Make sure you have:"
echo "  ✓ Run the pipeline (python src/pipeline/graph_builder.py)"
echo "  ✓ Neo4j is running"
echo "  ✓ Installed dependencies (pip install -r requirements.txt)"
echo ""
echo "Opening app at http://localhost:8501"
echo ""

streamlit run streamlit_app.py

