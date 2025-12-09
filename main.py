#!/usr/bin/env python3
"""
Competitive Intelligence Database - Main Entry Point

This is the primary way to interact with the competitive intelligence system.

USAGE:
    # Run the agentic pipeline (RECOMMENDED)
    python main.py --mode agentic
    
    # Run the traditional pipeline
    python main.py --mode pipeline
    
    # Start the Streamlit dashboard
    python main.py --streamlit
    
    # Quick test with fewer iterations
    python main.py --mode agentic --iterations 10 --competitors 3
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_pipeline(args):
    """Run the data extraction pipeline."""
    from src.pipeline.graph_builder import run_agentic_mode, run_pipeline_mode
    
    if args.mode == "agentic":
        print("🤖 Starting AGENTIC mode...")
        result = run_agentic_mode(
            target_product=args.product,
            max_competitors=args.competitors,
            max_iterations=args.iterations,
        )
    else:
        print("⚙️ Starting PIPELINE mode...")
        result = run_pipeline_mode(
            max_competitors=args.competitors,
            max_iterations=args.iterations,
        )
    
    return result


def run_streamlit():
    """Launch the Streamlit dashboard."""
    print("🚀 Starting Streamlit dashboard...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.headless", "true"
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Honeywell Competitive Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run agentic pipeline (recommended)
  python main.py --mode agentic
  
  # Run with custom settings
  python main.py --mode agentic --competitors 3 --iterations 20
  
  # Launch Streamlit dashboard
  python main.py --streamlit
  
  # Run traditional pipeline
  python main.py --mode pipeline
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["agentic", "pipeline"],
        default="agentic",
        help="Pipeline mode: 'agentic' (AI decides) or 'pipeline' (fixed phases)"
    )
    
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Launch Streamlit dashboard instead of running pipeline"
    )
    
    parser.add_argument(
        "--competitors",
        type=int,
        default=5,
        help="Number of competitors to find (default: 5)"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Maximum iterations (default: 30)"
    )
    
    parser.add_argument(
        "--product",
        type=str,
        default="SmartLine ST700",
        help="Target Honeywell product (default: SmartLine ST700)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("🎯 HONEYWELL COMPETITIVE INTELLIGENCE SYSTEM")
    print("="*60)
    
    if args.streamlit:
        run_streamlit()
    else:
        result = run_pipeline(args)
        
        # Print summary
        relationships = result.get("relationships", [])
        print("\n" + "="*60)
        print("📊 EXTRACTION SUMMARY")
        print("="*60)
        
        competes = len([r for r in relationships if r.get("relationship") == "COMPETES_WITH"])
        products = len([r for r in relationships if r.get("relationship") == "OFFERS_PRODUCT"])
        prices = len([r for r in relationships if r.get("relationship") == "HAS_PRICE"])
        specs = len([r for r in relationships if r.get("relationship") == "HAS_SPEC"])
        reviews = len([r for r in relationships if r.get("relationship") == "HAS_REVIEW"])
        
        print(f"  Competitors found: {competes}")
        print(f"  Products found: {products}")
        print(f"  Prices found: {prices}")
        print(f"  Specifications: {specs}")
        print(f"  Reviews: {reviews}")
        print("="*60)
        print("\n✅ Data has been written to Neo4j")
        print("🚀 Run 'python main.py --streamlit' to view the dashboard")


if __name__ == "__main__":
    main()

