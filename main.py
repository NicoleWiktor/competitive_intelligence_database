#!/usr/bin/env python3
"""
Competitive Intelligence Database

USAGE:
    python main.py                    # 5 competitors (default)
    python main.py --competitors 3    # 3 competitors
    python main.py --streamlit        # Launch dashboard
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Honeywell Competitive Intelligence")
    
    parser.add_argument("--streamlit", action="store_true", help="Launch dashboard")
    parser.add_argument("--competitors", type=int, default=5, help="Number of competitors (max 10)")
    parser.add_argument("--incremental", action="store_true", help="Keep existing data")
    
    args = parser.parse_args()
    
    if args.streamlit:
        print("🚀 Starting Streamlit dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
        return
    
    # Run pipeline
    from src.pipeline.graph_builder import run_pipeline
    
    result = run_pipeline(
        max_competitors=min(args.competitors, 10),
        incremental=args.incremental,
    )
    
    # Summary
    print("\n📊 SUMMARY:")
    print(f"   Competitors: {len(result.get('competitors', {}))}")
    print(f"   Products: {len(result.get('products', {}))}")
    print(f"   Specs: {sum(len(s) for s in result.get('specifications', {}).values())}")
    print("\n🚀 Run 'python main.py --streamlit' to view dashboard")


if __name__ == "__main__":
    main()
