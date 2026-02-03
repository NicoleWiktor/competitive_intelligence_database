#!/usr/bin/env python3
"""
Competitive Intelligence Database

USAGE:
    python main.py                              # 5 competitors, 25 iterations (default)
    python main.py --competitors 3              # 3 competitors
    python main.py --iterations 15              # limit to 15 agent iterations
    python main.py --industry "oil and gas"    # specific industry
    python main.py --streamlit                  # Launch dashboard
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Honeywell Competitive Intelligence")
    
    parser.add_argument("--streamlit", action="store_true", help="Launch dashboard")
    parser.add_argument("--competitors", type=int, default=5, help="Number of competitors (max 10)")
    parser.add_argument("--industry", type=str, default="process industries", help="Target industry (e.g., 'oil and gas', 'chemical processing')")
    parser.add_argument("--iterations", type=int, default=25, help="Max agent iterations (default 25)")
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
        industry=args.industry,
        max_iterations=args.iterations,
        incremental=args.incremental,
    )
    
    # Summary
    print("\n📊 SUMMARY:")
    print(f"   Competitors: {len(result.get('competitors', {}))}")
    print(f"   Products: {len(result.get('products', {}))}")
    print(f"   Specs: {sum(len(s) for s in result.get('specifications', {}).values())}")
    has_report = "✅" if result.get('industry_needs_report') else "❌"
    print(f"   Industry Report: {has_report} ({len(result.get('report_sources', []))} sources)")
    print(f"   Customer Needs: {len(result.get('customer_needs', {}))}")
    print(f"   Need Mappings: {len(result.get('need_mappings', []))}")
    print("\n🚀 Run 'python main.py --streamlit' to view dashboard")


if __name__ == "__main__":
    main()
