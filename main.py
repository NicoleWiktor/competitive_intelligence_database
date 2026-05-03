#!/usr/bin/env python3
"""
Competitive Intelligence Database

USAGE:
    python main.py                                          # defaults
    python main.py --company "Tesla" --product "Model Y" --product-category "electric vehicles" --industry "automotive"
    python main.py --competitors 3
    python main.py --iterations 15
    python main.py --industry "oil and gas"
    python main.py --multi-agent
    python main.py --streamlit
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Competitive Intelligence Pipeline")
    
    parser.add_argument("--streamlit", action="store_true", help="Launch dashboard")
    parser.add_argument("--company", type=str, default="Honeywell", help="Target company name")
    parser.add_argument("--product", type=str, default="SmartLine ST700", help="Target product name")
    parser.add_argument("--product-category", type=str, default="pressure transmitters", help="Product category (e.g., 'electric vehicles')")
    parser.add_argument("--competitors", type=int, default=5, help="Number of competitors (max 10)")
    parser.add_argument("--industry", type=str, default="process industries", help="Target industry")
    parser.add_argument("--iterations", type=int, default=25, help="Max agent iterations")
    parser.add_argument("--allowed-domains", nargs="+", default=None)
    parser.add_argument("--allowed-source-types", nargs="+", default=None)
    parser.add_argument("--incremental", action="store_true", help="Keep existing data")
    parser.add_argument("--multi-agent", action="store_true", help="Use supervisor-worker architecture")
    
    args = parser.parse_args()
    
    if args.streamlit:
        print("🚀 Starting Streamlit dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
        return
    
    from src.pipeline.graph_builder import run_pipeline
    
    result = run_pipeline(
        company=args.company,
        target_product=args.product,
        product_category=args.product_category,
        max_competitors=min(args.competitors, 10),
        industry=args.industry,
        max_iterations=args.iterations,
        allowed_domains=args.allowed_domains,
        allowed_source_types=args.allowed_source_types,
        incremental=args.incremental,
        multi_agent=args.multi_agent,
    )
    
    print("\n📊 SUMMARY:")
    print(f"   Company: {args.company} | Product: {args.product}")
    print(f"   Competitors: {len(result.get('competitors', {}))}")
    print(f"   Products: {len(result.get('products', {}))}")
    print(f"   Specs: {sum(len(s) for s in result.get('specifications', {}).values())}")
    print(f"   Customer Segments: {len(result.get('customer_segments', []))} ({len(result.get('segment_mappings', []))} mappings)")
    has_report = "✅" if result.get('industry_needs_report') else "❌"
    print(f"   Industry Report: {has_report} ({len(result.get('report_sources', []))} sources)")
    print(f"   Customer Needs: {len(result.get('customer_needs', {}))}")
    print(f"   Need Mappings: {len(result.get('need_mappings', []))}")
    has_hoq = "✅" if result.get('house_of_quality') else "❌"
    print(f"   House of Quality: {has_hoq}")
    print("\n🚀 Run 'python main.py --streamlit' to view dashboard")


if __name__ == "__main__":
    main()