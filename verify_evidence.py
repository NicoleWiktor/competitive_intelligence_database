"""
Evidence Verification Script

Query ChromaDB to verify source evidence for Neo4j relationships.

Usage:
    python verify_evidence.py <evidence_id>
    python verify_evidence.py --search "Wika A-10 price"
    python verify_evidence.py --stats
"""

import sys
from src.pipeline.chroma_store import get_chunk_by_id, query_evidence, get_collection


def show_chunk(evidence_id: str):
    """Display a specific evidence chunk by ID."""
    chunk = get_chunk_by_id(evidence_id)
    
    if not chunk:
        print(f"❌ Evidence ID '{evidence_id}' not found")
        return
    
    print("\n" + "="*80)
    print(f"EVIDENCE CHUNK: {chunk['id']}")
    print("="*80)
    
    metadata = chunk['metadata']
    print(f"\n📄 Source URL: {metadata.get('source_url', 'N/A')}")
    print(f"🔍 Search Query: {metadata.get('query', 'N/A')}")
    print(f"📰 Page Title: {metadata.get('page_title', 'N/A')}")
    print(f"📊 Chunk {metadata.get('chunk_index', 0) + 1} of {metadata.get('total_chunks', 'N/A')}")
    print(f"⏰ Timestamp: {metadata.get('timestamp', 'N/A')}")
    print(f"📏 Size: {metadata.get('chunk_size', 'N/A')} chars")
    
    print("\n" + "-"*80)
    print("CONTENT:")
    print("-"*80)
    print(chunk['document'])
    print("\n" + "="*80 + "\n")


def search_evidence(query: str, n_results: int = 5):
    """Search ChromaDB for relevant evidence."""
    results = query_evidence(query, n_results)
    
    if not results:
        print(f"❌ No evidence found for query: '{query}'")
        return
    
    print("\n" + "="*80)
    print(f"SEARCH RESULTS: '{query}'")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        distance = result.get('distance', 0)
        
        print(f"\n[{i}] Relevance: {1 - distance:.3f}")
        print(f"    Source: {metadata.get('source_url', 'N/A')}")
        print(f"    Query: {metadata.get('query', 'N/A')}")
        print(f"    Chunk: {metadata.get('chunk_index', 0) + 1} of {metadata.get('total_chunks', 'N/A')}")
        print(f"    ID: {result['id']}")
        print(f"    Preview: {result['document'][:200]}...")
    
    print("\n" + "="*80 + "\n")


def show_stats():
    """Show ChromaDB collection statistics."""
    collection = get_collection()
    count = collection.count()
    
    print("\n" + "="*80)
    print("CHROMADB STATISTICS")
    print("="*80)
    print(f"\n📦 Collection: {collection.name}")
    print(f"📊 Total Chunks: {count}")
    
    if count > 0:
        # Sample a few chunks to show variety
        sample = collection.peek(limit=5)
        print(f"\n📋 Sample Evidence IDs:")
        for i, chunk_id in enumerate(sample['ids'], 1):
            metadata = sample['metadatas'][i-1]
            print(f"    [{i}] {chunk_id}")
            print(f"        Source: {metadata.get('source_url', 'N/A')}")
    
    print("\n" + "="*80 + "\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    arg = sys.argv[1]
    
    if arg == "--stats":
        show_stats()
    elif arg == "--search":
        if len(sys.argv) < 3:
            print("Usage: python verify_evidence.py --search <query>")
            return
        query = " ".join(sys.argv[2:])
        search_evidence(query)
    else:
        # Assume it's an evidence ID
        show_chunk(arg)


if __name__ == "__main__":
    main()

