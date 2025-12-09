"""
ChromaDB Store - Manages source evidence storage for knowledge graph traceability.

After extracting content from web pages, this module:
1. Chunks the raw HTML/text into semantic segments
2. Stores chunks in ChromaDB with metadata (URL, timestamp, etc.)
3. Returns chunk IDs for linking to Neo4j relationships

Chunking Method: RecursiveCharacterTextSplitter
- Tries to split on paragraphs, then sentences, then words
- Preserves semantic context better than fixed-size splits
- Chunk size: 800 chars (balance between context and granularity)
- Overlap: 100 chars (maintains context across boundaries)
"""

from __future__ import annotations

import chromadb
from datetime import datetime
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Initialize ChromaDB client (persistent storage in ./chroma_db directory)
_client = None


def get_chroma_client() -> chromadb.Client:
    """Get or create ChromaDB client with persistent storage."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path="./chroma_db")
    return _client


def get_collection(name: str = "competitive_intelligence_sources"):
    """Get or create the ChromaDB collection for source evidence."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"description": "Raw source evidence for competitive intelligence knowledge graph"}
    )


def chunk_and_store(
    raw_content: str,
    source_url: str,
    query: str = "",
    page_title: str = ""
) -> List[str]:
    """
    Chunk raw content and store in ChromaDB.
    
    Chunking Strategy:
    - RecursiveCharacterTextSplitter: Splits on ["\n\n", "\n", " ", ""]
    - Chunk size: 800 characters (balance context vs granularity)
    - Overlap: 100 characters (preserve context across boundaries)
    
    Args:
        raw_content: Full HTML/text from web page
        source_url: URL of the source page
        query: Search query that led to this source
        page_title: Title of the page
    
    Returns:
        List of chunk IDs (evidence_ids) stored in ChromaDB
    """
    if not raw_content:
        return []
    
    # Initialize text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,          # ~200 tokens per chunk
        chunk_overlap=100,       # Preserve context
        separators=["\n\n", "\n", ". ", " ", ""],  # Try paragraph → sentence → word
        length_function=len,
    )
    
    # Split into chunks
    chunks = splitter.split_text(raw_content)
    
    if not chunks:
        return []
    
    # Prepare data for ChromaDB
    timestamp = datetime.utcnow().isoformat()
    collection = get_collection()
    
    chunk_ids = []
    documents = []
    metadatas = []
    
    for i, chunk_text in enumerate(chunks):
        # Create unique ID: URL + chunk index + timestamp
        chunk_id = f"{source_url}__chunk_{i}__{timestamp}"
        chunk_ids.append(chunk_id)
        documents.append(chunk_text)
        metadatas.append({
            "source_url": source_url,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "query": query,
            "page_title": page_title,
            "timestamp": timestamp,
            "chunk_size": len(chunk_text)
        })
    
    # Store in ChromaDB
    collection.add(
        ids=chunk_ids,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"[chroma] Stored {len(chunks)} chunks from {source_url}")
    
    return chunk_ids


def query_evidence(query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """
    Query ChromaDB for relevant evidence chunks.
    
    Useful for verification or finding supporting evidence.
    
    Args:
        query_text: Text to search for
        n_results: Number of results to return
    
    Returns:
        List of dicts with: id, document, metadata, distance
    """
    collection = get_collection()
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    # Format results
    formatted = []
    for i in range(len(results['ids'][0])):
        formatted.append({
            "id": results['ids'][0][i],
            "document": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "distance": results['distances'][0][i] if 'distances' in results else None
        })
    
    return formatted


def get_chunk_by_id(chunk_id: str) -> Dict[str, Any] | None:
    """
    Retrieve a specific chunk by its ID.
    
    Useful for verifying evidence linked to Neo4j relationships.
    """
    collection = get_collection()
    try:
        result = collection.get(ids=[chunk_id])
        if result['ids']:
            return {
                "id": result['ids'][0],
                "document": result['documents'][0],
                "metadata": result['metadatas'][0]
            }
    except Exception as e:
        print(f"[chroma] Error retrieving chunk {chunk_id}: {e}")
    
    return None


def find_best_evidence_for_relationship(
    source: str, 
    relationship: str, 
    target: str,
    evidence_ids: List[str] = None
) -> Dict[str, Any] | None:
    """
    **Semantic search-based evidence retrieval** - Find the best supporting evidence.
    
    This automatically verifies data quality using ChromaDB's semantic search:
    - Searches ChromaDB semantically for the relationship claim
    - Returns the chunk that best supports the relationship
    - Includes relevance score (distance) for quality assessment
    
    Args:
        source: Source entity (e.g., "Wika")
        relationship: Relationship type (e.g., "OFFERS_PRODUCT")
        target: Target entity (e.g., "A-10")
        evidence_ids: Optional - search only within these chunks
    
    Returns:
        Best matching chunk with: document, metadata, distance, id
        None if no good match found
    """
    # Build semantic query from the relationship
    query = f"{source} {relationship.replace('_', ' ').lower()} {target}"
    
    collection = get_collection()
    
    # Strategy 1: If evidence_ids provided, search within those first
    if evidence_ids:
        try:
            # Query with semantic search, but only return results from evidence_ids
            all_results = collection.query(
                query_texts=[query],
                n_results=20,  # Get more results to filter
                include=['documents', 'metadatas', 'distances']
            )
            
            if all_results and all_results['ids']:
                # Filter to only chunks in evidence_ids
                for i, chunk_id in enumerate(all_results['ids'][0]):
                    if chunk_id in evidence_ids:
                        # Found a match in our evidence set
                        return {
                            "id": chunk_id,
                            "document": all_results['documents'][0][i],
                            "metadata": all_results['metadatas'][0][i],
                            "distance": all_results['distances'][0][i]
                        }
        except Exception as e:
            print(f"[chroma] Semantic search within evidence_ids failed: {e}")
    
    # Strategy 2: Search entire collection for best match
    try:
        results = collection.query(
            query_texts=[query],
            n_results=1,  # Just get the best match
            include=['documents', 'metadatas', 'distances']
        )
        
        if results and results['ids'] and results['ids'][0]:
            return {
                "id": results['ids'][0][0],
                "document": results['documents'][0][0],
                "metadata": results['metadatas'][0][0],
                "distance": results['distances'][0][0]
            }
    except Exception as e:
        print(f"[chroma] Semantic search failed: {e}")
    
    return None

