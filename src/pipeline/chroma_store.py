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

