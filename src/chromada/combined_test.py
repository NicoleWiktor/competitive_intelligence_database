import uuid
from neo4j import GraphDatabase
import chromadb
import os
from dotenv import load_dotenv

from tavily import TavilyClient

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")

# --- 1. Tavily search ---
QUERY = "Honeywell ST700 pressure transmitter competitors"
tavily = TavilyClient(api_key=api_key)
raw_results = tavily.search(query=QUERY).get("results", [])

# --- 2. Store in ChromaDB (bridge via doc_id) ---
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="tavily_results")

documents = []
metadatas = []
ids = []

for r in raw_results:
    doc_id = str(uuid.uuid4())
    content = r.get("content", "")
    documents.append(content)
    metadatas.append({
        "doc_id": doc_id,
        "url": r.get("url", ""),
        "title": r.get("title", ""),
        "snippet": r.get("snippet", ""),
        "source": r.get("source", ""),
        "query_used": QUERY
    })
    ids.append(doc_id)

if ids:
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"[chroma] Stored {len(ids)} Tavily documents")

# --- 3. Write to Neo4j ---
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "Test1234")
DATABASE = "chromada-neo4j-testdb2"

# def delete_document(doc_id: str):
#     driver = GraphDatabase.driver(URI, auth=AUTH)
#     try:
#         with driver.session(database=DATABASE) as s:
#             s.run("MATCH (d:Document {doc_id:$id}) DETACH DELETE d", id=doc_id)
#     finally:
#         driver.close()

def write_results_to_neo4j(docs, query_text):
    driver = GraphDatabase.driver(URI, auth=AUTH)
    try:
        with driver.session(database=DATABASE) as session:
            # Constraint
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE")

            # SearchQuery node
            session.run(
                "MERGE (q:SearchQuery {text:$text}) "
                "ON CREATE SET q.created = timestamp(), q.name = $text "
                "ON MATCH SET q.name = $text",
                text=query_text
            )

            for md in docs:
                session.run(
                    """
                    MERGE (d:Document {doc_id:$doc_id})
                    ON CREATE SET d.url=$url, d.title=$title, d.snippet=$snippet, d.source=$source, d.query_used=$query_used, d.created=timestamp()
                    WITH d
                    MATCH (q:SearchQuery {text:$search_query})
                    MERGE (q)-[:RETURNS]->(d)
                    """,
                    doc_id=md["doc_id"],
                    url=md["url"],
                    title=md["title"],
                    snippet=md["snippet"],
                    source=md["source"],
                    query_used=md["query_used"],
                    search_query=query_text,
                )
            print(f"[neo4j] Wrote {len(docs)} documents for query '{query_text}'")
    finally:
        driver.close()

write_results_to_neo4j(metadatas, QUERY)

# --- 4. Optional: simple retrieval demo ---
# Get top-2 similar docs via vector search then list their Neo4j nodes
search_results = collection.query(query_texts=["pressure transmitter competitors"], n_results=2)
top_ids = search_results["ids"][0]
print(f"[chroma] Top doc_ids: {top_ids}")


# --- 5. Summary ---
print("[done] Complete integration Tavily → ChromaDB → Neo4j")
