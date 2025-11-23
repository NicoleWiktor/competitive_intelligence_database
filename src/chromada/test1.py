import os
import chromadb
from tavily import TavilyClient
from dotenv import load_dotenv
import uuid

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")

# --- 1. Initialize Tavily ---
tavily = TavilyClient(api_key=api_key)

# Example Tavily Query
query = "Honeywell ST700 pressure transmitter competitors"
tavily_results = tavily.search(query=query).get("results")[0:3]

# --- 2. Initialize ChromaDB ---
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="tavily_results")

# --- 3. Store Tavily results into ChromaDB ---
documents = []
metadatas = []
ids = []

for result in tavily_results:
    doc_id = str(uuid.uuid4())  # Chroma–Neo4j bridge key
    documents.append(result.get("content", ""))
    metadatas.append({
        "doc_id": doc_id,
        "url": result.get("url", ""),
        "title": result.get("title", ""),
        "snippet": result.get("snippet", ""),
        "source": result.get("source", ""),
        "query_used": query
    })
    ids.append(doc_id)

# print("Documents:", documents)
# print("Metadatas:", metadatas)

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

# print("Successfully stored Tavily results in ChromaDB!")

results = collection.query(
    query_texts=["pressure transmitter competitors"],
    n_results=2
)

# print("Query Results:", results)
