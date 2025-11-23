import os
import chromadb
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

from tavily import TavilyClient
import chromadb
import uuid

api_key = os.getenv("TAVILY_API_KEY")

# --- 1. Initialize Tavily ---
tavily = TavilyClient(api_key=api_key)

# Example Tavily Query
query = "Honeywell ST700 pressure transmitter competitors"
tavily_results = tavily.search(query=query).get("results")[0:3]

# print("Tavily Search Results:", tavily_results)

# --- 2. Initialize ChromaDB ---
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="tavily_results")

# --- 3. Store Tavily results into ChromaDB ---
documents = []
metadatas = []
ids = []

for search_result in tavily_results:
    for key, value in search_result.items():
        # print(f"{key}: {value}")
        if key == "content":
            documents.append(value)
        else:
            metadatas.append({key: value})

print("Documents:", documents)
print("Metadatas:", metadatas)

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=[x for x in range(len(documents))]
)