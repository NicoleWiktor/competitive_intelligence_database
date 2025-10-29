import os
from dotenv import load_dotenv


# Load environment variables from a local .env if present
load_dotenv()


# ----- Tavily API -----
def get_tavily_api_key() -> str:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Create a .env with TAVILY_API_KEY=<your-key>."
        )
    return api_key


# ----- Chunking defaults -----
def get_chunk_params() -> dict:
    """Return chunker parameters with safe defaults.

    You can override via env:
      - CHUNK_SIZE (characters)
      - CHUNK_OVERLAP (characters)
    """
    size = int(os.getenv("CHUNK_SIZE", "3000"))
    overlap = int(os.getenv("CHUNK_OVERLAP", "300"))
    return {"chunk_size": size, "chunk_overlap": overlap}


# ----- OpenAI API -----
def get_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env to run LLM extraction."
        )
    return key


# ----- Neo4j -----
def get_neo4j_config() -> dict:
    return {
        "uri": os.getenv("NEO4J_URI", ""),
        "user": os.getenv("NEO4J_USER", ""),
        "password": os.getenv("NEO4J_PASSWORD", ""),
    }

