"""
Configuration module for the competitive intelligence pipeline.

Loads API keys and database credentials from environment variables (.env file).
Required environment variables:
    - OPENAI_API_KEY: OpenAI API key for GPT-4o-mini
    - TAVILY_API_KEY: Tavily API key for web search
    - NEO4J_URI: Neo4j database URI (e.g., bolt://localhost:7687)
    - NEO4J_USER: Neo4j username (usually 'neo4j')
    - NEO4J_PASSWORD: Neo4j password
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file in project root
load_dotenv()


def get_tavily_api_key() -> str:
    """Get Tavily API key for web search functionality."""
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Create a .env with TAVILY_API_KEY=<your-key>."
        )
    return api_key


def get_openai_api_key() -> str:
    """Get OpenAI API key for LLM-based data extraction."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env to run LLM extraction."
        )
    return key


def get_neo4j_config() -> dict:
    """Get Neo4j database connection configuration."""
    return {
        "uri": os.getenv("NEO4J_URI", ""),
        "user": os.getenv("NEO4J_USER", ""),
        "password": os.getenv("NEO4J_PASSWORD", ""),
    }

