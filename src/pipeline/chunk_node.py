from __future__ import annotations

from typing import Iterable, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import get_chunk_params


def recursive_character_split(text: str) -> List[str]:
    """Chunk text using LangChain's RecursiveCharacterTextSplitter.

    Uses env-configured chunk size/overlap from settings.
    """
    params = get_chunk_params()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=params["chunk_size"],
        chunk_overlap=params["chunk_overlap"],
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def preview_chunks(chunks: Iterable[str], limit: int = 2) -> None:
    for i, c in enumerate(chunks):
        if i >= limit:
            break
        print(f"--- chunk {i} ({len(c)} chars) ---")
        print(c[:600] + ("..." if len(c) > 600 else ""))
        print()


if __name__ == "__main__":
    demo = (
        "LangChain supports modular pipelines for AI workflows. "
        "These workflows include document loading, chunking, retrieval, and LLM integration. "
        "LangChain simplifies AI model deployment."
    )
    cs = recursive_character_split(demo)
    preview_chunks(cs, limit=3)

