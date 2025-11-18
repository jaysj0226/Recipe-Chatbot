"""Vectorstore Utility - Cached Chroma Instance (with test-friendly fallback)"""
from functools import lru_cache
from typing import Any

from config.settings import (
    VECTOR_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    USE_FAKE_LLM,
)


class _FakeCollection:
    def count(self) -> int:
        return 0


class _FakeVectorStore:
    def __init__(self) -> None:
        self._collection = _FakeCollection()

    def similarity_search_with_score(self, query: str, k: int = 10) -> list[tuple[Any, float]]:
        return []


def _real_chroma():
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    from chromadb.config import Settings

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DIR,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        client_settings=Settings(
            allow_reset=False,
            anonymized_telemetry=False,
        ),
    )


@lru_cache(maxsize=1)
def get_vectorstore():
    """
    Return a cached vectorstore. In USE_FAKE_LLM mode or when OPENAI_API_KEY is
    missing, return a no-op fake that yields empty results for deterministic tests.
    """
    if USE_FAKE_LLM or not OPENAI_API_KEY:
        return _FakeVectorStore()
    try:
        return _real_chroma()
    except Exception as e:
        print(f"Falling back to fake vectorstore due to error: {e}")
        return _FakeVectorStore()


def get_collection_count() -> int:
    try:
        vs = get_vectorstore()
        return getattr(vs, "_collection", _FakeCollection()).count()
    except Exception as e:
        print(f"Error getting collection count: {e}")
        return 0
