"""Retrieve Node – single‑instance Chroma with read‑only SQLite (Windows friendly)

• collection_name = "recipe_hybrid_rag"  (≈ 680 k vectors)
• persist_directory = VECTOR_DIR (local DB)
• lru_cache → 파일 잠금 충돌 방지
• 검색 결과 0 건이면 branch = "no_docs"
"""
from functools import lru_cache
from typing import Dict, Any, List

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings  # read‑only / telemetry 설정용

VECTOR_DIR = r"C:/Users/SunjaeJeong/Desktop/data/files_data/chroma_recipes_2025_09_16"
COLL_NAME  = "recipes-v1"
K = 4

# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_vectorstore():
    """Create Chroma client *once* in read‑only mode to avoid Windows file‑lock."""
    return Chroma(
        collection_name=COLL_NAME,
        # 배포할때만 VECTOR_DIR로 고치기
        persist_directory=VECTOR_DIR,
        embedding_function=OpenAIEmbeddings(),
        client_settings=Settings(allow_reset=False, anonymized_telemetry=False),
    )

# -----------------------------------------------------------------------------

def retrieve_node(state) -> Dict[str, Any]:
    """LangGraph Runnable: similarity search → docs / no_docs."""
    query = (state.query or "").strip()
    # ✅ 개선
    if not query:
        return {
            **state.dict(),
            "retrieved_docs": [],
            "retrieved_scores": [],  # ✅ 추가 (filter_node와 일관성)
            "branch": "no_docs"
        }
    
    try:
        vs = get_vectorstore()  # cached instance
        results = vs.similarity_search_with_score(query, k=K)
    except Exception as e:
        return {**state.dict(), "retrieved_docs": [], "branch": "no_docs"}
    

    if not results:
        return {**state.dict(), "retrieved_docs": [], "branch": "no_docs"}

    docs = [doc.page_content for doc, _ in results]
    scores = [score for _, score in results]

    return {
        **state.dict(),
        "retrieved_docs": docs,
        "retrieved_scores": scores,  # ✅ 점수 추가
    }
