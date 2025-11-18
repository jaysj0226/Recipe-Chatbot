from fastapi import APIRouter
from config.settings import (
    VECTOR_DIR,
    COLLECTION_NAME,
    SCORE_THRESHOLD,
    ROUTER_MODEL,
    ALLOW_NO_CONTEXT_ANSWER,
    ENABLE_CRAG,
    JUDGE_MODEL,
    EMBEDDING_MODEL,
    USE_FAKE_LLM,
    USE_CE_RERANK,
    CE_MODEL,
    SIMILARITY_THRESHOLD,
    DOMAIN_CAP,
    LOWCONF_MODE,
    ENABLE_MODERATION,
    MODERATION_MODEL,
)
from config.schemas import HealthResponse
from utils.vectorstore import get_collection_count


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return {
        "ok": True,
        "persist": VECTOR_DIR,
        "collection": COLLECTION_NAME,
        "score_threshold": SCORE_THRESHOLD,
        "embed_model": EMBEDDING_MODEL,
        "total_docs": get_collection_count(),
        "router_model": ROUTER_MODEL,
        "judge_model": JUDGE_MODEL,
        "allow_no_context_answer": ALLOW_NO_CONTEXT_ANSWER,
        "enable_crag": ENABLE_CRAG,
        "architecture": "Modular Nodes with Conversation Memory",
        "status": "지식기반 처리(세션 기반)",
        # Diagnostics
        "fake_mode": USE_FAKE_LLM,
        "ce_rerank_enabled": USE_CE_RERANK,
        "ce_model": CE_MODEL,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "domain_cap": DOMAIN_CAP,
        "lowconf_mode": LOWCONF_MODE,
        "moderation_enabled": ENABLE_MODERATION,
        "moderation_model": MODERATION_MODEL,
    }


@router.get("/doc_count")
def get_doc_count():
    try:
        count = get_collection_count()
        return {"total_docs": count}
    except Exception as e:
        from config.settings import DEBUG_RAW
        if DEBUG_RAW:
            print(f"doc_count_error: {e}")
        return {"error": str(e)}
