"""Configuration and Environment Variables"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Project root paths
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

# Load .env from project root (override=True to override system env vars)
load_dotenv(BASE_DIR / ".env", override=True)

# Vector DB configuration
_VECTOR_DIR_RAW = os.environ.get("VECTOR_DIR", str(BASE_DIR / "chroma_db"))
try:
    VECTOR_DIR = str(Path(_VECTOR_DIR_RAW).expanduser())
except Exception:
    VECTOR_DIR = _VECTOR_DIR_RAW

COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "recipes-v1")

# ✅ 수정: OpenAI 임베딩 기본값
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")

# Defaults
# 검색 문서 개수 기본값 (Answer Relevancy 향상을 위해 12개로 증가)
K_DEFAULT = int(os.environ.get("K_DEFAULT", "12"))
SCORE_THRESHOLD = float(os.environ.get("GROUPA_SCORE_THRESHOLD", "0.0"))

# LLM configuration
ROUTER_MODEL = os.environ.get("GROUPA_ROUTER_MODEL", "gpt-4o")
GENERATION_MODEL = os.environ.get("GROUPA_GENERATION_MODEL", "gpt-4o")
REWRITE_MODEL = os.environ.get("GROUPA_REWRITE_MODEL", "gpt-4o")
JUDGE_MODEL = os.environ.get("GROUPA_JUDGE_MODEL", "gpt-4o")
OOD_MODEL = os.environ.get("GROUPA_OOD_MODEL", ROUTER_MODEL)
OOD_TEMPERATURE = float(os.environ.get("GROUPA_OOD_TEMPERATURE", "0"))
OOD_PROTOTYPES_PATH = os.environ.get(
    "GROUPA_OOD_PROTOTYPES_PATH", str(BASE_DIR / "config" / "ood_prototypes.json")
)

# Temperature settings (Faithfulness 개선)
GENERATION_TEMPERATURE = float(os.environ.get("GENERATION_TEMPERATURE", "0.0"))
ROUTER_TEMPERATURE = float(os.environ.get("ROUTER_TEMPERATURE", "0.0"))
REWRITE_TEMPERATURE = float(os.environ.get("REWRITE_TEMPERATURE", "0.0"))
JUDGE_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.0"))

# ✅ 수정: prefix 제거
OOD_COS_THRESHOLD = float(os.environ.get("OOD_COS_THRESHOLD", "0.35"))
OOD_COS_MARGIN = float(os.environ.get("OOD_COS_MARGIN", "0.05"))

# Moderation (harmful content filtering)
ENABLE_MODERATION = os.environ.get("ENABLE_MODERATION", "1") == "1"
MODERATION_MODEL = os.environ.get("MODERATION_MODEL", "omni-moderation-latest")

# Feature flags
ALLOW_NO_CONTEXT_ANSWER = os.environ.get("ALLOW_NO_CONTEXT_ANSWER", "0") == "1"
ENABLE_QUERY_REWRITE = os.environ.get("ENABLE_QUERY_REWRITE", "1") == "1"
ENABLE_CRAG = os.environ.get("ENABLE_CRAG", "1") == "1"
DEBUG_RAW = os.environ.get("GROUPA_DEBUG_RAW", "1") == "1"  # ✅ 디버그 모드 활성화

# CI/test mode (fake LLM / no vector)
USE_FAKE_LLM = os.environ.get("USE_FAKE_LLM", "0") == "1"

# Retrieval/Rerank configuration
RERANK_MMR = os.environ.get("RERANK_MMR", "1") == "1"
MMR_FETCH = int(os.environ.get("MMR_FETCH", "150"))  # ✅ 100 → 150 증가
MMR_LAMBDA = float(os.environ.get("MMR_LAMBDA", "0.7"))
MIN_DOC_LEN = int(os.environ.get("MIN_DOC_LEN", "20"))  # ✅ 80 → 20 완화 (짧은 레시피 허용)

# ✅ 수정: Answer Relevancy 향상을 위해 더 완화 (0.15 → 0.08)
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.08"))

DOMAIN_CAP = int(os.environ.get("DOMAIN_CAP", "3"))
LOWCONF_MODE = os.environ.get("LOWCONF_MODE", "balanced").strip().lower()
MIN_CONF_DOCS = int(os.environ.get("MIN_CONF_DOCS", "1"))  # ✅ 2 → 1

# Hybrid Search configuration (BM25 + Vector)
USE_HYBRID_SEARCH = os.environ.get("USE_HYBRID_SEARCH", "1") == "1"
HYBRID_ALPHA = float(os.environ.get("HYBRID_ALPHA", "0.5"))  # 0.5 = 동등 가중치
HYBRID_K_RRF = int(os.environ.get("HYBRID_K_RRF", "60"))  # RRF 상수
HYBRID_FETCH_K = int(os.environ.get("HYBRID_FETCH_K", "24"))  # Dense/Sparse 각각 fetch 수

# Cross-Encoder reranker (optional)
USE_CE_RERANK = os.environ.get("USE_CE_RERANK", "0") == "1"
CE_MODEL = os.environ.get("CE_MODEL", "BAAI/bge-reranker-base")  # ✅ 수정
CE_TOPN = int(os.environ.get("CE_TOPN", "30"))

# ✅ 수정: Cross-Encoder 기본값 완화
CE_SENT_T = float(os.environ.get("CE_SENT_T", "0.15"))      # 0.30 → 0.15
CE_SUPPORT_P = float(os.environ.get("CE_SUPPORT_P", "0.15"))  # 0.60 → 0.15
CE_MAX_DOCS = int(os.environ.get("CE_MAX_DOCS", "8"))
CE_SNIPPETS_PER_DOC = int(os.environ.get("CE_SNIPPETS_PER_DOC", "3"))

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Supported intents
SUPPORTED_INTENTS = [
    "recipe",
    "dish_overview",
    "storage",
    "substitution",
    "nutrition",
    "equipment",
    "shopping",
    "unknown",
    "out_of_domain",
]
