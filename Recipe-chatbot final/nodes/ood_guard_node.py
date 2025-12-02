from __future__ import annotations

"""OOD Guard Node (pre-router, hybrid)

Hybrid gating: Moderation -> embedding domain score -> LLM fallback near threshold.
"""

from typing import Dict, Any, Optional, List
import json
from functools import lru_cache
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from openai import OpenAI

from config.settings import (
    OOD_MODEL,
    OOD_TEMPERATURE,
    USE_FAKE_LLM,
    OOD_PROTOTYPES_PATH,
    OOD_COS_THRESHOLD,
    OOD_COS_MARGIN,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    ENABLE_MODERATION,
    MODERATION_MODEL,
)


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def get_moderation_report(q: str) -> Optional[Dict[str, Any]]:
    """Return raw moderation result {flagged, categories, category_scores} if available."""
    if not ENABLE_MODERATION or not OPENAI_API_KEY or USE_FAKE_LLM:
        return None
    try:
        client = OpenAI()
        resp = client.moderations.create(model=MODERATION_MODEL, input=q)
        if not resp or not getattr(resp, "results", None):
            return None
        res = resp.results[0]
        out = {
            "flagged": bool(getattr(res, "flagged", False)),
            "categories": dict(getattr(res, "categories", {}) or {}),
        }
        # Some SDKs expose category_scores; if present, include
        if hasattr(res, "category_scores"):
            try:
                out["category_scores"] = dict(getattr(res, "category_scores", {}) or {})
            except Exception:
                pass
        return out
    except Exception:
        return None


def _moderate_text(q: str) -> Optional[Dict[str, Any]]:
    """Use OpenAI Moderation to detect harmful content in a maintainable, data-driven way.

    The moderation API evolves; avoid hard-coding logic in many if/elses.
    We interpret the boolean category flags and apply ordered rules defined
    in a single mapping for easy maintenance.
    Docs: https://platform.openai.com/docs/guides/moderation
    """
    rep = get_moderation_report(q)
    if not rep:
        return None

    cats: Dict[str, bool] = dict(rep.get("categories", {}) or {})
    flagged: bool = bool(rep.get("flagged", False))

    def out(msg: str) -> Dict[str, Any]:
        return {"branch": "out", "answer": msg, "method": "moderation"}

    # Ordered rules: [(category_key, localized_message)]
    # Keep this single source of truth aligned with the moderation docs.
    RULES: List[tuple[str, str]] = [
        ("sexual/minors", "정책상 미성년자가 포함된 성적 내용은 엄격히 금지되어 답변할 수 없습니다."),
        ("self-harm/instructions", "자해/자살과 관련된 방법이나 조언은 제공할 수 없습니다."),
        ("violence/graphic", "잔혹하거나 매우 폭력적인 내용에는 답변할 수 없습니다."),
        ("illicit/violent", "폭력적 불법 행위에 대한 조언은 제공할 수 없습니다."),
        ("illicit", "불법 행위에 대한 조언은 제공할 수 없습니다."),
        ("hate/threatening", "혐오·차별적 내용에는 답변할 수 없습니다. 다른 방식으로 질문해 주세요."),
        ("hate", "혐오·차별적 내용에는 답변할 수 없습니다. 다른 방식으로 질문해 주세요."),
        ("harassment/threatening", "폭력적·협박적 표현은 허용되지 않습니다. 정중한 표현으로 바꿔 주세요."),
        ("harassment", "모욕적 표현은 허용되지 않습니다. 정중한 표현으로 질문해 주세요."),
        ("sexual", "성적·음란한 내용에는 답변할 수 없습니다."),
    ]

    for key, message in RULES:
        if cats.get(key, False):
            return out(message)

    # Fallback: if overall flagged but no category above matched
    if flagged:
        return out("안전 및 정책상 해당 문의에는 답변할 수 없습니다.")
    return None


@lru_cache(maxsize=1)
def _load_prototypes() -> List[str]:
    # Load from JSON if present, else fallback defaults
    try:
        p = Path(OOD_PROTOTYPES_PATH)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            pros = data.get("prototypes_in") or []
            pros = [s for s in pros if isinstance(s, str) and s.strip()]
            if pros:
                return pros
    except Exception:
        pass
    # Defaults (compact, domain-representative)
    return [
        "이 요리는 어떻게 만들지?",
        "레시피 단계와 필요한 재료",
        "조리 시간과 온도는 어떻게 조절하지?",
        "남은 재료로 만들 수 있는 요리 추천",
        "보관 방법과 유통기한",
        "칼로리와 영양 성분 안내",
        "How to cook this dish?",
        "Recipe steps and ingredients list",
        "Cooking time and oven temperature",
        "Food storage and shelf life",
        "Calories and nutrition facts",
    ]


@lru_cache(maxsize=1)
def _load_centroid() -> Optional[List[float]]:
    # If no API key or fake mode, skip centroid
    if USE_FAKE_LLM or not OPENAI_API_KEY:
        return None
    try:
        emb = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        texts = _load_prototypes()
        vecs = emb.embed_documents(texts)
        if not vecs:
            return None
        dim = len(vecs[0])
        acc = [0.0] * dim
        for v in vecs:
            if not v or len(v) != dim:
                continue
            for i in range(dim):
                acc[i] += float(v[i])
        n = float(len(vecs))
        if n <= 0:
            return None
        return [x / n for x in acc]
    except Exception:
        return None


_PROMPT = PromptTemplate.from_template(
    """너는 질문이 '요리/레시피/조리/재료/보관/영양' 주제인지 분류하는 분류기다.
규칙: 해당하면 in, 아니면 out 만 출력(설명 금지).
질문: {q}
"""
)


def ood_guard(query: str) -> Dict[str, Any]:
    """Return {branch: 'in'|'out', answer?: str, score?: float, method?: str}."""
    q = (query or "").strip()
    if not q:
        return {
            "branch": "out",
            "answer": "질문을 입력해 주세요. 요리·레시피·조리·재료·영양 주제에 맞춰 도와드릴게요.",
            "method": "empty",
        }

    # Moderation first (safety)
    mod = _moderate_text(q)
    if mod:
        return mod

    # 1) Embedding-based domain score
    centroid = _load_centroid()
    if centroid is not None:
        try:
            emb = OpenAIEmbeddings(model=EMBEDDING_MODEL)
            q_vec = emb.embed_query(q)
            score = _cosine(q_vec, centroid)
            # Two-sided margin for LLM arbitration near the threshold
            lo = OOD_COS_THRESHOLD - OOD_COS_MARGIN
            hi = OOD_COS_THRESHOLD + OOD_COS_MARGIN
            if score >= hi:
                return {"branch": "in", "score": float(score), "method": "embed"}
            if score <= lo:
                return {
                    "branch": "out",
                    "answer": "죄송해요. 해당 문의는 요리·레시피·조리·보관·영양 주제에 한해 답변해 드려요.",
                    "score": float(score),
                    "method": "embed",
                }
            # fallthrough to LLM if borderline
        except Exception:
            pass

    # 2) Fake mode: be permissive
    if USE_FAKE_LLM:
        return {"branch": "in", "method": "fake"}

    # 3) LLM fallback
    try:
        llm = ChatOpenAI(model=OOD_MODEL, temperature=OOD_TEMPERATURE)
        verdict = (llm.invoke(_PROMPT.format_messages(q=q)).content or "").strip().lower()
        if verdict == "in":
            return {"branch": "in", "method": "llm"}
        return {
            "branch": "out",
            "answer": "죄송해요. 해당 문의는 요리·레시피·조리·보관·영양 주제에 한해 답변해 드려요.",
            "method": "llm",
        }
    except Exception:
        # On error, be permissive
        return {"branch": "in", "method": "error-permissive"}
