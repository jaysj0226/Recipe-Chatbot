from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple, Dict
import re

from config.settings import (
    CE_MODEL,
    CE_SENT_T,
    CE_SUPPORT_P,
    CE_MAX_DOCS,
    CE_SNIPPETS_PER_DOC,
    DEBUG_RAW,
    USE_FAKE_LLM,
)


@lru_cache(maxsize=1)
def _load_reranker():
    try:
        from FlagEmbedding import FlagReranker  # type: ignore
    except Exception as e:
        if DEBUG_RAW:
            print(f"verifier_ce: FlagEmbedding import failed: {e}")
        return None
    try:
        return FlagReranker(CE_MODEL, use_fp16=False)
    except Exception as e:
        if DEBUG_RAW:
            print(f"verifier_ce: load model failed: {e}")
        return None


def _normalize_text(text: str) -> str:
    # Lower, collapse spaces, mask numbers
    t = text.strip().lower()
    t = re.sub(r"\d+([.,]\d+)?", "NUM", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # Simple sentence splitter for ko/en
    parts = re.split(r"(?<=[.!?\n])\s+|(?<=\.)\s+|(?<=\n)\n+|(?<=\?|!|\.)\s+", text.strip())
    sents = [s.strip() for s in parts if s and len(s.strip()) >= 5]
    # De-duplicate
    seen = set()
    out = []
    for s in sents:
        key = s[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _extract_snippets(docs: List[str], max_docs: int, per_doc: int) -> List[str]:
    if not docs:
        return []
    snippets: List[str] = []
    for d in docs[: max_docs]:
        if not isinstance(d, str):
            continue
        # Take first N sentences as simple snippets
        sents = _split_sentences(d)
        if not sents:
            continue
        # Sample evenly across the document to cover more content
        if len(sents) <= per_doc:
            picks = sents
        else:
            step = max(1, len(sents) // per_doc)
            picks = [sents[i] for i in range(0, min(len(sents), step * per_doc), step)]
            # Ensure per_doc cap
            picks = picks[:per_doc]
        for s in picks:
            s2 = s[:400]
            snippets.append(s2)
    return snippets


def _is_neutral_sentence(sent: str) -> bool:
    """Filter only generic disclaimers, not substantive recipe content."""
    if not sent:
        return False
    t = sent.lower()
    # êµ¬ì²´ì  ìˆ˜ì¹˜/ë°©ë²•ì´ í¬í•¨ëœ ë¬¸ìž¥ì€ ì œì™¸í•˜ì§€ ì•ŠìŒ
    # ì˜¤ì§ ì¼ë°˜ì  ë©´ì±… ì¡°í•­ë§Œ í•„í„°ë§
    neutral_cues = [
        "ì‹í’ˆ ì•ˆì „ ìˆ˜ì¹™ì„ ì¤€ìˆ˜í•˜ì„¸ìš”",
        "ì•Œë ˆë¥´ê¸°ê°€ ìžˆëŠ” ê²½ìš° ì „ë¬¸ê°€ì™€ ìƒë‹´",
        "ê°œì¸ì˜ ê±´ê°• ìƒíƒœì— ë”°ë¼ ë‹¤ë¥¼ ìˆ˜ ìžˆìŠµë‹ˆë‹¤",
    ]
    return any(cue in t for cue in neutral_cues)

def verify_answer_with_ce(answer: str, docs: List[str]) -> Dict:
    """
    Cross-encoder ê¸°ë°˜ ë¬¸ìž¥-ìŠ¤ë‹ˆíŽ« ë§¤ì¹­ìœ¼ë¡œ grounded ì—¬ë¶€ë¥¼ íŒë‹¨.
    Returns dict: {branch, support_rate, avg, median, supported, total}
    """
    # Early outs
    sents = _split_sentences(answer)
    total = len(sents)
    if total == 0:
        return {
            "branch": "notSure", 
            "confidence_level": "unknown",
            "reasoning": "",
            "support_rate": 0.0, 
            "supported": 0, 
            "total": 0
        }

    # Filter neutral/safety sentences from scoring
    target_sents = [s for s in sents if not _is_neutral_sentence(s)]
    if not target_sents:
        return {
            "branch": "notSure", 
            "confidence_level": "unknown",
            "reasoning": "",
            "support_rate": 0.0, 
            "supported": 0, 
            "total": 0
        }

    snippets = _extract_snippets(docs or [], CE_MAX_DOCS, CE_SNIPPETS_PER_DOC)
    if not snippets:
        return {
            "branch": "notSure", 
            "confidence_level": "unknown",
            "reasoning": "문서 스니펫 없음",
            "support_rate": 0.0, 
            "supported": 0, 
            "total": len(target_sents)
        }

    # If fake LLM mode or reranker missing, fall back to notSure
    if USE_FAKE_LLM:
        return {
            "branch": "notSure", 
            "confidence_level": "unknown",
            "reasoning": "Fake LLM 모드",
            "support_rate": 0.0, 
            "supported": 0, 
            "total": len(target_sents)
        }

    reranker = _load_reranker()
    if reranker is None:
        return {
            "branch": "notSure", 
            "confidence_level": "unknown",
            "reasoning": "Reranker 모델 로딩 실패",
            "support_rate": 0.0, 
            "supported": 0, 
            "total": len(target_sents)
        }

    # Compute sentence max scores
    max_scores: List[float] = []
    for sent in target_sents:
        q = _normalize_text(sent)
        pairs = [[q, _normalize_text(sn)] for sn in snippets]
        try:
            scores = reranker.compute_score(pairs, normalize=True)
            m = max(scores) if scores else 0.0
        except Exception as e:
            if DEBUG_RAW:
                print(f"verifier_ce: score error: {e}")
            m = 0.0
        max_scores.append(float(m))

    # Aggregate
    supported = sum(1 for s in max_scores if s >= CE_SENT_T)
    denom = max(1, len(target_sents))
    support_rate = supported / denom
    avg = sum(max_scores) / len(max_scores) if max_scores else 0.0
    median = sorted(max_scores)[len(max_scores) // 2] if max_scores else 0.0

    # Branch with tolerance delta
    delta = 0.05
    if support_rate >= CE_SUPPORT_P:
        branch = "grounded"
    elif support_rate >= max(0.0, CE_SUPPORT_P - delta):
        branch = "notSure"
    else:
        branch = "notGrounded"
    
    # ✨ notSure 세부 분류 추가
    confidence_level = "unknown"
    if branch == "notSure":
        if support_rate >= 0.40:  # 40~45% (CE_SUPPORT_P-delta ~ 40%)
            confidence_level = "borderline"
        elif support_rate >= 0.20:  # 20~40%
            confidence_level = "weak"
        else:  # < 20%
            confidence_level = "very_weak"
    elif branch == "grounded":
        confidence_level = "high"
    elif branch == "notGrounded":
        confidence_level = "none"

    return {
        "branch": branch,
        "confidence_level": confidence_level,  # ✨ 새 필드
        "support_rate": support_rate,
        "avg": avg,
        "median": median,
        "supported": supported,
        "total": denom,
    }