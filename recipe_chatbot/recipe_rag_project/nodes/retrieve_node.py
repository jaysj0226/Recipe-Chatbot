"""Retrieve Node - Vector Search with MMR and basic filtering"""
from typing import Dict, Any, List
import re
from urllib.parse import urlparse

from config.settings import (
    K_DEFAULT,
    RERANK_MMR,
    MMR_FETCH,
    MMR_LAMBDA,
    MIN_DOC_LEN,
    SIMILARITY_THRESHOLD,
    DOMAIN_CAP,
)
from utils.vectorstore import get_vectorstore


def _extract_image_url_from_meta(meta: dict) -> str | None:
    if not meta:
        return None
    candidates = []
    for key in ("image_url", "image", "img_url", "thumbnail", "thumb_url", "url"):
        val = meta.get(key)
        if isinstance(val, str):
            candidates.append(val)
    for val in candidates:
        if isinstance(val, str) and (val.startswith("http://") or val.startswith("https://")):
            return val
    return None


def _extract_image_url_from_text(text: str) -> str | None:
    if not text:
        return None
    # Pattern for lines like: "Image: https://..." or embed-style URLs
    m = re.search(r"(?im)^\s*image\s*:\s*(https?://\S+)", text)
    if m:
        return m.group(1).strip()
    # Fallback: any http(s) URL ending with common image extensions
    m2 = re.search(r"(https?://\S+\.(?:png|jpe?g|gif|webp|svg))", text, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return None


def _extract_title_from_meta(meta: dict) -> str | None:
    if not meta:
        return None
    for key in ("title", "name", "recipe", "page_title"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _extract_url_from_meta(meta: dict) -> str | None:
    if not meta:
        return None
    for key in ("source", "url", "link"):
        val = meta.get(key)
        if isinstance(val, str) and (val.startswith("http://") or val.startswith("https://")):
            return val.strip()
    return None


def retrieve_node(query: str, k: int = K_DEFAULT) -> Dict[str, Any]:
    """
    Retrieve Node: 벡터 DB에서 유사도 검색
    
    Args:
        query: 검색 쿼리
        k: 검색할 문서 수
        
    Returns:
        Dict containing:
            - retrieved_docs: 검색된 문서 리스트
            - retrieved_scores: 유사도 점수 리스트
            - branch: "has_docs" | "no_docs"
    """
    # 빈 쿼리 처리
    if not query or not query.strip():
        return {
            "retrieved_docs": [],
            "retrieved_scores": [],
            "branch": "no_docs"
        }
    
    try:
        vs = get_vectorstore()
        if RERANK_MMR and hasattr(vs, "max_marginal_relevance_search"):
            # Fetch wider, then MMR to k
            docs_mmr = vs.max_marginal_relevance_search(
                query, k=k, fetch_k=max(k, MMR_FETCH), lambda_mult=MMR_LAMBDA
            )
            results = [(d, None) for d in docs_mmr]
            score_mode = "mmr"
        else:
            results = vs.similarity_search_with_score(query, k=k)
            score_mode = "distance"
    except Exception as e:
        from config.settings import DEBUG_RAW
        if DEBUG_RAW:
            print(f"retrieve_vectorstore_error: {e}")
        return {
            "retrieved_docs": [],
            "retrieved_scores": [],
            "branch": "no_docs"
        }
    
    # 검색 결과가 없을 경우
    if not results:
        return {
            "retrieved_docs": [],
            "retrieved_scores": [],
            "branch": "no_docs"
        }
    
    # 문서와 점수 분리
    docs: List[str] = []
    scores: List[float] = []
    images: List[str] = []
    metas: List[dict] = []

    for doc, score in results:
        content = doc.page_content
        # min length filter
        if not content or len(content) < MIN_DOC_LEN:
            continue
        docs.append(content)
        # Convert distance to similarity if present
        if isinstance(score, (int, float)):
            sim = 1.0 - float(score)
            scores.append(sim)
        else:
            scores.append(None)

        meta = getattr(doc, "metadata", {}) or {}
        url = _extract_image_url_from_meta(meta)
        if not url:
            url = _extract_image_url_from_text(content)
        images.append(url or "")
        metas.append({
            "title": _extract_title_from_meta(meta) or "",
            "url": _extract_url_from_meta(meta) or "",
        })

    # If some scores are missing (MMR path), try to backfill using a scored search
    if any(s is None for s in scores):
        try:
            fetch_n = max(k, MMR_FETCH)
            vs = get_vectorstore()
            scored = vs.similarity_search_with_score(query, k=fetch_n)

            def _key(meta: dict, content: str) -> tuple:
                urlk = (meta.get("url") or meta.get("source") or "").strip() if isinstance(meta, dict) else ""
                titlek = (meta.get("title") or meta.get("name") or "").strip() if isinstance(meta, dict) else ""
                sig = hash((content[:256] if isinstance(content, str) else ""))
                return (urlk, titlek, sig)

            score_map = {}
            for d, dist in scored:
                m = getattr(d, "metadata", {}) or {}
                score_map[_key(m, d.page_content)] = 1.0 - float(dist)

            for i in range(len(docs)):
                if scores[i] is None:
                    key = _key(metas[i], docs[i])
                    if key in score_map:
                        scores[i] = score_map[key]
        except Exception as e:
            from config.settings import DEBUG_RAW
            if DEBUG_RAW:
                print(f"retrieve_backfill_score_error: {e}")

    # Optional similarity cutoff (only if we have computed similarity values)
    if SIMILARITY_THRESHOLD and any(s is not None for s in scores):
        kept_docs: List[str] = []
        kept_scores: List[float] = []
        kept_images: List[str] = []
        kept_metas: List[dict] = []
        for d, s, i, m in zip(docs, scores, images, metas):
            if s is None or s >= SIMILARITY_THRESHOLD:
                kept_docs.append(d)
                kept_scores.append(s)
                kept_images.append(i)
                kept_metas.append(m)
        docs, scores, images, metas = kept_docs, kept_scores, kept_images, kept_metas

    # Domain cap to reduce same-site dominance
    if DOMAIN_CAP and DOMAIN_CAP > 0:
        seen: dict[str, int] = {}
        kept_docs: List[str] = []
        kept_scores: List[float] = []
        kept_images: List[str] = []
        kept_metas: List[dict] = []
        for d, s, i, m in zip(docs, scores, images, metas):
            domain = ""
            try:
                u = (m.get("url") or "").strip()
                domain = urlparse(u).netloc or ""
            except Exception:
                domain = ""
            cnt = seen.get(domain, 0)
            if domain and cnt >= DOMAIN_CAP:
                continue
            seen[domain] = cnt + 1
            kept_docs.append(d)
            kept_scores.append(s)
            kept_images.append(i)
            kept_metas.append(m)
        docs, scores, images, metas = kept_docs, kept_scores, kept_images, kept_metas

    return {
        "retrieved_docs": docs,
        "retrieved_scores": scores,
        "branch": "has_docs",
        "retrieved_images": images,
        "retrieved_meta": metas,
        "score_mode": "mmr_backfilled" if any(s is None for s in scores) is False and RERANK_MMR else ("mmr" if RERANK_MMR else "distance"),
    }
