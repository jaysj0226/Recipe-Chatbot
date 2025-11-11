from __future__ import annotations

from functools import lru_cache
from typing import List


@lru_cache(maxsize=1)
def _load_reranker(model_name: str):
    try:
        from FlagEmbedding import FlagReranker  # type: ignore
    except Exception as e:
        print(f"Cross-encoder reranker not available (FlagEmbedding import failed): {e}")
        return None
    try:
        return FlagReranker(model_name, use_fp16=False)
    except Exception as e:
        print(f"Failed to load reranker model {model_name}: {e}")
        return None


def rerank_pairs(query: str, docs: List[str], topn: int, model_name: str) -> List[int]:
    """Return indices of docs sorted by cross-encoder score (desc), up to topn.

    If reranker is unavailable, return the first topn indices as-is.
    """
    reranker = _load_reranker(model_name)
    if reranker is None or not docs:
        return list(range(min(len(docs), topn)))

    pairs = [[query, d] for d in docs]
    try:
        scores = reranker.compute_score(pairs, normalize=True)
    except Exception as e:
        print(f"Rerank scoring failed: {e}")
        return list(range(min(len(docs), topn)))

    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    return order[: min(len(order), topn)]

