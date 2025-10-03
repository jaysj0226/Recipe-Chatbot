
from typing import Dict, Any, List
import os

FILTER_SCORE = float(os.environ.get("FILTER_SCORE", "0.20"))

def filter_low_similarity_node(state) -> Dict[str, Any]:
    docs: List[str] = state.retrieved_docs or []
    scores: List[float] = getattr(state, "retrieved_scores", []) or []
    if not docs:
        return {**state.dict()}
    if not scores or len(scores) != len(docs):
        kept_docs = docs
    else:
        kept_docs = [d for d,s in zip(docs, scores) if (s is not None and s >= FILTER_SCORE)]
    return {**state.dict(), "retrieved_docs": kept_docs}
