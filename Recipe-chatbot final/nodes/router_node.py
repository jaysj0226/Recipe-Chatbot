"""Router Node - Intent Classification (Structured Output)"""
from __future__ import annotations

import json
import re
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from config.settings import ROUTER_MODEL, SUPPORTED_INTENTS, USE_FAKE_LLM
from prompts.templates import ROUTER_PROMPT


class _RouteSchema(BaseModel):
    intent: str = Field(..., description="One of supported intents or out_of_domain")
    needs_retrieval: bool = Field(True, description="Whether vector retrieval is needed")
    notes: Optional[str] = Field(None, description="Optional notes or rationale")


def _looks_in_domain(text: str) -> bool:
    """Lightweight heuristic: detect cooking/recipe topics quickly."""
    if not text:
        return False
    t = text.lower()
    cues = [
        # Korean
        "요리", "레시피", "만드는", "방법", "재료", "보관", "영양", "조리", "메뉴", "추천",
        "카레", "소스", "치킨", "수프", "찌개", "스튜", "볶음", "구이",
        # English
        "recipe", "cook", "cooking", "ingredients", "storage", "nutrition", "substitute", "dish",
    ]
    return any(cue in t for cue in cues)


def _semantic_router_fallback(q: str) -> tuple[str, bool, str]:
    """Lightweight intent guesser using keyword cues.

    Returns (intent, needs_retrieval, note)
    """
    t = (q or "").lower()
    # Simple priority mapping
    patterns = [
        (r"보관|온도|포장|냉동|보존|storage|shelf life|expire", "storage"),
        (r"대체|치환|없\s*이|substitut|replace|allerg", "substitution"),
        (r"칼로리|영양|영양소|탄수|단백|지방|nutrition|calorie|macro|kcal", "nutrition"),
        (r"도구|장비|에어\s*프라이어|팬|오븐|equipment|tool|pan|oven|air fryer", "equipment"),
        (r"구매|쇼핑|살까|사기|shopping|buy|purchase", "shopping"),
        (r"무엇|뭐야|기원|유래|특징|overview|about", "dish_overview"),
        (r"레시피|만드|어떻게|방법|steps|how to|make|cook", "recipe"),
    ]
    for pat, intent in patterns:
        if re.search(pat, t):
            return intent, (intent != "out_of_domain"), "semantic_fallback"
    # default
    intent = "recipe" if _looks_in_domain(q) else "out_of_domain"
    return intent, (intent != "out_of_domain"), "semantic_default"


def router_node(query: str, context: str = "") -> Dict[str, Any]:
    """
    Router Node: 질의 의도 분류 (구조화 출력 기반)

    Returns dict with: intent, needs_retrieval, notes
    """
    # Fake/deterministic mode for tests/CI
    if USE_FAKE_LLM:
        intent = "recipe" if _looks_in_domain(query) else "out_of_domain"
        needs_retrieval = intent != "out_of_domain"
        return {
            "intent": intent,
            "needs_retrieval": needs_retrieval,
            "notes": "fake_router",
        }

    q_for_router = query if not context else f"{query}\n\n[참고맥락]\n{context}"

    # Try 1) Pydantic-structured output
    data: Dict[str, Any] = {}
    try:
        llm_struct = ChatOpenAI(model=ROUTER_MODEL, temperature=0)
        parser_llm = llm_struct.with_structured_output(_RouteSchema)
        res: _RouteSchema = parser_llm.invoke(ROUTER_PROMPT.format_messages(q=q_for_router))
        data = res.dict()
    except Exception:
        # Try 2) JSON object forced response_format
        try:
            llm_json = ChatOpenAI(
                model=ROUTER_MODEL,
                temperature=0,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
            raw = llm_json.invoke(ROUTER_PROMPT.format_messages(q=q_for_router)).content or "{}"
            data = json.loads(raw)
        except Exception as e:
            from config.settings import DEBUG_RAW
            if DEBUG_RAW:
                print(f"router_structured_error: {e}")
            data = {}

    # Validate and normalize
    intent = (data.get("intent") or "").strip() if isinstance(data, dict) else ""
    if intent not in SUPPORTED_INTENTS:
        # apply semantic fallback when intent invalid or missing
        s_intent, s_need, s_note = _semantic_router_fallback(query)
        intent = s_intent
        needs_retrieval = s_need
        notes = (data.get("notes", "") if isinstance(data, dict) else "").strip()
        notes = (notes + (" | " if notes else "") + s_note).strip()
        return {"intent": intent, "needs_retrieval": needs_retrieval, "notes": notes}

    needs_retrieval = bool(data.get("needs_retrieval", True))
    notes = (data.get("notes", "") or "").strip()

    # Heuristic override if LLM says out_of_domain but looks like cooking
    if intent == "out_of_domain" and _looks_in_domain(query):
        s_intent, s_need, s_note = _semantic_router_fallback(query)
        intent = s_intent if s_intent in SUPPORTED_INTENTS else "recipe"
        needs_retrieval = s_need
        notes = (notes + " | overridden_from_ood_by_heuristic").strip()

    return {"intent": intent, "needs_retrieval": needs_retrieval, "notes": notes}

