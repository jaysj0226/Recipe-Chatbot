# -*- coding: utf-8 -*-
"""CRAG Relevance/Judge Node

Grounding verifier that judges whether an answer is grounded in the retrieved
documents. Returns a branch label among: 'grounded', 'notGrounded', 'notSure'.
"""
from typing import Dict, Any, List

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from config.settings import JUDGE_MODEL, USE_FAKE_LLM, DEBUG_RAW
from utils.verifier_ce import verify_answer_with_ce


def relevance_check_node(answer: str, docs: List[str]) -> Dict[str, Any]:
    """Judge grounding of an answer against docs and return branch verdict.

    Args:
        answer: Generated answer string.
        docs: List of retrieved context strings.

    Returns:
        Dict with keys: 'branch' in {'grounded','notGrounded','notSure'}
    """
    context = "\n\n".join(docs or [])

    # Fake/deterministic mode for tests/CI
    if USE_FAKE_LLM:
        if not docs or not context.strip() or not (answer or "").strip():
            return {"branch": "notSure"}
        # simple overlap check
        snippet = (answer or "")[:50]
        return {"branch": "grounded" if snippet and snippet in context else "notSure"}

    # If no docs, we cannot judge confidently
    if not docs or not context.strip():
        return {"branch": "notSure", "metrics": {"support_rate": 0.0, "supported": 0, "total": 0}}

    # Try CE-based verifier first
    try:
        ce_res = verify_answer_with_ce(answer, docs)
        if ce_res and isinstance(ce_res, dict):
            return {"branch": ce_res.get("branch", "notSure"), "metrics": ce_res}
    except Exception as _e:
        if DEBUG_RAW:
            print(f"verifier_ce fallback to LLM: {_e}")

    prompt = PromptTemplate.from_template(
        """당신은 '답변'의 근거성을 판단하는 평가자입니다.

아래 '답변' 내용이 '문서' 근거에 충실한지 구분하세요.

답변:
{answer}

문서:
{context}

판정 기준:
- grounded: 핵심 사실(재료, 분량, 조리 단계/시간, 온도 등)이 문서에 근거하고 있음. 상식 정리, 구조화, 표현 개선은 허용.
- notGrounded: 핵심 재료/조리 관계가 문서에 없거나 문서와 다른 지시/지식/설명이 있음. 문서와 모순되는 내용 포함.
- notSure: 문서가 불충분해 나머지를 추론해야 하거나, 문서가 질문과 간접적으로만 관련. 확신이 어려운 경우.

아래 중 하나만 출력하세요: 'grounded' | 'notGrounded' | 'notSure'
설명은 출력하지 마세요."""
    )

    llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)
    judge_chain = prompt | llm | StrOutputParser()

    try:
        verdict = (judge_chain.invoke({"answer": answer, "context": context}) or "").strip().lower()
        if DEBUG_RAW:
            print(f"\n=== JUDGE DEBUG ===\nVerdict: {verdict}\nAns: {answer[:200]}...\nCtx: {context[:300]}...\n===================\n")
    except Exception as e:
        if DEBUG_RAW:
            print(f"relevance_check_invoke_error: {e}")
        verdict = "notsure"

    if verdict not in ("grounded", "notgrounded", "notsure"):
        verdict = "notsure"

    map_back = {"grounded": "grounded", "notgrounded": "notGrounded", "notsure": "notSure"}
    return {"branch": map_back.get(verdict, "notSure"), "metrics": {}}

