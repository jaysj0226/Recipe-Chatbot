# -*- coding: utf-8 -*-
"""Rewrite Node - Query Rewriting"""
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from config.settings import REWRITE_MODEL, USE_FAKE_LLM
from prompts.templates import REWRITE_PROMPT
from utils.allergy import detect_triggers, extract_allergens, build_constraint_text


# Rewrite Chain
rewrite_chain = REWRITE_PROMPT | ChatOpenAI(model=REWRITE_MODEL, temperature=0.5) | StrOutputParser()


def rewrite_node(query: str, recent_context: str = "") -> str:
    """
    Rewrite Node: 검색 최적화를 위한 쿼리 재작성
    
    Args:
        query: 원본 사용자 질문
        
    Returns:
        str: 재작성된 쿼리 (실패 시 원본 반환)
    """
    # Fake mode: just return the original (with optional constraints)
    if USE_FAKE_LLM:
        combined = (recent_context or "").strip()
        augment = ""
        if combined:
            if detect_triggers(query + "\n" + combined):
                allergens = extract_allergens(query + "\n" + combined)
                ctext = build_constraint_text(allergens)
                if ctext:
                    augment = f"\n\n{ctext}"
        elif detect_triggers(query):
            allergens = extract_allergens(query)
            ctext = build_constraint_text(allergens)
            if ctext:
                augment = f"\n\n{ctext}"
        return f"{query}{augment}".strip()

    try:
        # Augment with allergy/substitution constraints inferred from recent context
        combined = (recent_context or "").strip()
        augment = ""
        if combined:
            if detect_triggers(query + "\n" + combined):
                allergens = extract_allergens(query + "\n" + combined)
                ctext = build_constraint_text(allergens)
                if ctext:
                    augment = f"\n\n{ctext}"
        elif detect_triggers(query):
            allergens = extract_allergens(query)
            ctext = build_constraint_text(allergens)
            if ctext:
                augment = f"\n\n{ctext}"

        final_query = f"{query}{augment}"
        rewritten = rewrite_chain.invoke({"query": final_query}).strip()
        return rewritten if rewritten else query
    except Exception as e:
        from config.settings import DEBUG_RAW
        if DEBUG_RAW:
            print(f"rewrite_error: {e}")
        return query
