"""Generate Node with Conversation Context"""
from __future__ import annotations

import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import (
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    ALLOW_NO_CONTEXT_ANSWER,
    USE_FAKE_LLM,
)
from prompts.templates import PROMPT_BY_INTENT, GENERAL_PROMPT
from utils.text_formatter import clean_newlines


# --- Heuristics ---------------------------------------------------------------

TARGET_CUES = [
    "레시피", "만드는 법", "만드는 방법", "칼로리", "영양", "요약", "무엇", "뭐야",
]


def extract_target_dish(query: str) -> str:
    """Extract a likely target dish/item in a simple, robust way."""
    if not query:
        return ""
    q = query.strip()
    for cue in TARGET_CUES:
        if cue in q:
            left = q.split(cue, 1)[0].strip()
            left = re.sub(r"[은는이가\s]+$", "", left)
            if len(left) >= 2:
                return left
    m = re.search(r"(.+?)(?:\?|\s*무엇|\s*뭐야)$", q)
    if m:
        cand = re.sub(r"[은는이가\s]+$", "", m.group(1).strip())
        if len(cand) >= 2:
            return cand
    return ""


# --- Generation ---------------------------------------------------------------

def generate_with_history(
    query: str,
    intent: str,
    context: str,
    conversation_history: list | None = None,
    model: str = GENERATION_MODEL,
) -> str:
    """Generate answer, optionally using conversation history."""
    # No-context refusal (clean Korean)
    if not context and not ALLOW_NO_CONTEXT_ANSWER:
        return (
            "컨텍스트(벡터 DB)에서 해당 내용을 찾지 못했어요.\n"
            "출처가 있는 신뢰도 높은 답변을 위해 아래 중 하나를 시도해 주세요:\n"
            "- 검색 기능이 활성화된 상태로 다시 질문,\n"
            "- '추론 허용' 옵션을 명시해 일반 지식 기반 답변 허용"
        )

    # Fake/deterministic mode for tests/CI
    if USE_FAKE_LLM:
        if not context:
            return (
                "질문을 확인했습니다. 현재 컨텍스트가 없어 일반 지식을 기반으로 간단히 안내할게요.\n"
                f"질문: {query}"
            )
        snippet = context.strip().split("\n", 1)[0][:180]
        return f"요약 기반 안내: {snippet} ..."

    llm = ChatOpenAI(model=model, temperature=GENERATION_TEMPERATURE)
    prompt_template = PROMPT_BY_INTENT.get(intent, GENERAL_PROMPT)

    try:
        if not context:
            context = "컨텍스트가 비어 있으므로 보편적인 요리 지식으로 보완합니다."

        formatted_messages = prompt_template.format_messages(context=context, question=query)
        system_content = formatted_messages[0].content
        # ✅ FIX: Use the formatted human message that includes context, not just the query
        human_content = formatted_messages[1].content

        messages = [SystemMessage(content=system_content)]
        if conversation_history:
            # recent 3 turns (user+assistant × 3) = 6 messages
            messages.extend(conversation_history[-6:])
        messages.append(HumanMessage(content=human_content))

        raw_ans = llm.invoke(messages).content
        ans = clean_newlines((raw_ans or "").strip())
        return ans
    except Exception as e:
        from config.settings import DEBUG_RAW
        if DEBUG_RAW:
            print(f"generate_with_history_error: {e}")
        return f"응답 생성 중 오류가 발생했습니다: {str(e)}"


def generate_node(query: str, intent: str, context: str, model: str = GENERATION_MODEL) -> str:
    return generate_with_history(query, intent, context, None, model)