from typing import Dict, Any, List
import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI  # ✅ 최신 import (langchain-openai)

load_dotenv()

# ──────────────────────────────────────────────────────────────
# LLM 초기화 (gpt‑4o, 0 온도)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# 프롬프트 템플릿
prompt = PromptTemplate.from_template(
    """당신은 레시피 전문 AI 어시스턴트입니다.
    
다음 문서들을 참고하여 질문에 정확하고 자세하게 답변해주세요.
- 재료와 양을 명확히 표시
- 조리 단계를 순서대로 설명
- 조리 시간과 온도 등 구체적 정보 포함
- 문서에 없는 내용은 추측하지 말 것

문서:
{context}

질문: {question}

답변:"""
)
# prompt → llm → 문자열 파서 파이프라인
answer_chain = prompt | llm | StrOutputParser()


# ──────────────────────────────────────────────────────────────
def llm_answer_node(state) -> Dict[str, Any]:
    """검색 문서와 질문을 조합해 최종 답변을 생성."""

    query: str = state.query
    docs: List[str] = state.retrieved_docs or []
    context: str = "\n\n".join(docs)

    answer: str = answer_chain.invoke({"context": context, "question": query}).strip()

    return {**state.dict(), "answer": answer}
