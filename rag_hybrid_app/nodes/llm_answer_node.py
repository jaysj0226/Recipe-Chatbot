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
    "다음 문서들을 참고하여 질문에 답하세요.\n\n"
    "문서:\n{context}\n\n"
    "질문:\n{question}"
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
