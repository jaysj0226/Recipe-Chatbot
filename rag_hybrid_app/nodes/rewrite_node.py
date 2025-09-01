from typing import Dict, Any
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI  # 최신 패키지
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ──────────────────────────────────────────────────────────────
# LLM 초기화
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY"),  # 명시적 키 전달
)

# 프롬프트 템플릿
prompt = PromptTemplate.from_template(
    """당신은 사용자의 질문을 명확하게 다시 작성하는 전문가입니다.\n\n"
    "원본 질문:\n{query}\n\n"
    "문서 검색이 실패했기 때문에, 이 질문을 더 명확하고 구체적으로 다시 작성해주세요.\n"
    "예상되는 재료, 상황, 목적 등을 고려하여 검색 가능한 형태로 바꿔주세요.\n\n"
    "다시 쓴 질문:"""
)

# Runnable 체인 (LLMChain 대체)
rewrite_chain = prompt | llm | StrOutputParser()

def rewrite_query_node(state) -> Dict[str, Any]:
    """state.query를 더 구체적인 검색용 쿼리로 재작성"""
    new_query: str = rewrite_chain.invoke({"query": state.query}).strip()
    return {**state.dict(), "query": new_query}
