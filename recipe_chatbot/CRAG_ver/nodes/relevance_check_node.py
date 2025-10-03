from typing import Dict, Any, List
import logging

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# 최신 LangGraph 호환: LLMChain 제거, prompt | llm 파이프라인 사용

def relevance_check_node(state) -> Dict[str, Any]:
    """문서 근거 대비 답변 타당성을 평가해 분기 키를 반환."""

    answer: str = state.answer
    docs: List[str] = state.retrieved_docs or []
    context = "\n\n".join(docs)

    # 문서가 없으면 즉시 notSure 반환
    if not docs or not context.strip():
        logger.warning("No documents to check relevance")
        return {**state.dict(), "branch": "notSure"}

    prompt = PromptTemplate.from_template(
        """당신은 답변의 신뢰성을 평가하는 전문가입니다.

아래 답변이 제공된 문서 근거에 충분히 기반했는지 판단해 주세요.

답변:
{answer}

문서:
{context}

평가 기준:
- grounded: 답변의 주요 내용이 문서에서 직접 추출되었고, 정확함
- notGrounded: 답변이 문서와 관련 없거나, 문서에 없는 내용을 포함함
- notSure: 문서가 불충분하거나 애매모호하여 판단하기 어려움

결과를 'grounded', 'notGrounded', 'notSure' 중 정확히 하나만 출력하세요.
다른 설명은 절대 추가하지 마세요."""
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # mini로 변경 (비용/속도)
    judge_chain = prompt | llm | StrOutputParser()

    try:
        verdict: str = judge_chain.invoke({"answer": answer, "context": context}).strip().lower()
        
        # 로깅
        logger.info(f"Relevance check verdict: '{verdict}'")
        
        # 정규화: 유효한 값만 허용
        if verdict not in ["grounded", "notgrounded", "notsure"]:
            logger.warning(f"Invalid verdict '{verdict}', defaulting to 'notSure'")
            verdict = "notsure"
        
        # LangGraph 분기용 정확한 케이스 매칭
        branch_map = {
            "grounded": "grounded",
            "notgrounded": "notGrounded",
            "notsure": "notSure"
        }
        
        branch = branch_map.get(verdict, "notSure")
        logger.info(f"Final branch: '{branch}'")
        
    except Exception as e:
        logger.error(f"Error in relevance check: {e}")
        branch = "notSure"

    return {**state.dict(), "branch": branch}