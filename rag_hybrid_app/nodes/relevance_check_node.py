from typing import Dict, Any, List

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# ──────────────────────────────────────────────────────────────
# 최신 LangGraph 호환: LLMChain 제거, prompt | llm 파이프라인 사용

def relevance_check_node(state) -> Dict[str, Any]:
    """문서 근거 대비 답변 타당성을 평가해 분기 키를 반환."""

    answer: str = state.answer
    docs: List[str] = state.retrieved_docs or []
    context = "\n\n".join(docs)

    prompt = PromptTemplate.from_template(
        "아래 답변이 문서 근거에 충분히 기반했는지 판단해 주세요.\n"
        "답변:\n{answer}\n\n"
        "문서:\n{context}\n\n"
        "결과를 'grounded', 'notGrounded', 'notSure' 중 하나로만 출력하세요."
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    judge_chain = prompt | llm | StrOutputParser()

    verdict: str = judge_chain.invoke({"answer": answer, "context": context}).strip()

    # LangGraph는 'branch' 키의 값을 조건 분기 키로 사용
    return {**state.dict(), "branch": verdict}
