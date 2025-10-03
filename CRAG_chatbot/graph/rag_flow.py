"""LangGraph State Machine for Hybrid RAG 앱 (최신 버전).

• LangGraph ≥0.0.39 호환
• 모든 노드는 최신 스타일 함수(Runnable)로 구성
• 조건 분기 라우터 함수 사용
"""

from typing import List, Optional
import logging

from pydantic import BaseModel
from langgraph.graph import StateGraph, END

# ──────────────────────────────── 노드 import ────────────────────────────────
from rag_hybrid_app.nodes.retrieve_node import retrieve_node
from rag_hybrid_app.nodes.llm_answer_node import llm_answer_node
from rag_hybrid_app.nodes.relevance_check_node import relevance_check_node
from rag_hybrid_app.nodes.rewrite_node import rewrite_query_node
from rag_hybrid_app.nodes.web_search_node import web_search_node
from rag_hybrid_app.nodes.filter_node import filter_low_similarity_node
from rag_hybrid_app.nodes.ood_guard_node import ood_guard_node

# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# 1) 상태 스키마
class RAGState(BaseModel):
    """그래프 전역 상태."""
    query: str
    retrieved_docs: Optional[List[str]] = None
    retrieved_scores: Optional[List[float]] = None
    answer: Optional[str] = None
    branch: Optional[str] = None
    clarify_next: Optional[str] = None
    clarify_message: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


# 2) 라우터 함수 정의
def route_ood_guard(state) -> str:
    """OOD Guard 분기: in/out"""
    branch = getattr(state, "branch", "out")
    logger.info(f"[OOD Guard] branch='{branch}'")
    
    if branch == "in":
        return "retrieve"
    return END


def route_relevance_check_1(state) -> str:
    """첫 번째 관련성 검증 분기"""
    branch = getattr(state, "branch", "notSure")
    logger.info(f"[Relevance Check 1] branch='{branch}'")
    
    # grounded면 바로 종료
    if branch == "grounded":
        logger.info("✅ Answer is grounded, ending workflow")
        return END
    
    # notGrounded 또는 notSure → 재작성
    logger.info("⚠️ Answer needs improvement, going to rewrite")
    return "rewrite_query"


def route_relevance_check_2(state) -> str:
    """두 번째 관련성 검증 분기 - 모두 종료"""
    branch = getattr(state, "branch", "notSure")
    logger.info(f"[Relevance Check 2] branch='{branch}', ending workflow")
    return END


# 3) 그래프 초기화
graph = StateGraph(RAGState)

# 4) 노드 등록
graph.add_node("ood_guard",         ood_guard_node)
graph.add_node("retrieve",          retrieve_node)
graph.add_node("filter",            filter_low_similarity_node)
graph.add_node("llm_answer_1",      llm_answer_node)
graph.add_node("relevance_check_1", relevance_check_node)
graph.add_node("rewrite_query",     rewrite_query_node)
graph.add_node("web_search",        web_search_node)
graph.add_node("llm_answer_2",      llm_answer_node)
graph.add_node("relevance_check_2", relevance_check_node)

# 5) 진입점 설정
graph.set_entry_point("ood_guard")

# 6) OOD Guard 분기
graph.add_conditional_edges(
    "ood_guard",
    route_ood_guard
)

# 7) 기본 검색 → 필터 → 답변 → 검증 흐름
graph.add_edge("retrieve", "filter")
graph.add_edge("filter", "llm_answer_1")
graph.add_edge("llm_answer_1", "relevance_check_1")

# 8) 첫 번째 관련성 검증 분기
graph.add_conditional_edges(
    "relevance_check_1",
    route_relevance_check_1
)

# 9) 재작성 후 웹 검색 → 재답변 → 재검증
graph.add_edge("rewrite_query", "web_search")
graph.add_edge("web_search", "llm_answer_2")
graph.add_edge("llm_answer_2", "relevance_check_2")

# 10) 최종 관련성 검증 분기
graph.add_conditional_edges(
    "relevance_check_2",
    route_relevance_check_2
)

# 11) 그래프 컴파일
rag_flow = graph.compile()

# 12) 디버깅용 - 그래프 구조 출력
if __name__ == "__main__":
    print("✅ RAG Flow compiled successfully!")
    print(f"Nodes: {list(graph.nodes.keys())}")