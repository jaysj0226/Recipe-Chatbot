"""LangGraph State Machine for Hybrid RAG 앱 (최신 버전).

• LangGraph ≥0.0.39 호환
• 모든 노드는 최신 스타일 함수(Runnable)로 구성
• 조건 분기에 문자열 대신 RunnableLambda("next", state) 사용
"""

from typing import List, Optional

from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

# ──────────────────────────────── 노드 import ────────────────────────────────
from rag_hybrid_app.nodes.retrieve_node import retrieve_node
from rag_hybrid_app.nodes.llm_answer_node import llm_answer_node
from rag_hybrid_app.nodes.relevance_check_node import relevance_check_node
from rag_hybrid_app.nodes.rewrite_node import rewrite_query_node
from rag_hybrid_app.nodes.web_search_node import web_search_node
# ─────────────────────────────────────────────────────────────────────────────

from langchain_core.runnables import RunnableLambda
end_runnable = RunnableLambda(lambda state: END)
rewrite_runnable = RunnableLambda(lambda state: ("rewrite_query", state))

# 1) 상태 스키마
class RAGState(BaseModel):
    """그래프 전역 상태."""

    query: str
    retrieved_docs: Optional[List[str]] = None
    answer: Optional[str] = None


# 2) 그래프 초기화
graph = StateGraph(RAGState)

# 3) 노드 등록
graph.add_node("retrieve",          retrieve_node)
graph.add_node("llm_answer_1",      llm_answer_node)
graph.add_node("relevance_check_1", relevance_check_node)
graph.add_node("rewrite_query",     rewrite_query_node)
graph.add_node("web_search",        web_search_node)
graph.add_node("llm_answer_2",      llm_answer_node)
graph.add_node("relevance_check_2", relevance_check_node)

# 4) 기본 흐름 설정
graph.set_entry_point("retrieve")
graph.add_edge("retrieve",     "llm_answer_1")
graph.add_edge("llm_answer_1", "relevance_check_1")

# 5) relevance_check_1 조건 분기
#    문자열 대신 (next_node_name, state) 튜플을 반환하는 Runnable 사용
rewrite_runnable = RunnableLambda(lambda state: ("rewrite_query", state))

graph.add_conditional_edges(
    "relevance_check_1",
    {"grounded": end_runnable,
     "notGrounded": rewrite_runnable,
     "notSure": rewrite_runnable}
)


# 6) rewrite 후 검색 → 재답변 → 재검증
graph.add_edge("rewrite_query", "web_search")
graph.add_edge("web_search",    "llm_answer_2")
graph.add_edge("llm_answer_2",  "relevance_check_2")

# 7) 최종 검증 분기 (모두 종료)
graph.add_conditional_edges(
    "relevance_check_2",
    {"grounded": end_runnable,
     "notGrounded": end_runnable,
     "notSure": end_runnable}
)
# 8) 그래프 컴파일
rag_flow = graph.compile()
