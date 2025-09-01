from typing import Dict, Any, List
import os
from dotenv import load_dotenv

from langchain_community.utilities import GoogleSerperAPIWrapper

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Google Serper 검색 노드 (LangGraph 호환)
#   • state.query 로 보조 웹 검색
#   • 검색 결과 문자열 리스트를 retrieved_docs 에 추가
#   • SERPER_API_KEY 가 없으면 검색 스킵
# -----------------------------------------------------------------------------

def web_search_node(state) -> Dict[str, Any]:
    query: str = state.query
    docs:  List[str] = state.retrieved_docs or []

    # 환경 변수 확인
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        # API 키 없으면 기존 문서만 유지
        return {**state.dict(), "retrieved_docs": docs}

    serper = GoogleSerperAPIWrapper(k=5, serper_api_key=api_key)
    try:
        results = serper.run(query)  # 문자열 리스트 반환
    except Exception as e:
        # 네트워크 오류 등 → 검색 스킵
        results = []

    return {
        **state.dict(),
        "retrieved_docs": docs + results,
    }