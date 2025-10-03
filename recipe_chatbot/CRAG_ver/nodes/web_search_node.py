from typing import Dict, Any, List
import os
from dotenv import load_dotenv

from langchain_community.utilities import GoogleSerperAPIWrapper

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Google Serper 검색 노드 (LangGraph 호환)
#   • state.query 로 보조 웹 검색
#   • 검색 결과를 retrieved_docs 에 추가
#   • SERPER_API_KEY 가 없으면 검색 스킵
# -----------------------------------------------------------------------------

def web_search_node(state) -> Dict[str, Any]:
    """웹 검색으로 추가 문서를 가져와 retrieved_docs에 추가"""
    query: str = state.query
    docs: List[str] = state.retrieved_docs or []

    # 환경 변수 확인
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        # API 키 없으면 기존 문서만 유지
        return {**state.dict(), "retrieved_docs": docs}

    serper = GoogleSerperAPIWrapper(k=5, serper_api_key=api_key)
    
    try:
        # results() 메서드로 구조화된 결과 가져오기
        search_results = serper.results(query)
        
        # organic 검색 결과에서 snippet 추출
        web_docs = []
        if search_results and "organic" in search_results:
            for result in search_results["organic"]:
                # snippet (요약)이 있으면 추가
                if "snippet" in result:
                    web_docs.append(result["snippet"])
                # 또는 title + snippet 결합
                # title = result.get("title", "")
                # snippet = result.get("snippet", "")
                # web_docs.append(f"{title}\n{snippet}")
        
        # 결과가 없으면 run() 메서드로 대체 (요약 텍스트)
        if not web_docs:
            result_str = serper.run(query)
            if result_str and isinstance(result_str, str):
                web_docs = [result_str]
            
    except Exception as e:
        # 네트워크 오류 등 → 검색 스킵
        web_docs = []

    return {
        **state.dict(),
        "retrieved_docs": docs + web_docs,  # ✅ 리스트 + 리스트
    }