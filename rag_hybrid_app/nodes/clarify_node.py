
import os
from typing import Dict, Any

CLARIFY_MODE = os.environ.get("CLARIFY_MODE", "ask")  # "ask" | "auto"

def clarify_question_node(state) -> Dict[str, Any]:
    """
    NotSure & 필터 결과가 비었을 때 호출.
    - CLARIFY_MODE="ask": 사용자에게 재질문 문구를 담아 종료 경로로.
    - CLARIFY_MODE="auto": 재작성 경로로 넘김(실제 재작성은 rewrite_node가 수행).
    """
    mode = (CLARIFY_MODE or "ask").lower()
    if mode == "auto":
        # 재작성 경로로 라우팅
        msg = "질문이 모호하여 자동으로 재작성 후 재검색합니다."
        return {**state.dict(), "clarify_next": "rewrite", "clarify_message": msg}
    else:
        # 사용자에게 명확 재질문 유도
        examples = [
            "계란 두 개로 만들 수 있는 아침 메뉴 레시피 알려줘",
            "닭가슴살 에어프라이어 조리 방법과 시간",
            "비건 크림 없이 파스타 레시피",
        ]
        tip = "질문에 '메뉴/레시피/조리/조리순서/방법/요리' 같은 단어를 포함해 주세요."
        msg = "질문이 애매해요. " + tip + " 예) " + " | ".join(examples)
        return {**state.dict(), "clarify_next": "ask", "answer": msg}

def clarify_router(state) -> str:
    nxt = getattr(state, "clarify_next", "ask")
    return "rewrite" if nxt == "rewrite" else "ask"
