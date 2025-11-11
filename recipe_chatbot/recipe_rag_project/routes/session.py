from fastapi import APIRouter
from utils.conversation_memory import memory_manager


router = APIRouter()


@router.post("/session/new")
def create_new_session():
    session_id = memory_manager.create_session()
    return {
        "session_id": session_id,
        "message": "세션이 생성되었어요",
    }


@router.get("/session/{session_id}/history")
def get_session_history(session_id: str):
    history = memory_manager.get_history(session_id)
    if not history:
        return {
            "session_id": session_id,
            "history": [],
            "message": "세션을 찾을 수 없거나 만료되었습니다",
        }

    return {
        "session_id": session_id,
        "history": history,
        "total_turns": len(history) // 2,
    }


@router.delete("/session/{session_id}")
def clear_session(session_id: str):
    memory_manager.clear_session(session_id)
    return {
        "session_id": session_id,
        "message": "세션이 초기화되었습니다",
    }


@router.get("/sessions/count")
def get_active_sessions():
    memory_manager.cleanup_expired_sessions()

    return {
        "active_sessions": memory_manager.get_session_count(),
        "timeout_minutes": 30,
    }

