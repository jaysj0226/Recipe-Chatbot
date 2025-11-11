# -*- coding: utf-8 -*-
"""Session-based Conversation Memory Manager"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import uuid


class ConversationMemory:
    """간단한 세션 기반 메모리 관리자"""

    def __init__(self, max_history: int = 5, session_timeout: int = 30):
        """
        Args:
            max_history: 유지할 최근 턴 수(user+assistant 묶음 기준)
            session_timeout: 세션 만료 시간(분)
        """
        self.sessions: Dict[str, Dict] = {}
        self.max_history = max_history
        self.session_timeout = timedelta(minutes=session_timeout)

    def create_session(self) -> str:
        """새 세션 생성"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "history": [],
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "metadata": {},
        }
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """세션 조회"""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        # 만료 체크
        if datetime.now() - session["last_accessed"] > self.session_timeout:
            del self.sessions[session_id]
            return None

        return session

    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """메시지 추가"""
        session = self.get_session(session_id)
        if not session:
            return

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        session["history"].append(message)
        session["last_accessed"] = datetime.now()

        # 최근 턴만 유지 (user+assistant × max_history)
        if len(session["history"]) > self.max_history * 2:
            session["history"] = session["history"][-self.max_history * 2 :]

    def get_history(self, session_id: str, as_langchain: bool = False) -> List:
        """대화 이력 조회"""
        session = self.get_session(session_id)
        if not session:
            return []

        if as_langchain:
            from langchain_core.messages import HumanMessage, AIMessage

            messages = []
            for msg in session["history"]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            return messages

        return session["history"]

    def get_context_summary(self, session_id: str, max_turns: int = 3) -> str:
        """최근 n턴 요약 컨텍스트 생성"""
        session = self.get_session(session_id)
        if not session or not session["history"]:
            return ""

        recent_history = session["history"][-(max_turns * 2) :]

        context_parts = []
        for msg in recent_history:
            role_label = "사용자" if msg["role"] == "user" else "어시스턴트"
            context_parts.append(f"{role_label}: {msg['content']}")

        return "\n".join(context_parts)

    def clear_session(self, session_id: str):
        """세션 제거"""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def cleanup_expired_sessions(self):
        """만료 세션 정리"""
        now = datetime.now()
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session["last_accessed"] > self.session_timeout
        ]
        for sid in expired:
            del self.sessions[sid]

    def get_session_count(self) -> int:
        """활성 세션 수"""
        return len(self.sessions)

    def update_metadata(self, session_id: str, key: str, value):
        """세션 메타데이터 업데이트"""
        session = self.get_session(session_id)
        if session:
            session["metadata"][key] = value


# 전역 메모리 매니저 인스턴스
memory_manager = ConversationMemory(max_history=5, session_timeout=30)

