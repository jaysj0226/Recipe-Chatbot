"""Pydantic Schemas for API."""

from typing import Optional, Literal, Union, List, Dict

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """사용자 질의 요청 스키마."""

    query: str = Field(..., description="사용자 질문")
    # 검색 문서 개수 (faithfulness 향상을 위해 기본 8개)
    k: int = Field(default=8, ge=1, le=50, description="검색할 문서 개수")
    model: str = Field(default="gpt-4o", description="사용 LLM 모델")
    enable_rewrite: bool = Field(default=True, description="쿼리 보정(리라이트) 사용 여부")
    allow_low_confidence: bool = Field(
        default=False,
        description="근거가 약한 경우 경고 후 진행을 허용할지 여부",
    )
    decision: Optional[str] = Field(
        default=None,
        description="신뢰도 낮음 상태에서의 사용자 결정: proceed|clarify",
    )
    session_id: Optional[str] = Field(default=None, description="세션 ID (대화 유지용)")

    # Image controls (request-level)
    include_images: bool = Field(default=True, description="이미지 포함 여부")
    image_policy: Literal["strict", "lenient", "always"] = Field(
        default="lenient", description="이미지 출력 정책"
    )
    max_images: int = Field(default=5, ge=0, le=12, description="최대 이미지 개수")


class HealthResponse(BaseModel):
    """Health check 응답 스키마."""

    ok: bool
    persist: str
    collection: str
    score_threshold: float
    embed_model: str
    total_docs: Union[int, str]
    router_model: str
    judge_model: str
    allow_no_context_answer: bool
    enable_crag: bool
    architecture: str
    status: str
    # Optional diagnostics
    fake_mode: Optional[bool] = None
    ce_rerank_enabled: Optional[bool] = None
    ce_model: Optional[str] = None
    similarity_threshold: Optional[float] = None
    domain_cap: Optional[int] = None
    lowconf_mode: Optional[str] = None


class AskResponse(BaseModel):
    """질의 응답 스키마 (참고용)."""

    answer: str
    router: Dict
    intent: str
    original_query: str
    rewritten_query: Optional[str]
    context_len: int
    used_docs: int
    context_found: bool
    retrieved_count: int
    retrieved_scores: List
    k: int
    mode: str
    branch: str
    pipeline: List
    session_id: Optional[str] = None
    history_used: bool = False
    conversation_turns: int = 0

    # Optional diagnostics
    low_confidence: Optional[bool] = None
    warning: Optional[str] = None
    decision_required: Optional[bool] = None
    suggested_actions: Optional[List[str]] = None

