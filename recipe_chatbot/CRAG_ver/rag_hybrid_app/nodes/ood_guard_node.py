
from typing import Dict, Any
import os, re
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 환경변수
OOD_MODEL = os.getenv("OOD_MODEL", "gpt-4o-mini")
OOD_TEMPERATURE = float(os.getenv("OOD_TEMPERATURE", "0"))

# 빠른 휴리스틱 (선택): 매우 명백한 키워드는 바로 in/out로 short-circuit
KW_IN  = ["요리","레시피","레시피명","조리","조리순서","만드는법","요리법","메뉴",
          "재료","양념","칼로리","영양","식단","다이어트","굽는","삶는","찌는","끓이는","볶는",
          "recipe","cook","cooking","ingredients","directions","steps","how to make","how to cook"]
KW_OUT = ["주가","비트코인","부동산","금리","환율","주식","코인","스마트폰","아이폰","삼성전자",
          "여행","항공권","호텔","게임","코딩","프로그래밍","야구","축구","농구","정치","선거"]

def _kw_gate(q: str):
    ql = q.lower()
    if any(k.lower() in ql for k in KW_OUT): return "out"
    if any(k.lower() in ql for k in KW_IN):  return "in"
    return None

# LLM 분류 프롬프트: in/out만 출력
_prompt = PromptTemplate.from_template(
    """당신은 질문이 '음식/요리/레시피/조리/재료/메뉴/영양(칼로리 포함)/식단' 주제에 속하는지 판별하는 분류기입니다.
다음 규칙을 따르세요.
- 속하면 'in', 아니면 'out'만 출력합니다 (따옴표·여분 텍스트 금지).
- 예시
  - '삼겹살 굽는 방법' → in
  - '계란찜 레시피' → in
  - '칼로리 계산법' → in
  - '아이폰 16 가격' → out
  - '비트코인 단타' → out
질문: {q}
정답:"""
)

_chain = _prompt | ChatOpenAI(model=OOD_MODEL, temperature=OOD_TEMPERATURE) | StrOutputParser()

def ood_guard_node(state) -> Dict[str, Any]:
    q = (state.query or "").strip()

    if not q:
        # 빈 질문은 out 처리하고 안내
        msg = "질문을 입력해 주세요. 요리·레시피·조리·재료·영양 주제만 지원합니다."
        return {**state.dict(), "answer": msg, "branch": "out"}

    # 1) 키워드 휴리스틱 (초고속 경로)
    hit = _kw_gate(q)
    if hit in ("in","out"):
        if hit == "out":
            msg = "죄송합니다. 저는 음식·요리·레시피·칼로리·영양 관련 질문만 답변합니다."
            return {**state.dict(), "answer": msg, "branch": "out"}
        return {**state.dict(), "branch": "in"}

    # 2) LLM 분류 (엄밀 경로)
    verdict = (_chain.invoke({"q": q}) or "").strip().lower()
    branch = "in" if verdict == "in" else "out"

    if branch == "out":
        msg = "죄송합니다. 저는 음식·요리·레시피·칼로리·영양 관련 질문만 답변합니다."
        return {**state.dict(), "answer": msg, "branch": "out"}

    return {**state.dict(), "branch": "in"}
