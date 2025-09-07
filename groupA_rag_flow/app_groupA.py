# main_fastapi_groupA_router.py
import os, re, json, unicodedata
from typing import Optional, List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ── CSV-backed context builder (optional) ─────────────────────────────────────
import pandas as _pd, os as _os

_CSV_PATH = _os.environ.get("GROUPA_CSV", "C:/Users/SunjaeJeong/Desktop/data/10000recipe_dataset.csv")
_DF = None
if _CSV_PATH and _os.path.exists(_CSV_PATH):
    try:
        _DF = _pd.read_csv(_CSV_PATH, low_memory=False)
        _DF.columns = [str(c).strip() for c in _DF.columns]
    except Exception:
        _DF = None

def _row_text_from_csv(recipe_id_value):
    """Return a composed text from CSV for given recipe_id (as str), or ''."""
    if _DF is None:
        return ""
    rid = str(recipe_id_value)
    col_candidates = {
        "title": ["title","Title","name","menu","CKG_NM","레시피명","메뉴","메뉴명","레시피"],
        "ingredients": ["ingredients_text","ingredients","CKG_MTRL_CN","재료"],
        "steps": ["steps","directions","instruction","방법","요리순서","조리순서","조리"],
        "description": ["description","intro","설명","summary","요약","CKG_IPDC"]
    }
    row = None
    if "recipe_id" in _DF.columns:
        hit = _DF[_DF["recipe_id"].astype(str) == rid]
        if len(hit) > 0:
            row = hit.iloc[0]
    if row is None:
        for idcol in ["RCP_SNO","id","ID"]:
            if idcol in _DF.columns:
                hit = _DF[_DF[idcol].astype(str) == rid]
                if len(hit) > 0:
                    row = hit.iloc[0]; break
    if row is None:
        return ""
    parts = []
    for key, cands in col_candidates.items():
        for c in cands:
            if c in _DF.columns:
                val = str(row.get(c, "")).strip()
                if val and val.lower() != "nan":
                    val = re.sub(r"\s+", " ", val)
                    parts.append(f"{key.upper()}: {val}")
                    break
    return " | ".join(parts)

# ── App / Settings ────────────────────────────────────────────────────────────
load_dotenv()
app = FastAPI(title="Group A — RAG with LLM Router")

EMB = OpenAIEmbeddings(model=os.environ.get("GROUPA_EMBED_MODEL","text-embedding-3-small"))
PERSIST = os.environ.get("GROUPA_PERSIST", "./chroma_rag_hybrid_db")
COLLECTION = os.environ.get("GROUPA_COLLECTION", "recipe_hybrid_rag")
SCORE_THRESHOLD = float(os.environ.get("GROUPA_SCORE_THRESHOLD", "0.0"))   # 0 = always k
ALLOW_NO_CONTEXT_ANSWER = os.environ.get("ALLOW_NO_CONTEXT_ANSWER", "1") == "1"
ROUTER_MODEL = os.environ.get("GROUPA_ROUTER_MODEL", "gpt-4o-mini")

# ── Router: 질의 의도 판별 + 재작성 ───────────────────────────────────────────
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 요리 도메인 라우터야. 아래 중 하나의 intent를 고르고 JSON만 출력해.\n"
     "가능한 intent: ['recipe','dish_overview','storage','substitution','nutrition','equipment','shopping','unknown','out_of_domain']\n"
     "fields: intent(str), needs_retrieval(bool), rewritten_query(str; 검색 시 유리하게 재작성), notes(str; 선택). "
     "요리/레시피/재료/보관/영양/도구/장보기면 out_of_domain이 아님."),
    ("human", "질문: {q}\n\nJSON으로만 답해.")
])

def run_router(query: str) -> Dict[str, Any]:
    llm = ChatOpenAI(model=ROUTER_MODEL, temperature=0)
    try:
        raw = llm.invoke(ROUTER_PROMPT.format_messages(q=query)).content
        # JSON만 강제
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(raw[start:end+1])
        else:
            data = {}
    except Exception:
        data = {}
    # 기본값 보정
    intent = data.get("intent", "recipe")
    if intent not in ['recipe','dish_overview','storage','substitution','nutrition','equipment','shopping','unknown','out_of_domain']:
        intent = "recipe"
    needs_retrieval = bool(data.get("needs_retrieval", True))
    rewritten_query = data.get("rewritten_query") or query
    notes = data.get("notes", "")
    return {"intent": intent, "needs_retrieval": needs_retrieval, "rewritten_query": rewritten_query, "notes": notes}

# ── Intent별 프롬프트(컨텍스트가 있으면 우선 사용) ───────────────────────────
RECIPE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "역할: 레시피 도우미.\n"
     "가능하면 제공된 컨텍스트의 사실만 사용해 답하고, 없으면 일반 지침을 간결히 제시.\n"
     "출력 형식: 1) 재료(계량 포함) 2) 단계(번호 목록) 3) 핵심 팁 4) 변형/대체 옵션(있으면).\n"
     "컨텍스트 없을 때는 '근거 문서 없음'이라고 한 줄로 명시."),
    ("human","컨텍스트:\n{context}\n\n질문: {question}")
])

DISH_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "역할: 음식 소개 도우미. 요리의 개요(특징/풍미/난이도/소요시간)와 기본 재료, 대표 변형을 간결히.\n"
     "컨텍스트 우선, 없으면 일반 지침 + '근거 문서 없음' 표시."),
    ("human","컨텍스트:\n{context}\n\n질문: {question}")
])

STORAGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "역할: 식재료/요리 보관 도우미. 보관 온도/용기/기간/해동/식품안전(위험요소)을 단계별로.\n"
     "컨텍스트 우선, 없으면 일반 지침 + '근거 문서 없음' 표시."),
    ("human","컨텍스트:\n{context}\n\n질문: {question}")
])

SUBSTITUTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "역할: 대체재 도우미. 풍미/기능(점도, 결착, 팽창 등) 기준으로 1→N 대체안을 표기하고, 비율/주의점을 명시.\n"
     "컨텍스트 우선, 없으면 일반 지침 + '근거 문서 없음' 표시."),
    ("human","컨텍스트:\n{context}\n\n질문: {question}")
])

NUTRITION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "역할: 영양 도우미. 대략적 열량/주요 영양성분/알레르겐/주의대상(저나트륨 등)을 간결히.\n"
     "컨텍스트 우선, 없으면 일반 지침 + '근거 문서 없음' 표시."),
    ("human","컨텍스트:\n{context}\n\n질문: {question}")
])

GENERAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "역할: 요리 Q&A 도우미. 질문 취지에 맞춰 간략한 체크리스트/팁으로 답변.\n"
     "컨텍스트 우선, 없으면 일반 지침 + '근거 문서 없음' 표시."),
    ("human","컨텍스트:\n{context}\n\n질문: {question}")
])

PROMPT_BY_INTENT = {
    "recipe": RECIPE_PROMPT,
    "dish_overview": DISH_PROMPT,
    "storage": STORAGE_PROMPT,
    "substitution": SUBSTITUTION_PROMPT,
    "nutrition": NUTRITION_PROMPT,
    "equipment": GENERAL_PROMPT,
    "shopping": GENERAL_PROMPT,
    "unknown": GENERAL_PROMPT
}

# ── Retrieval helpers ─────────────────────────────────────────────────────────
def _build_context(docs):
    merged, seen = [], set()
    for d in docs:
        content = (getattr(d, "page_content", "") or "").strip()
        if not content:
            md = getattr(d, "metadata", {}) if isinstance(getattr(d, "metadata", {}), dict) else {}
            rid = md.get("recipe_id")
            if rid is not None:
                content = _row_text_from_csv(rid)
        # dedup key
        key = None
        md = getattr(d, "metadata", {}) if isinstance(getattr(d, "metadata", {}), dict) else {}
        key = md.get("recipe_id") or md.get("RCP_SNO") or md.get("id") or md.get("url") or id(d)
        if key in seen:
            continue
        seen.add(key)
        if content:
            merged.append(content)
    return merged

def _search(vs, q: str, k: int) -> List[Any]:
    # 기본 similarity → 필요 시 threshold 적용
    search_type = "similarity_score_threshold" if SCORE_THRESHOLD > 0 else "similarity"
    kwargs = {"k": k}
    if SCORE_THRESHOLD > 0:
        kwargs["score_threshold"] = SCORE_THRESHOLD
    retr = vs.as_retriever(search_type=search_type, search_kwargs=kwargs)
    docs = retr.get_relevant_documents(q)
    if docs:
        return docs
    # 보수적 완화: threshold 해제 및 MMR
    retr2 = vs.as_retriever(search_type="similarity", search_kwargs={"k": max(k, 6)})
    docs = retr2.get_relevant_documents(q)
    if docs:
        return docs
    retr3 = vs.as_retriever(search_type="mmr", search_kwargs={"k": max(k, 8), "fetch_k": max(20, k*5)})
    return retr3.get_relevant_documents(q)

# ── API Schemas ───────────────────────────────────────────────────────────────
class AskReq(BaseModel):
    query: str
    k: int = 4
    model: str = "gpt-4o-mini"

@app.get("/health")
def health():
    return {
        "ok": True,
        "persist": PERSIST,
        "collection": COLLECTION,
        "score_threshold": SCORE_THRESHOLD,
        "embed_model": os.environ.get("GROUPA_EMBED_MODEL","text-embedding-3-small"),
        "router_model": ROUTER_MODEL,
        "allow_no_context_answer": ALLOW_NO_CONTEXT_ANSWER,
    }

@app.post("/ask")
def ask(req: AskReq):
    # 1) LLM Router로 의도/전략 결정
    route = run_router(req.query)
    intent = route["intent"]
    needs_retrieval = route["needs_retrieval"]
    q_for_search = route["rewritten_query"]

    if intent == "out_of_domain":
        return {
            "answer": "죄송해요. 저는 음식·요리·레시피·재료·보관·영양 관련 질문에만 답합니다.",
            "retrieved_count": 0, "k": req.k, "sources": [], "router": route
        }

    # 2) 필요한 경우에만 검색
    vs = Chroma(collection_name=COLLECTION, embedding_function=EMB, persist_directory=PERSIST)
    docs, merged = [], []
    if needs_retrieval:
        docs = _search(vs, q_for_search, req.k)
        merged = _build_context(docs)

    # 3) 컨텍스트가 없는데 무근거 허용 X → 거절
    if not merged and not ALLOW_NO_CONTEXT_ANSWER:
        return {
            "answer": "제공된 문서가 없어 답변할 수 없습니다. 질문을 조금 더 구체화해 주세요.",
            "retrieved_count": 0, "k": req.k, "sources": [], "router": route
        }

    # 4) 의도별 프롬프트로 생성
    context_text = "\n\n".join(merged)[:6000]
    llm = ChatOpenAI(model=req.model, temperature=0.2)
    prompt = PROMPT_BY_INTENT.get(intent, GENERAL_PROMPT)
    ans = llm.invoke(prompt.format_messages(context=context_text, question=req.query)).content.strip()

    # 5) 응답
    return {
        "router": route,
        "intent": intent,
        "context_len": len(context_text),
        "used_docs": len(merged),
        "context_preview": context_text[:180],
        "answer": ans,
        "retrieved_count": len(docs),
        "k": req.k,
        "sources": [getattr(d, "metadata", {}) for d in docs][:10],
        "persist": PERSIST,
        "collection": COLLECTION,
    }
