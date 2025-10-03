import os, re, json, unicodedata
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

BASE_DIR   = Path(__file__).resolve().parent
STATIC_DIR = Path("C:/Users/SunjaeJeong/Desktop/data/files_data/recipe_chatbot_CRAG/static")      # test/static/
STATIC_DIR = BASE_DIR/"static"

load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / "rag_hybrid_app" / ".env")


# ── App / Settings ────────────────────────────────────────────────────────────
load_dotenv()
app = FastAPI(title="Group A — RAG with LLM Router (Enhanced)")

# 🔧 벡터 DB 설정
EMB = OpenAIEmbeddings(model="text-embedding-3-large")
PERSIST = "C:/Users/SunjaeJeong/Desktop/data/files_data/chroma_recipes_2025_09_16"
COLLECTION = "recipes-v1"

SCORE_THRESHOLD = float(os.environ.get("GROUPA_SCORE_THRESHOLD", "0.0"))
ALLOW_NO_CONTEXT_ANSWER = os.environ.get("ALLOW_NO_CONTEXT_ANSWER", "1") == "1"
ROUTER_MODEL = os.environ.get("GROUPA_ROUTER_MODEL", "gpt-4o-mini")
DEBUG_RAW = os.environ.get("GROUPA_DEBUG_RAW", "0") == "1"

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
        start = raw.find("{"); end = raw.rfind("}")
        data = json.loads(raw[start:end+1]) if (start != -1 and end != -1) else {}
    except Exception:
        data = {}
    intent = data.get("intent", "recipe")
    if intent not in ['recipe','dish_overview','storage','substitution','nutrition','equipment','shopping','unknown','out_of_domain']:
        intent = "recipe"
    needs_retrieval = bool(data.get("needs_retrieval", True))
    rewritten_query = data.get("rewritten_query") or query
    notes = data.get("notes", "")
    return {"intent": intent, "needs_retrieval": needs_retrieval, "rewritten_query": rewritten_query, "notes": notes}

# ── Intent별 프롬프트 (일반 지식 활용 허용) ──────────────────────────────────
COMMON_RULE = (
    "주의: 컨텍스트가 비어 있지 않으면 반드시 컨텍스트를 우선적으로 활용하라. "
    "컨텍스트가 있을 때는 최소 3줄 이상으로, 요리명과 단계 요약을 반드시 포함해 출력하라. "
    "컨텍스트가 없거나 부족하면 일반적인 요리 지식을 활용해 유용한 답변을 제공하라."
)

RECIPE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "역할: 레시피 도우미.\n"
     "제공된 컨텍스트가 있으면 우선 활용하되, 없거나 부족하면 일반 요리 지식을 활용해 답변.\n"
     "출력 형식: 1) 재료(계량 포함) 2) 단계(번호 목록) 3) 핵심 팁 4) 변형/대체 옵션(있으면).\n"
     "**영양 정보가 질문에 포함되었다면 일반적인 영양 정보도 간략히 추가.**\n" + COMMON_RULE),
    ("human","컨텍스트:\n{context}\n\n질문: {question}")
])

DISH_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "역할: 음식 소개 도우미. 요리의 개요(특징/풍미/난이도/소요시간)와 기본 재료, 대표 변형을 설명.\n"
     "컨텍스트가 있으면 우선 활용하고, 없으면 일반 지식으로 보완.\n" + COMMON_RULE),
    ("human","컨텍스트:\n{context}\n\n질문: {question}")
])

STORAGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "역할: 식재료/요리 보관 도우미. 보관 온도/용기/기간/해동/식품안전을 설명.\n"
     "컨텍스트가 있으면 우선 활용하고, 없으면 일반적인 식품 보관 지침 제공.\n" + COMMON_RULE),
    ("human","컨텍스트:\n{context}\n\n질문: {question}")
])

SUBSTITUTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "역할: 대체재 도우미. 풍미/기능 기준으로 대체안과 비율/주의점 제시.\n"
     "컨텍스트가 있으면 우선 활용하고, 없으면 일반적인 대체재 지식 활용.\n" + COMMON_RULE),
    ("human","컨텍스트:\n{context}\n\n질문: {question}")
])

NUTRITION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "역할: 영양 도우미. 대략적 열량/주요 영양성분/알레르겐/주의사항 제공.\n"
     "컨텍스트가 있으면 우선 활용하고, 없으면 일반적인 영양 정보 제공.\n" + COMMON_RULE),
    ("human","컨텍스트:\n{context}\n\n질문: {question}")
])

GENERAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "역할: 요리 Q&A 도우미. 질문 취지에 맞춰 실용적인 답변 제공.\n"
     "컨텍스트가 있으면 우선 활용하고, 없으면 일반 요리 지식 활용.\n" + COMMON_RULE),
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

# ── 컨텍스트 구성 함수 ────────────────────────────────────────────────────────
def _build_context_new(docs):
    """벡터 DB에서 컨텍스트 구성"""
    contexts = []
    seen = set()
    
    for doc in docs:
        content = getattr(doc, "page_content", "").strip()
        
        if content and len(content) > 100:
            metadata = getattr(doc, "metadata", {})
            url = metadata.get("url", "")
            
            if url and url in seen:
                continue
            if url:
                seen.add(url)
            
            formatted_content = _format_markdown_content(content)
            contexts.append(formatted_content)
    
    return contexts[:5]

def _format_markdown_content(content: str) -> str:
    """마크다운 형식을 한국어 친화적으로 변환"""
    content = re.sub(r'^# (.+)$', r'[제목] \1', content, flags=re.MULTILINE)
    content = re.sub(r'^## Ingredients$', '[재료]', content, flags=re.MULTILINE)
    content = re.sub(r'^## Steps$', '[조리법]', content, flags=re.MULTILINE)
    content = re.sub(r'^Source:.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^Image:.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content.strip()

# ── 검색 함수 ─────────────────────────────────────────────────────────────────
def _search_new(vs, query: str, k: int) -> List[Any]:
    """벡터 DB에서 검색"""
    try:
        docs = vs.similarity_search(query, k=k)
        if docs and len(docs) >= 2:
            return docs
        docs = vs.similarity_search(query, k=k*2)
        return docs
    except Exception as e:
        print(f"검색 오류: {e}")
        return []

# ── API Schemas ───────────────────────────────────────────────────────────────
class AskReq(BaseModel):
    query: str
    k: int = 10
    model: str = "gpt-4o-mini"

@app.get("/health")
def health():
    return {
        "ok": True,
        "persist": PERSIST,
        "collection": COLLECTION,
        "score_threshold": SCORE_THRESHOLD,
        "embed_model": "text-embedding-3-large",
        "total_docs": get_doc_count().get("total_docs", "N/A"),
        "router_model": ROUTER_MODEL,
        "allow_no_context_answer": ALLOW_NO_CONTEXT_ANSWER,
        "status": "일반 지식 활용 가능 (사용자 경험 최적화)"
    }

@app.get("/doc_count")
def get_doc_count():
    """문서 개수 확인"""
    try:
        chroma_db = Chroma(collection_name=COLLECTION, embedding_function=EMB, persist_directory=PERSIST)
        count = chroma_db._collection.count()
        return {"total_docs": count}
    except Exception as e:
        return {"error": str(e)}
    
# ── 메인 엔드포인트 (일반 지식 활용 버전) ─────────────────────────────────────
@app.post("/ask")
def ask(req: AskReq):
    # 1) Router로 의도 판별
    route = run_router(req.query)
    intent = route["intent"]
    needs_retrieval = route["needs_retrieval"]
    q_for_search = route["rewritten_query"]

    if intent == "out_of_domain":
        return {
            "answer": "죄송해요. 저는 음식·요리·레시피·재료·보관·영양 관련 질문에만 답합니다.",
            "retrieved_count": 0, "k": req.k, "sources": [], "router": route
        }

    # 2) 벡터 DB 검색
    vs = Chroma(collection_name=COLLECTION, embedding_function=EMB, persist_directory=PERSIST)
    docs, contexts = [], []
    
    if needs_retrieval:
        docs = _search_new(vs, q_for_search, req.k)
        contexts = _build_context_new(docs)

    # 3) 컨텍스트 준비 (없어도 진행 - LLM이 일반 지식 활용)
    context_text = "\n\n---\n\n".join(contexts)[:6000] if contexts else ""
    
    # 4) 의도별 프롬프트로 답변 생성 (컨텍스트 없어도 답변 가능)
    llm = ChatOpenAI(model=req.model, temperature=0.3)  # 약간 높여서 창의적 답변 가능
    prompt = PROMPT_BY_INTENT.get(intent, GENERAL_PROMPT)
    
    try:
        raw_ans = llm.invoke(prompt.format_messages(
            context=context_text or "컨텍스트가 제공되지 않았습니다. 일반적인 요리 지식을 활용해 답변해주세요.",
            question=req.query
        )).content
        ans = (raw_ans or "").strip()
    except Exception as e:
        ans = f"답변 생성 중 오류가 발생했습니다: {str(e)}"

    # 5) 개행 정리
    ans = re.sub(r"\n{3,}", "\n\n", ans)

    # 6) 응답 (컨텍스트 유무를 메타데이터로 표시)
    response = {
        "router": route,
        "intent": intent,
        "context_len": len(context_text),
        "used_docs": len(contexts),
        "context_found": bool(contexts),  # 컨텍스트 발견 여부
        "answer": ans,
        "retrieved_count": len(docs),
        "k": req.k,
        "sources": [getattr(d, "metadata", {}) for d in docs][:10],
        "mode": "context_based" if contexts else "general_knowledge"  # 답변 모드
    }
    
    if DEBUG_RAW:
        response["raw_answer"] = raw_ans
        response["context_preview"] = context_text[:300] if context_text else ""
        
    return response

# ── 테스트 엔드포인트 ─────────────────────────────────────────────────────────
@app.get("/test_search/{query}")
def test_search(query: str):
    """검색 테스트"""
    vs = Chroma(collection_name=COLLECTION, embedding_function=EMB, persist_directory=PERSIST)
    docs = _search_new(vs, query, 5)
    
    results = []
    for doc in docs:
        results.append({
            "title": doc.metadata.get("title", "N/A"),
            "url": doc.metadata.get("url", "N/A"),
            "content_length": len(doc.page_content),
            "content_preview": doc.page_content[:200]
        })
    
    return {
        "query": query,
        "found_docs": len(docs),
        "results": results
    }

# ── 정적 파일 서빙 ────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.post("/query")
def query(req: AskReq):
    """프론트엔드 호환용 엔드포인트"""
    return ask(req)

# ── 로컬 실행 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatbot_rag_oriented_version2:app", host="127.0.0.1", port=8000, reload=True)