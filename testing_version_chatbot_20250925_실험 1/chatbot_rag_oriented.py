import os, re, json, unicodedata
from typing import Optional, List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ── App / Settings ────────────────────────────────────────────────────────────
load_dotenv()
app = FastAPI(title="Group A — RAG with LLM Router (Updated)")

# 🔧 새로운 벡터 DB 설정
EMB = OpenAIEmbeddings(model="text-embedding-3-large")  # 올바른 모델 지정
PERSIST = "C:/Users/SunjaeJeong/Desktop/data/files_data/chroma_recipes_2025_09_16"  # 새 DB 경로
COLLECTION = "recipes-v1"  # 새 컬렉션명

SCORE_THRESHOLD = float(os.environ.get("GROUPA_SCORE_THRESHOLD", "0.0"))   # 0 = always k
ALLOW_NO_CONTEXT_ANSWER = os.environ.get("ALLOW_NO_CONTEXT_ANSWER", "1") == "1"
ROUTER_MODEL = os.environ.get("GROUPA_ROUTER_MODEL", "gpt-4o-mini")
DEBUG_RAW = os.environ.get("GROUPA_DEBUG_RAW", "0") == "1"  # optional

# ── Router: 질의 의도 판별 + 재작성 ───────────────────────────────────────────
# 프롬프트 few shot 적용!!!
# 테스트 버전
'''
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 요리 도메인 라우터야. 아래 중 하나의 intent를 고르고 JSON만 출력해.\n"
     "가능한 intent: ['recipe','dish_overview','storage','substitution','nutrition','equipment','shopping','unknown','out_of_domain']\n"
     "fields: intent(str), needs_retrieval(bool), rewritten_query(str; 검색 시 유리하게 재작성), notes(str; 선택). "
     "요리/레시피/재료/보관/영양/도구/장보기면 out_of_domain이 아님."),
    ("human", "질문: {q}\n\nJSON으로만 답해.")
])
'''
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 요리 도메인 라우터야. 아래 중 하나의 intent를 고르고 JSON만 출력해.\n\n"
     "중요한 구분:\n"
     "- out_of_domain: 요리/음식과 전혀 관련 없는 질문 (날씨, 코딩, 주식, 정치, 연애, 게임 등)\n"
     "- unknown: 요리 관련이지만 애매한 질문 ('뭔가 매운 거', '간단한 요리 추천')\n\n"
     "가능한 intent: ['recipe','dish_overview','storage','substitution','nutrition','equipment','shopping','unknown','out_of_domain']\n\n"
     "예시:\n"
     "- '날씨 어때?' → out_of_domain\n"
     "- '파이썬 코딩' → out_of_domain\n" 
     "- '주식 투자' → out_of_domain\n"
     "- '김치찌개 만들기' → recipe\n"
     "- '뭔가 매운 거 추천' → unknown\n\n"
     "fields: intent(str), needs_retrieval(bool), rewritten_query(str; 검색 시 유리하게 재작성), notes(str; 선택)."),
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

    # 업데이트 부분: out_of_domain이면 무조건 retrieval 불필요 
    if intent == "out_of_domain":
        needs_retrieval = False
    return {"intent": intent, "needs_retrieval": needs_retrieval, "rewritten_query": rewritten_query, "notes": notes}

# ── Intent별 프롬프트 ─────────────────────────────────────────────────────────
COMMON_RULE = (
    "주의: 컨텍스트가 비어 있지 않으면 '근거 문서 없음'이라는 문구를 절대 출력하지 말고, "
    "반드시 컨텍스트에서 근거를 찾아 답하라. "
    "컨텍스트가 있을 때는 최소 3줄 이상으로, 요리명(또는 레시피명)과 단계 요약을 반드시 포함해 출력하라."
)

RECIPE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "역할: 레시피 도우미.\n"
     "가능하면 제공된 컨텍스트의 사실만 사용해 답하고, 없으면 일반 지침을 간결히 제시.\n"
     "출력 형식: 1) 재료(계량 포함) 2) 단계(번호 목록) 3) 핵심 팁 4) 변형/대체 옵션(있으면).\n"
     "컨텍스트 없을 때는 '근거 문서 없음'이라고 한 줄로 명시.\n" + COMMON_RULE),
    ("human","컨텍스트:\n{context}\n\n질문: {question}")
])

DISH_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "역할: 음식 소개 도우미. 요리의 개요(특징/풍미/난이도/소요시간)와 기본 재료, 대표 변형을 간결히.\n"
     "컨텍스트 우선, 없으면 일반 지침 + '근거 문서 없음' 표시.\n" + COMMON_RULE),
    ("human","컨텍스트:\n{context}\n\n질문: {question}")
])

STORAGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "역할: 식재료/요리 보관 도우미. 보관 온도/용기/기간/해동/식품안전(위험요소)을 단계별로.\n"
     "컨텍스트 우선, 없으면 일반 지침 + '근거 문서 없음' 표시.\n" + COMMON_RULE),
    ("human","컨텍스트:\n{context}\n\n질문: {question}")
])

SUBSTITUTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "역할: 대체재 도우미. 풍미/기능(점도, 결착, 팽창 등) 기준으로 1→N 대체안을 표기하고, 비율/주의점을 명시.\n"
     "컨텍스트 우선, 없으면 일반 지침 + '근거 문서 없음' 표시.\n" + COMMON_RULE),
    ("human","컨텍스트:\n{context}\n\n질문: {question}")
])

NUTRITION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "역할: 영양 도우미. 대략적 열량/주요 영양성분/알레르겐/주의대상(저나트륨 등)을 간결히.\n"
     "컨텍스트 우선, 없으면 일반 지침 + '근거 문서 없음' 표시.\n" + COMMON_RULE),
    ("human","컨텍스트:\n{context}\n\n질문: {question}")
])

GENERAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "역할: 요리 Q&A 도우미. 질문 취지에 맞춰 간략한 체크리스트/팁으로 답변.\n"
     "컨텍스트 우선, 없으면 일반 지침 + '근거 문서 없음' 표시.\n" + COMMON_RULE),
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

# ── 🔧 새로운 컨텍스트 구성 함수 (CSV 의존성 제거) ─────────────────────────
def _build_context_new(docs):
    """새로운 벡터 DB에서 컨텍스트 구성 (CSV 불필요)"""
    contexts = []
    seen = set()
    
    for doc in docs:
        # 새 벡터 DB는 page_content가 완전하므로 직접 사용
        content = getattr(doc, "page_content", "").strip()
        
        if content and len(content) > 100:  # 충분한 내용이 있는 경우
            # 중복 제거 (URL 기반)
            metadata = getattr(doc, "metadata", {})
            url = metadata.get("url", "")
            
            if url and url in seen:
                continue
            if url:
                seen.add(url)
            
            # 마크다운 형식을 더 읽기 쉽게 변환
            formatted_content = _format_markdown_content(content)
            contexts.append(formatted_content)
    
    return contexts[:5]  # 상위 5개 문서만 사용

def _format_markdown_content(content: str) -> str:
    """마크다운 형식을 한국어 친화적으로 변환"""
    # # 제목 → [제목] 형식으로 변환
    content = re.sub(r'^# (.+)$', r'[제목] \1', content, flags=re.MULTILINE)
    
    # ## Ingredients → [재료] 형식으로 변환
    content = re.sub(r'^## Ingredients$', '[재료]', content, flags=re.MULTILINE)
    
    # ## Steps → [조리법] 형식으로 변환
    content = re.sub(r'^## Steps$', '[조리법]', content, flags=re.MULTILINE)
    
    # 불필요한 Source/Image 라인 제거
    content = re.sub(r'^Source:.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^Image:.*$', '', content, flags=re.MULTILINE)
    
    # 연속된 빈 줄 정리
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content.strip()

# ── 검색 함수 (간소화) ────────────────────────────────────────────────────────
def _search_new(vs, query: str, k: int) -> List[Any]:
    """새로운 벡터 DB에서 검색 (간소화된 전략)"""
    try:
        # 기본 유사도 검색
        docs = vs.similarity_search(query, k=k)
        if docs and len(docs) >= 2:
            return docs
        
        # 백업: 더 많은 결과로 재시도
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
'''
@app.get("/health")
def health():
    return {
        "ok": True,
        "persist": PERSIST,
        "collection": COLLECTION,
        "score_threshold": SCORE_THRESHOLD,
        "embed_model": "text-embedding-3-large",
        "router_model": ROUTER_MODEL,
        "allow_no_context_answer": ALLOW_NO_CONTEXT_ANSWER,
        "total_docs": Chroma(collection_name=COLLECTION, embedding_function=EMB, persist_directory=PERSIST).count(),
        "status": "새 벡터 DB 적용됨"
    }
'''
# 방법 3: 별도의 카운트 엔드포인트 만들기
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
        "status": "새 벡터 DB 적용됨"
    }

@app.get("/doc_count")
def get_doc_count():
    """별도로 문서 개수를 확인하고 싶을 때 사용"""
    try:
        chroma_db = Chroma(collection_name=COLLECTION, embedding_function=EMB, persist_directory=PERSIST)
        count = chroma_db._collection.count()
        return {"total_docs": count}
    except Exception as e:
        return {"error": str(e)}
    
# ── 🔧 업데이트된 메인 엔드포인트 ──────────────────────────────────────────
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

    # 2) 새로운 벡터 DB에서 검색
    vs = Chroma(collection_name=COLLECTION, embedding_function=EMB, persist_directory=PERSIST)
    docs, contexts = [], []
    
    if needs_retrieval:
        docs = _search_new(vs, q_for_search, req.k)
        contexts = _build_context_new(docs)

    # 3) 컨텍스트가 없는데 무근거 허용 X → 거절
    if not contexts and not ALLOW_NO_CONTEXT_ANSWER:
        return {
            "answer": "관련된 레시피 정보를 찾을 수 없습니다. 더 구체적인 요리명이나 재료를 포함해서 질문해 주세요.",
            "retrieved_count": 0, "k": req.k, "sources": [], "router": route
        }

    # 4) 의도별 프롬프트로 생성
    context_text = "\n\n---\n\n".join(contexts)[:6000]
    llm = ChatOpenAI(model=req.model, temperature=0.2)
    prompt = PROMPT_BY_INTENT.get(intent, GENERAL_PROMPT)
    
    try:
        raw_ans = llm.invoke(prompt.format_messages(context=context_text, question=req.query)).content
        ans = (raw_ans or "").strip()
    except Exception as e:
        ans = f"답변 생성 중 오류가 발생했습니다: {str(e)}"

    # 5) 답변 후처리
    if context_text.strip() and ans:
        # '근거 문서 없음' 제거 (컨텍스트가 있을 때)
        cleaned = re.sub(r"근거\s*문서\s*없음", "", ans).strip()
        if cleaned:
            ans = cleaned

    # 6) 백업 답변 생성 (컨텍스트는 있지만 답변이 비어있는 경우)
    if context_text.strip() and (not ans or len(ans.strip()) < 50):
        ans = _generate_fallback_answer(context_text, req.query)

    # 7) 개행 정리
    ans = re.sub(r"\n{3,}", "\n\n", ans)

    # 8) 응답
    response = {
        "router": route,
        "intent": intent,
        "context_len": len(context_text),
        "used_docs": len(contexts),
        "context_preview": context_text[:300] if context_text else "",
        "answer": ans,
        "retrieved_count": len(docs),
        "k": req.k,
        "sources": [getattr(d, "metadata", {}) for d in docs][:10],
        "persist": PERSIST,
        "collection": COLLECTION,
        "db_version": "new_vectordb_2025_09_16"
    }
    
    if DEBUG_RAW:
        response["raw_answer"] = raw_ans
        
    return response

def _generate_fallback_answer(context_text: str, query: str) -> str:
    """컨텍스트는 있지만 답변이 부족한 경우 백업 답변 생성"""
    # 제목 추출
    title_match = re.search(r'\[제목\]\s*(.+)', context_text)
    title = title_match.group(1).strip() if title_match else "레시피"
    
    # 재료 추출
    ingredients_match = re.search(r'\[재료\]\s*(.*?)(?:\[|$)', context_text, re.DOTALL)
    ingredients = ingredients_match.group(1).strip()[:200] if ingredients_match else ""
    
    # 조리법 추출  
    steps_match = re.search(r'\[조리법\]\s*(.*?)(?:\[|$)', context_text, re.DOTALL)
    steps = steps_match.group(1).strip()[:300] if steps_match else ""
    
    parts = [f"**{title}**"]
    
    if ingredients:
        parts.append(f"\n**재료:**\n{ingredients}")
    
    if steps:
        parts.append(f"\n**조리법:**\n{steps}")
    
    if not ingredients and not steps:
        parts.append("\n제공된 정보를 바탕으로 레시피 정보를 구성했습니다.")
    
    return "\n".join(parts)

# ── 테스트 엔드포인트 ─────────────────────────────────────────────────────────
@app.get("/test_search/{query}")
def test_search(query: str):
    """검색 테스트용 엔드포인트"""
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)