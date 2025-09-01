
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# --- CSV-backed context builder (optional) ------------------------------------
# If GROUPA_CSV is provided and documents lack page_content, we fill from CSV using recipe_id.
import pandas as _pd, os as _os

_CSV_PATH = _os.environ.get("GROUPA_CSV", "C:/Users/SunjaeJeong/Desktop/data/10000recipe_dataset.csv")
_DF = None
if _CSV_PATH and _os.path.exists(_CSV_PATH):
    try:
        _DF = _pd.read_csv(_CSV_PATH, low_memory=False)
        # normalize columns
        _DF.columns = [str(c).strip() for c in _DF.columns]
    except Exception as _e:
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
    # Prefer recipe_id exact match if exists
    if "recipe_id" in _DF.columns:
        hit = _DF[_DF["recipe_id"].astype(str) == rid]
        if len(hit) > 0:
            row = hit.iloc[0]
    # Fallback: try other id-like columns
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
                    # lightweight whitespace cleanup
                    val = re.sub(r"\s+", " ", val)
                    parts.append(f"{key.upper()}: {val}")
                    break
    return " | ".join(parts)


load_dotenv()
app = FastAPI(title="Group A — Basic RAG")

EMB = OpenAIEmbeddings(model=os.environ.get("GROUPA_EMBED_MODEL","text-embedding-3-small"))
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 레시피 도우미입니다. 제공된 컨텍스트의 사실만 사용해 간결히 답하세요. "
     "컨텍스트가 비어있거나 근거가 부족하면 반드시 '제공된 문서가 없어 답변할 수 없습니다. 질문을 더 구체화해 주세요.'라고 답하세요."),
    ("human", "컨텍스트:\n{context}\n\n질문: {question}")
])

PERSIST = os.environ.get("GROUPA_PERSIST", "./chroma_rag_hybrid_db")
COLLECTION = os.environ.get("GROUPA_COLLECTION", "recipe_hybrid_rag")
SCORE_THRESHOLD = float(os.environ.get("GROUPA_SCORE_THRESHOLD", "0.0"))  # 0.0 = 항상 k개

class AskReq(BaseModel):
    query: str
    k: int = 4
    model: str = "gpt-4o-mini"

@app.get("/health")
def health():
    return {"ok": True, "persist": PERSIST, "collection": COLLECTION, "score_threshold": SCORE_THRESHOLD, "embed_model": os.environ.get("GROUPA_EMBED_MODEL","text-embedding-3-small")}

@app.post("/ask")
def ask(req: AskReq):
    vs = Chroma(collection_name=COLLECTION, embedding_function=EMB, persist_directory=PERSIST)
    # score threshold 검색 모드 지원
    search_type = "similarity_score_threshold" if SCORE_THRESHOLD > 0 else "similarity"
    search_kwargs = {"k": req.k}
    if SCORE_THRESHOLD > 0:
        search_kwargs["score_threshold"] = SCORE_THRESHOLD

    retr = vs.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    docs = retr.get_relevant_documents(req.query)

    # Build context with CSV fallback if page_content is empty
    merged = []
    seen_keys = set()
    for d in docs:
        content = (d.page_content or "").strip()
        if not content:
            # try CSV using recipe_id
            rid = d.metadata.get("recipe_id") if isinstance(d.metadata, dict) else None
            if rid is not None:
                content = _row_text_from_csv(rid)
        # key for dedup
        key = d.metadata.get("recipe_id") or d.metadata.get("url") or id(d)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if content:
            merged.append(content)

    if not docs or not merged:  # ✅ 빈 검색이거나 본문 구성 실패시 생성 금지
        fallback = "제공된 문서가 없어 답변할 수 없습니다. 질문을 더 구체화해 주세요."
        return {
            "answer": fallback,
            "retrieved_count": 0,
            "k": req.k,
            "sources": [],
            "persist": PERSIST,
            "collection": COLLECTION,
        }

    context = "\n\n".join(merged)[:4000]
    llm = ChatOpenAI(model=req.model, temperature=0.2)
    chain = PROMPT | llm
    ans = chain.invoke({"context": context, "question": req.query}).content.strip()
    return {
        "context_len": len(context),
        "used_docs": len(merged),
        "context_preview": context[:160],
        
        "answer": ans,
        "retrieved_count": len(docs),
        "k": req.k,
        "sources": [d.metadata for d in docs],
        "persist": PERSIST,
        "collection": COLLECTION,
    }
