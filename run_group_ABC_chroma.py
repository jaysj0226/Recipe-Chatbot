
"""
run_group_ABC_chroma.py — A/B/C experiment using your existing Chroma DB.
- Keyword gate (요리 의도 키워드) 적용
- Retrieval from Chroma (similarity + scores)
- Group A: answer anyway
- Group B: rewrite + re-retrieve if low
- Group C: refuse if low
- LLM judge (Faith/Rel/Hall 1~5)
"""
import os, re, json, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# ---------------- Config helpers ----------------
def env(name: str, default: str=""):
    v = os.environ.get(name, default)
    if v is None: v = default
    return v

PERSIST = env("EXP_PERSIST", "./chroma_rag_hybrid_db")
COLLECTION = env("EXP_COLLECTION", "recipe_hybrid_rag")
EMBED_MODEL = env("EXP_EMBED_MODEL", "text-embedding-3-small")
CSV_PATH = env("EXP_CSV", "")

# ---------------- CSV fallback ------------------
_DF = None
if CSV_PATH and os.path.exists(CSV_PATH):
    try:
        _DF = pd.read_csv(CSV_PATH, low_memory=False)
        _DF.columns = [str(c).strip() for c in _DF.columns]
    except Exception:
        _DF = None

def csv_row_text(recipe_id=None, url=None, title_hint=None):
    if _DF is None: return ""
    row = None
    if recipe_id is not None and "recipe_id" in _DF.columns:
        hit = _DF[_DF["recipe_id"].astype(str) == str(recipe_id)]
        if len(hit) > 0: row = hit.iloc[0]
    if row is None and url and "url" in _DF.columns:
        hit = _DF[_DF["url"].astype(str) == str(url)]
        if len(hit) > 0: row = hit.iloc[0]
    if row is None and title_hint and "title" in _DF.columns:
        t = str(title_hint).strip()
        if t:
            hit = _DF[_DF["title"].astype(str).str.contains(re.escape(t), na=False)]
            if len(hit) > 0: row = hit.iloc[0]
    if row is None: return ""

    def pick(*cols):
        for c in cols:
            if c in _DF.columns:
                v = str(row.get(c, "")).strip()
                if v and v.lower() != "nan": return v
        return ""

    title = pick("title","Title","name","menu","CKG_NM","레시피명")
    desc  = pick("description","intro","설명","summary","요약","CKG_IPDC")
    ing   = pick("ingredients","ingredients_text","CKG_MTRL_CN","재료")
    steps = pick("steps","directions","instruction","방법","요리순서")
    parts = []
    if title: parts.append(f"제목: {title}")
    if desc:  parts.append(f"설명: {desc}")
    if ing:   parts.append(f"재료: {ing}")
    if steps: parts.append(f"조리: {steps}")
    return " | ".join(parts)

# ---------------- Keyword gate ------------------
COOK_KEYWORDS = [
    "메뉴","레시피","레시피명","조리","조리순서","방법","요리",
    "만드는법","요리법","recipe","cook","how to cook","how to make"
]
def has_cook_keyword(q: str) -> Tuple[bool, List[str]]:
    qn = q.lower()
    hits = [kw for kw in COOK_KEYWORDS if kw.lower() in qn]
    return (len(hits) > 0, hits)

# ---------------- LLM ---------------------------
def get_llm(model="gpt-4o-mini", temperature=0.2):
    api = os.getenv("OPENAI_API_KEY")
    if not api: raise RuntimeError("OPENAI_API_KEY not set")
    return ChatOpenAI(model=model, temperature=temperature, timeout=60, max_retries=2, api_key=api)

def llm_answer(llm, question: str, contexts: List[str]) -> str:
    context = "\n\n".join(contexts)[:6000]
    prompt = (
        "당신은 레시피 도우미입니다. 아래 컨텍스트의 사실만 사용해, 가능한 범위에서 간결히 답하세요.\n"
        "부족한 세부는 '정보 부족'이라고 표기하세요. 레시피는 (메뉴명/재료/간단 조리 요약) 형태로 정리하세요.\n\n"
        f"컨텍스트:\n{context}\n\n질문: {question}\n\n답변:"
    )
    return llm.invoke(prompt).content.strip()

def llm_rewrite(llm, question: str) -> str:
    prompt = (
        "다음 질문은 모호합니다. '메뉴/레시피/조리/조리순서/방법/요리' 중 최소 한 단어를 포함하여, "
        "요리 의도가 드러나도록 검색 가능한 한국어 한 문장으로 재작성하세요. 예: '계란으로 만들 수 있는 간단한 아침 메뉴 레시피 알려줘'.\n\n"
        f"원문: {question}\n재작성:"
    )
    return llm.invoke(prompt).content.strip()

def llm_judge(llm, question: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
    context = "\n\n".join(contexts)[:6000]
    prompt = (
        "당신은 RAG 평가자입니다. 아래 답변을 1~5점으로 평가하세요.\n"
        "1) 정합성(Faithfulness): 문서 근거에 충실?\n"
        "2) 관련성(Relevance): 질문과 직접 관련?\n"
        "3) 할루시네이션(Hallucination): 문서에 없는 내용 생성 정도(1=없음, 5=심함)\n"
        "출력: 'Faith=#, Rel=#, Hall=#'\n\n"
        f"컨텍스트:\n{context}\n\n질문:{question}\n\n답변:{answer}\n\n평가:"
    )
    raw = llm.invoke(prompt).content.strip()
    m = re.search(r"Faith\s*=\s*(\d).*?Rel\s*=\s*(\d).*?Hall\s*=\s*(\d)", raw, re.I|re.S)
    if m:
        f, r, h = map(int, m.groups())
    else:
        digs = re.findall(r"\d", raw)
        f, r, h = (int(digs[0]), int(digs[1]), int(digs[2])) if len(digs)>=3 else (3,3,3)
    return {"faith": f, "rel": r, "hall": h, "raw": raw}

# ---------------- Retrieval ---------------------
def get_vectorstore():
    emb = OpenAIEmbeddings(model=EMBED_MODEL)
    return Chroma(collection_name=COLLECTION, embedding_function=emb, persist_directory=PERSIST)

def retrieve_with_scores(vs, query: str, k: int=4) -> Tuple[List[Any], List[float]]:
    # similarity_search_with_relevance_scores returns (docs, scores in [0,1])
    docs_scores = vs.similarity_search_with_relevance_scores(query, k=k)
    docs = [d for d,_ in docs_scores]
    scores = [float(s) for _,s in docs_scores]
    return docs, scores

def build_context(docs: List[Any]) -> List[str]:
    merged = []
    seen = set()
    for d in docs:
        meta = d.metadata if isinstance(d.metadata, dict) else {}
        key = meta.get("recipe_id") or meta.get("url") or id(d)
        if key in seen: continue
        seen.add(key)
        content = (getattr(d, "page_content", "") or "").strip()
        # CSV enrich / fallback
        rid = meta.get("recipe_id")
        url = meta.get("url")
        title = meta.get("title")
        csv_txt = csv_row_text(rid, url, title)
        if csv_txt:
            if len(content) < 200 or ("조리" not in content and "steps" not in content.lower() and "요리순서" not in content):
                content = (content + "\n" + csv_txt).strip() if content else csv_txt
        if content:
            merged.append(content)
    return merged

# ---------------- Core Logic --------------------
def run_for_query(llm, vs, q, top_k, threshold, group) -> Dict[str, Any]:
    # --- 1) 1차 검색/판정 (원 질문) ---
    key_ok0, key_hits0 = has_cook_keyword(q)
    docs0, scores0 = retrieve_with_scores(vs, q, k=top_k)
    ctx0 = build_context(docs0)
    top1_0 = float(scores0[0]) if scores0 else 0.0
    low0 = (top1_0 < threshold) or (not key_ok0)

    used_q = q
    used_ctx = ctx0
    used_scores = scores0
    key_ok = key_ok0
    key_hits = key_hits0
    low = low0

    # --- 2) 그룹 로직 ---
    if group == "A":
        if not ctx0:
            answer = "제공된 문서가 없어 답변이 제한적입니다. 질문을 '레시피/조리/방법' 포함해 더 구체화하면 품질이 좋아집니다."
        else:
            answer = llm_answer(llm, q, ctx0)

    elif group == "B":
        if low0:
            # 재작성 → 재검색 → 재판정
            rq = llm_rewrite(llm, q)
            key_ok1, key_hits1 = has_cook_keyword(rq)
            docs1, scores1 = retrieve_with_scores(vs, rq, k=top_k)
            ctx1 = build_context(docs1)
            top1_1 = float(scores1[0]) if scores1 else 0.0
            low1 = (top1_1 < threshold) or (not key_ok1)

            # 최종 사용값은 재작성 후 결과로 갱신
            used_q, used_ctx, used_scores = rq, ctx1, scores1
            key_ok, key_hits, low = key_ok1, key_hits1, low1

            if not ctx1:
                answer = "관련 문서가 충분치 않아 답변이 어렵습니다. 재료·조리법·시간을 포함해 다시 물어봐 주세요."
            else:
                answer = llm_answer(llm, rq, ctx1)
        else:
            answer = llm_answer(llm, q, ctx0)

    elif group == "C":
        if low0:
            answer = "해당 질문은 제공된 레시피 지식만으로는 충분히 답변하기 어렵습니다. '메뉴/레시피/조리/방법'을 포함해 다시 질문해 주세요."
            used_ctx = []
        else:
            answer = llm_answer(llm, q, ctx0)
    else:
        raise ValueError("group must be A/B/C")

    # --- 3) 평가 ---
    judge = llm_judge(llm, q, answer, used_ctx)

    # --- 4) 로깅 (전/후 모두 기록) ---
    return {
        "query": q,
        "group": group,
        # 최종(보고용)
        "keyword_ok": key_ok,
        "keyword_hits": ", ".join(key_hits),
        "used_query": used_q,
        "top1": float(used_scores[0]) if used_scores else 0.0,
        "topk": ", ".join(f"{s:.3f}" for s in used_scores),
        "low": bool(low),
        "context_len": sum(len(x) for x in used_ctx),
        "ctx_docs": len(used_ctx),
        "answer": answer,
        "faith": judge["faith"],
        "rel": judge["rel"],
        "hall": judge["hall"],
        "judge_raw": judge["raw"],
        # 디버그/분석용(전/후 비교)
        "top1_pre": top1_0,
        "topk_pre": ", ".join(f"{s:.3f}" for s in scores0),
        "keyword_ok_pre": key_ok0,
        "keyword_hits_pre": ", ".join(key_hits0),
        "low_pre": bool(low0),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="queries_ko.json")
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.20)  # similarity score threshold
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    load_dotenv()
    llm = get_llm(model=args.model)
    vs = get_vectorstore()

    qlist = json.loads(Path(args.queries).read_text(encoding="utf-8"))

    rows = []
    for q in qlist:
        for g in ["A","B","C"]:
            try:
                row = run_for_query(llm, vs, q, args.top_k, args.threshold, g)
            except Exception as e:
                row = {"query": q, "group": g, "answer": f"[ERROR] {e}", "faith": 0, "rel":0, "hall": 5}
            rows.append(row)
            print(f"[{g}] {q[:28]}…  top1={row.get('top1',0):.3f} low={row.get('low')}  F{row['faith']} R{row['rel']} H{row['hall']}")

    df = pd.DataFrame(rows)
    resdir = Path("results"); resdir.mkdir(exist_ok=True)
    df.to_csv(resdir/"abc_results.csv", index=False, encoding="utf-8-sig")

    summary = (
        df.groupby("group")
          .agg(avg_faith=("faith","mean"),
               avg_rel=("rel","mean"),
               avg_hall=("hall","mean"),
               low_rate=("low","mean"))
          .reset_index()
          .sort_values("group")
    )
    summary.to_csv(resdir/"abc_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
