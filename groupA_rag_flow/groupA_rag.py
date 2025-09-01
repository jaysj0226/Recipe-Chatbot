
from __future__ import annotations
from typing import List, TypedDict, Optional
import argparse, os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

load_dotenv()

class AState(TypedDict, total=False):
    query: str
    docs: List[str]
    answer: Optional[str]

import os as _os
EMB = OpenAIEmbeddings(model=_os.environ.get("GROUPA_EMBED_MODEL","text-embedding-3-small"))
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 레시피 도우미입니다. 제공된 컨텍스트의 사실만 사용해 간결히 답하세요. "
     "컨텍스트가 비어있거나 근거가 부족하면 반드시 '제공된 문서가 없어 답변할 수 없습니다. 질문을 더 구체화해 주세요.'라고 답하세요."),
    ("human", "컨텍스트:\n{context}\n\n질문: {question}")
])

def load_vs(persist: str, collection: str):
    return Chroma(collection_name=collection, embedding_function=EMB, persist_directory=persist)

def retrieve(state: AState, k: int = 4, persist: str | None = None, collection: str | None = None,
             score_threshold: float = 0.0) -> AState:
    persist = persist or os.environ.get("GROUPA_PERSIST", "./chroma_rag_hybrid_db")
    collection = collection or os.environ.get("GROUPA_COLLECTION", "recipe_hybrid_rag")
    vs = load_vs(persist, collection)
    search_type = "similarity_score_threshold" if score_threshold > 0 else "similarity"
    search_kwargs = {"k": k}
    if score_threshold > 0:
        search_kwargs["score_threshold"] = score_threshold
    retr = vs.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    docs = retr.get_relevant_documents(state["query"])
    state["docs"] = [d.page_content for d in docs]
    return state

def llm_answer(state: AState, model: str = "gpt-4o-mini") -> AState:
    if not state.get("docs"):
        state["answer"] = "제공된 문서가 없어 답변할 수 없습니다. 질문을 더 구체화해 주세요."
        return state
    llm = ChatOpenAI(model=model, temperature=0.2)
    context = "\n\n".join(state.get("docs") or [])[:4000]
    chain = PROMPT | llm
    out = chain.invoke({"context": context, "question": state["query"]})
    state["answer"] = out.content.strip()
    return state

def build_graph():
    g = StateGraph(AState)
    g.add_node("retrieve", retrieve)
    g.add_node("llm_answer", llm_answer)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "llm_answer")
    g.add_edge("llm_answer", END)
    return g.compile()

graph = build_graph()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--persist", default="./chroma_basic")
    ap.add_argument("--collection", default="basic_rag")
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--score_threshold", type=float, default=0.0, help="0=항상 k개, >0이면 임계값 기반 필터")
    args = ap.parse_args()

    os.environ["GROUPA_PERSIST"] = args.persist
    os.environ["GROUPA_COLLECTION"] = args.collection

    init: AState = {"query": args.q}

    out = graph.invoke(init, config={
        "configurable": {
            "retrieve": {"k": args.k, "persist": args.persist, "collection": args.collection,
                         "score_threshold": args.score_threshold},
            "llm_answer": {"model": args.model},
        }
    })
    print(out["answer"])
