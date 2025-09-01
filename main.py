from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR   = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"      # test/static/

load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / "rag_hybrid_app" / ".env")

app = FastAPI(title="Hybrid RAG Recipe Assistant")

# ── 1) RAG API ────────────────────────────────────────────────
from rag_hybrid_app.graph.rag_flow import rag_flow, RAGState  # noqa: E402

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_handler(body: QueryRequest):
    state = RAGState(query=body.query)
    return rag_flow.invoke(state)

# ── 2) 루트 → index.html ─────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(STATIC_DIR / "index.html")

# ── 3) 정적 파일:  /static/…  ────────────────────────────────
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
#  ↑↑  반드시 RAG 라우터 정의 이후에 배치!

# ── 4) 로컬 실행 ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
