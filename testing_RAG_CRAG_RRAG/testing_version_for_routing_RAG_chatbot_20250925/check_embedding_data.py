# check_vectorized.py
# ─────────────────────────────────────────────────────────────
# 사용법:  python check_vectorized.py \
#            --db_dir /home/ubuntu/chatbot/chroma_rag_hybrid_db \
#            --collection recipe_hybrid_rag \
#            --query "마늘장아찌" \
#            --k 5
# ─────────────────────────────────────────────────────────────
import os, argparse, textwrap
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from chromadb.config import Settings

def main():
    parser = argparse.ArgumentParser(description="Chroma 벡터 DB 검색 점검")
    parser.add_argument("--db_dir", required=True, help="Chroma DB persist_directory")
    parser.add_argument("--collection", required=True, help="Chroma collection_name")
    parser.add_argument("--query", default="마늘장아찌", help="검색어 (default: 마늘장아찌)")
    parser.add_argument("--k", type=int, default=5, help="반환할 문서 수")
    args = parser.parse_args()

    # .env (OPENAI_API_KEY) 로드
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 가 .env 또는 환경 변수에 설정돼 있지 않습니다.")

    # Chroma 로드 (read-only 권장)
    vs = Chroma(
        collection_name     = args.collection,
        persist_directory   = args.db_dir,
        embedding_function  = OpenAIEmbeddings(),
        client_settings     = Settings(allow_reset=False, anonymized_telemetry=False),
    )

    # 유사도 검색
    docs = vs.similarity_search(args.query, k=args.k)

    # 결과 출력
    if not docs:
        print(f"❌ '{args.query}'(으)로 검색된 문서가 없습니다.")
        return

    print(f"✅ '{args.query}' 관련 검색 결과 {len(docs)}건")
    print("-" * 60)
    for i, d in enumerate(docs, 1):
        snippet = textwrap.shorten(d.page_content.replace("\n", " "), width=160, placeholder=" …")
        print(f"[{i}] {snippet}")

if __name__ == "__main__":
    main()
