
import os, argparse
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

ap = argparse.ArgumentParser()
ap.add_argument("--persist", default=os.environ.get("GROUPA_PERSIST","./chroma_rag_hybrid_db"))
ap.add_argument("--collection", default=os.environ.get("GROUPA_COLLECTION","recipe_hybrid_rag"))
ap.add_argument("--embed_model", default=os.environ.get("GROUPA_EMBED_MODEL","text-embedding-3-small"))
ap.add_argument("--q", required=True)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--threshold", type=float, default=float(os.environ.get("GROUPA_SCORE_THRESHOLD","0.0")))
args = ap.parse_args()

print(f"[persist] {args.persist}")
print(f"[collection] {args.collection}")
print(f"[embed_model] {args.embed_model}")
emb = OpenAIEmbeddings(model=args.embed_model)
vs = Chroma(collection_name=args.collection, embedding_function=emb, persist_directory=args.persist)

search_type = "similarity_score_threshold" if args.threshold > 0 else "similarity"
kwargs = {"k": args.k}
if args.threshold > 0:
    kwargs["score_threshold"] = args.threshold

retr = vs.as_retriever(search_type=search_type, search_kwargs=kwargs)
docs = retr.get_relevant_documents(args.q)
print(f"retrieved_count={len(docs)}")
for i,d in enumerate(docs,1):
    print(f"[{i}] {d.metadata} | {d.page_content[:80].replace('\\n',' ')}...")
