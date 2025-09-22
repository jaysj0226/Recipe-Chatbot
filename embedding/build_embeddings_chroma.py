#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a Chroma vector DB for advanced RAG from a recipes CSV.

CSV columns expected (detected from your file):
- title (str, required)
- ingredients (str or JSON-encoded list, optional)
- steps (str or JSON-encoded list, optional)
- url (str, unique-ish; used as stable ID when present)
- image_url (str, optional)
- description (str, optional; mostly empty in your file)

Key features
- Cleans & normalizes ingredients/steps (handles JSON list or plain text)
- Composes a structured document per recipe
- Optional recursive chunking (char-based) to keep chunks ~N chars
- Two embedding backends:
    1) OpenAI (default): text-embedding-3-large
    2) Local (HuggingFace): BAAI/bge-m3 (multilingual)
- Batch-wise ingestion to control memory
- Deterministic IDs (by URL or sha1(text))
- Persists a Chroma collection to disk (--persist_dir)

Usage
------
python build_embeddings_chroma.py   --csv /path/to/10000recipe_dataset.csv   --persist_dir ./chroma_recipes   --collection recipes-v1   --embedding_backend openai   --openai_model text-embedding-3-large   --chunk_size 1500 --chunk_overlap 200   --batch_size 200

Local backend example:
python build_embeddings_chroma.py   --csv /path/to/10000recipe_dataset.csv   --persist_dir ./chroma_recipes_local   --collection recipes-v1   --embedding_backend local   --local_model BAAI/bge-m3   --chunk_size 1500 --chunk_overlap 200   --batch_size 200

Requirements (pip)
------------------
pip install -U:
  langchain-core langchain-text-splitters langchain-openai langchain-chroma langchain-community
  chromadb sentence-transformers tiktoken pandas python-dotenv

Notes
-----
- Set OPENAI_API_KEY in your environment when using --embedding_backend openai
- With ~250k rows, plan storage & time accordingly; consider filtering or sharding by category if needed.
"""

import os, re, json, argparse, math, sys, unicodedata, hashlib
from typing import List, Dict, Any, Iterable, Optional
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

import pandas as pd

# ---- LangChain imports (support both >=0.2 and legacy community namespace) ----
try:
    from langchain_chroma import Chroma  # modern
except Exception:
    from langchain_community.vectorstores import Chroma  # fallback

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    # legacy fallback
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.docstore.document import Document  # legacy

# Embeddings
_BACKEND_OPENAI = "openai"
_BACKEND_LOCAL = "local"

def _make_embeddings(backend: str, openai_model: str, local_model: str, embed_chunk_texts: int):
    if backend == _BACKEND_OPENAI:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=openai_model, chunk_size=embed_chunk_texts, check_embedding_ctx_length=True)
    elif backend == _BACKEND_LOCAL:
        # sentence-transformers backend
        from langchain_community.embeddings import HuggingFaceEmbeddings
        # Normalize to improve cosine similarity behavior
        return HuggingFaceEmbeddings(
            model_name=local_model,
            encode_kwargs={"normalize_embeddings": True}
        )
    else:
        raise ValueError(f"Unsupported embedding backend: {backend}")

# ---- Text normalization helpers ----
def _maybe_load_json_list(x: Any) -> Optional[List[str]]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    s = str(x).strip()
    if not s:
        return None
    # Try JSON first
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(i).strip() for i in v if str(i).strip()]
    except Exception:
        pass
    # Fallback: split by common delimiters
    if "\n" in s:
        parts = [p.strip(" -•\t") for p in s.splitlines() if p.strip(" -•\t")]
    else:
        parts = [p.strip() for p in re.split(r"[;,•·\t]", s) if p.strip()]
    return parts or None

def _normalize_text(s: Any) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    # NFKC normalize; collapse whitespace
    text = unicodedata.normalize("NFKC", str(s))
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text

def compose_document(row: Dict[str, Any]) -> str:
    title = _normalize_text(row.get("title", ""))
    ing_list = _maybe_load_json_list(row.get("ingredients"))
    steps_list = _maybe_load_json_list(row.get("steps"))
    url = _normalize_text(row.get("url", ""))
    image_url = _normalize_text(row.get("image_url", ""))

    parts = []
    if title:
        parts.append(f"# {title}")
    if ing_list:
        parts.append("## Ingredients")
        parts.append("\n".join(f"- {i}" for i in ing_list))
    elif _normalize_text(row.get("ingredients", "")):
        parts.append("## Ingredients")
        parts.append(_normalize_text(row.get("ingredients", "")))

    if steps_list:
        parts.append("## Steps")
        parts.append("\n".join(f"{idx+1}. {s}" for idx, s in enumerate(steps_list)))
    elif _normalize_text(row.get("steps", "")):
        parts.append("## Steps")
        parts.append(_normalize_text(row.get("steps", "")))

    meta_lines = []
    if url:
        meta_lines.append(f"Source: {url}")
    if image_url:
        meta_lines.append(f"Image: {image_url}")
    if meta_lines:
        parts.append("\n".join(meta_lines))

    return "\n\n".join(p for p in parts if p and p.strip())

def deterministic_id(row: Dict[str, Any], text: str) -> str:
    u = str(row.get("url", "")).strip()
    if u:
        return u  # trust URL to be stable
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def iter_documents(df: pd.DataFrame) -> Iterable[Document]:
    for _, row in df.iterrows():
        text = compose_document(row)
        if not text or len(text) < 30:
            continue
        doc_id = deterministic_id(row, text)
        metadata = {
            "title": _normalize_text(row.get("title", "")),
            "url": _normalize_text(row.get("url", "")),
            "image_url": _normalize_text(row.get("image_url", "")),
            # you can add category fields here later if available
        }
        yield Document(page_content=text, metadata=metadata, id=doc_id)  # type: ignore

def chunk_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    if chunk_size <= 0:
        return docs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True,
    )
    out: List[Document] = []
    for d in docs:
        parts = splitter.split_documents([d])
        # propagate metadata & stable child IDs
        for i, p in enumerate(parts):
            # Derive child ID from parent
            pid = (d.id if hasattr(d, "id") else hashlib.sha1(d.page_content.encode("utf-8")).hexdigest())  # type: ignore
            p.metadata = {**d.metadata, **p.metadata, "parent_id": pid, "chunk": i}
            out.append(p)
    return out

def batched(iterable: Iterable[Any], n: int) -> Iterable[list]:
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    ap = argparse.ArgumentParser(description="Build Chroma embeddings from a recipes CSV.")
    ap.add_argument("--csv", required=True, help="Path to CSV (expects columns: title, ingredients, steps, url, image_url)")
    ap.add_argument("--persist_dir", required=True, help="Directory to persist the Chroma collection")
    ap.add_argument("--collection", default="recipes-v1", help="Chroma collection name")
    ap.add_argument("--embedding_backend", choices=["openai", "local"], default="openai")
    ap.add_argument("--openai_model", default="text-embedding-3-large")
    ap.add_argument("--local_model", default="BAAI/bge-m3")
    ap.add_argument("--batch_size", type=int, default=200, help="Docs per add_documents() call before persist")
    ap.add_argument("--max_rows", type=int, default=0, help="Optional cap; 0 means all rows")
    ap.add_argument("--drop_dupe_urls", action="store_true", help="Drop duplicated URLs keeping the longest steps")
    ap.add_argument("--min_text_len", type=int, default=30, help="Skip docs shorter than this many characters")
    ap.add_argument("--chunk_size", type=int, default=1500, help="Character-based chunk size (0 disables chunking)")
    ap.add_argument("--chunk_overlap", type=int, default=200, help="Character overlap between chunks")
    ap.add_argument("--embed_chunk_texts", type=int, default=64, help="Max number of texts per embeddings API request (mitigate 300k tokens/request limit)")
    args = ap.parse_args()

    os.makedirs(args.persist_dir, exist_ok=True)

    print(f"[Load] {args.csv}")
    df = pd.read_csv(args.csv, low_memory=False)

    # Optional: drop duplicates by URL keeping the row with longest steps length
    if args.drop_dupe_urls and "url" in df.columns:
        def steps_len(x):
            try:
                lst = _maybe_load_json_list(x)
                if lst:
                    return sum(len(s) for s in lst)
                return len(str(x)) if x == x else 0
            except Exception:
                return len(str(x)) if x == x else 0
        df["_steps_len"] = df.get("steps").apply(steps_len) if "steps" in df.columns else 0
        df = df.sort_values("_steps_len", ascending=False)               .drop_duplicates(subset=["url"], keep="first")               .drop(columns=["_steps_len"], errors="ignore")
        print(f"[Dedup] Rows after URL dedup: {len(df):,}")

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows)
        print(f"[Cap] Using first {len(df):,} rows")

    # Build embeddings func
    embeddings = _make_embeddings(args.embedding_backend, args.openai_model, args.local_model, args.embed_chunk_texts)

    # Prepare vectorstore
    print(f"[Chroma] persist_dir={args.persist_dir} collection={args.collection}")
    vectordb = Chroma(
        collection_name=args.collection,
        embedding_function=embeddings,
        persist_directory=args.persist_dir,
    )

    # Build & (optionally) chunk documents, then ingest in batches
    total_docs = 0
    for docs_batch in batched(iter_documents(df), args.batch_size):
        # filter by min length
        docs_batch = [d for d in docs_batch if len(d.page_content) >= args.min_text_len]
        if not docs_batch:
            continue
        if args.chunk_size and args.chunk_size > 0:
            docs_batch = chunk_documents(docs_batch, args.chunk_size, args.chunk_overlap)
        # Attach deterministic IDs for children chunks
        ids = []
        for d in docs_batch:
            base = d.metadata.get("parent_id") or getattr(d, "id", None) or hashlib.sha1(d.page_content.encode("utf-8")).hexdigest()
            chunk = d.metadata.get("chunk")
            ids.append(f"{base}::c{chunk}" if chunk is not None else base)

        vectordb.add_documents(documents=docs_batch, ids=ids)
        total_docs += len(docs_batch)
        print(f"[Ingest] +{len(docs_batch):,} (cum {total_docs:,}) -> persist...")
        

        # 변경 후:
        try:
            vectordb.persist()
            print(f"[Ingest] +{len(docs_batch):,} (cum {total_docs:,}) -> persist...")
        except AttributeError:
            print(f"[Ingest] +{len(docs_batch):,} (cum {total_docs:,}) -> auto-saved...")

    print(f"[Done] Total chunks ingested: {total_docs:,}. DB at: {args.persist_dir}")

if __name__ == "__main__":
    main()
