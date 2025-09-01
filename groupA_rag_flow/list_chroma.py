
import os, chromadb
path = os.environ.get("GROUPA_PERSIST","./chroma_rag_hybrid_db")
print(f"[persist] {path}")
client = chromadb.PersistentClient(path=path)
cols = client.list_collections()
if not cols:
    print("No collections found.")
else:
    for c in cols:
        try:
            count = c.count()
        except Exception as e:
            count = f"err: {e}"
        print(f"- name: {c.name} | id: {getattr(c,'id',None)} | count: {count}")
