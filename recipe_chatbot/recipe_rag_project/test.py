import sqlite3, json, os
DB = r"C:\Users\SunjaeJeong\Desktop\data\files_data\chroma_recipes_2025_09_16\chroma.sqlite3"
if os.path.exists(DB):
    con = sqlite3.connect(DB)
    cur = con.cursor()
    for name, meta in cur.execute("select name, metadata from collections"):
        try:
            j = json.loads(meta) if meta else {}
        except Exception:
            j = {}
            print("collection:", name, "metadata:", j)
    con.close()
else:
    print("sqlite not found:", DB)