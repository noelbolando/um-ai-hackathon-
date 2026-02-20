"""
faculty_ingest.py
-----------------
Reads the Ross faculty CSV, generates embeddings via Ollama (nomic-embed-text),
and stores everything in a dedicated ChromaDB collection called "faculty".

Run once (or re-run whenever your faculty data changes):
    python faculty_ingest.py
"""

import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH        = "data/ross_faculty.csv"
CHROMA_DIR      = "chroma_db"
COLLECTION_NAME = "faculty"

# Expected CSV columns
COL_NAME        = "name"
COL_TITLES      = "titles"
COL_BIO         = "bio"
COL_URL         = "profile_url"
# ─────────────────────────────────────────────────────────────────────────────


def build_document(row: pd.Series) -> str:
    """
    Combine faculty fields into a single rich string for embedding.
    We weight the bio heavily since it contains research interests.
    """
    name   = row.get(COL_NAME, "")
    titles = row.get(COL_TITLES, "")
    bio    = row.get(COL_BIO, "")

    # Some faculty have no bio — still embed name + title so they're findable
    parts = [f"Name: {name}."]
    if titles:
        parts.append(f"Title: {titles}.")
    if bio and str(bio).strip().lower() not in ("nan", ""):
        parts.append(f"Background and Research: {bio}")

    return " ".join(parts)


def get_embedding(text: str) -> list[float]:
    return _embedder.encode(text).tolist()


def main():
    # 1. Load faculty data
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Could not find '{CSV_PATH}'. "
            "Place ross_faculty.csv in the data/ folder."
        )

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip().str.lower()
    print(f"Loaded {len(df)} faculty records from {CSV_PATH}")

    # 2. Connect to ChromaDB (same persistent store as courses)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Drop existing faculty collection so re-runs start fresh
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Dropped existing faculty collection.")
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    # 3. Embed and insert each faculty member
    ids, embeddings, documents, metadatas = [], [], [], []

    for i, row in df.iterrows():
        doc = build_document(row)
        name = row.get(COL_NAME, f"Faculty {i}")
        print(f"  Embedding [{i+1}/{len(df)}]: {name}")

        embedding = get_embedding(doc)

        bio_val = row.get(COL_BIO, "")
        url_val = row.get(COL_URL, "")

        ids.append(str(i))
        embeddings.append(embedding)
        documents.append(doc)
        metadatas.append({
            COL_NAME:   str(name),
            COL_TITLES: str(row.get(COL_TITLES, "")),
            COL_BIO:    str(bio_val) if str(bio_val).lower() != "nan" else "",
            COL_URL:    str(url_val) if str(url_val).lower() != "nan" else "",
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"\nDone! {len(ids)} faculty records stored in ChromaDB at '{CHROMA_DIR}/'")


if __name__ == "__main__":
    main()
    