"""
ingest.py
---------
Reads your courses CSV, generates embeddings via Ollama (nomic-embed-text),
and stores everything in a local ChromaDB collection.

Run once (or re-run whenever your catalog changes):
    python ingest.py
"""

import os
import pandas as pd
import chromadb
import ollama

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH        = "data/courses.csv"
CHROMA_DIR      = "chroma_db"
COLLECTION_NAME = "courses"
EMBED_MODEL     = "nomic-embed-text"

# Expected CSV columns
COL_CODE        = "course code"
COL_DESCRIPTION = "course description"
COL_SEMESTER    = "semester taught"
COL_INSTRUCTOR  = "taught by"
# ─────────────────────────────────────────────────────────────────────────────


def build_document(row: pd.Series) -> str:
    """Combine course fields into a single string for embedding."""
    return (
        f"Course: {row[COL_CODE]}. "
        f"Description: {row[COL_DESCRIPTION]}. "
        f"Semester: {row[COL_SEMESTER]}. "
        f"Instructor: {row[COL_INSTRUCTOR]}."
    )


def get_embedding(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def main():
    # 1. Load catalog
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Could not find '{CSV_PATH}'. "
            "Place your courses CSV in the data/ folder."
        )

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip().str.lower()  # normalize column names
    print(f"Loaded {len(df)} courses from {CSV_PATH}")

    # 2. Connect to ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Drop existing collection so re-runs start fresh
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Dropped existing collection.")
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    # 3. Embed and insert each course
    ids, embeddings, documents, metadatas = [], [], [], []

    for i, row in df.iterrows():
        doc = build_document(row)
        print(f"  Embedding [{i+1}/{len(df)}]: {row[COL_CODE]}")

        embedding = get_embedding(doc)

        ids.append(str(i))
        embeddings.append(embedding)
        documents.append(doc)
        metadatas.append({
            COL_CODE:        str(row[COL_CODE]),
            COL_DESCRIPTION: str(row[COL_DESCRIPTION]),
            COL_SEMESTER:    str(row[COL_SEMESTER]),
            COL_INSTRUCTOR:  str(row[COL_INSTRUCTOR]),
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"\nDone! {len(ids)} courses stored in ChromaDB at '{CHROMA_DIR}/'")


if __name__ == "__main__":
    main()
