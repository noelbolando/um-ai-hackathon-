"""
events_ingest.py
----------------
Embeds campus events into a ChromaDB "events" collection.

Run once (or re-run when the events data changes):
    python events_ingest.py
"""

import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

CHROMA_DIR      = "chroma_db"
COLLECTION_NAME = "events"


def get_embedding(text: str) -> list[float]:
    return _embedder.encode(text).tolist()


def clean(val) -> str:
    s = str(val).strip()
    return "" if s.lower() in ("nan", "none", "") else s


def load_events(path: str) -> list[dict]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    records = []
    for _, row in df.iterrows():
        title       = clean(row.get("Title", ""))
        subtitle    = clean(row.get("Subtitle", ""))
        event_type  = clean(row.get("Type", ""))
        description = clean(row.get("Description", ""))
        tags        = clean(row.get("Tags", ""))
        sponsors    = clean(row.get("Sponsors", ""))
        location    = clean(row.get("Location Name", ""))
        building    = clean(row.get("Building Name", ""))
        room        = clean(row.get("Room", ""))
        start_dt    = clean(row.get("Start Date / Time", ""))
        end_dt      = clean(row.get("End Date / Time", ""))
        cost        = clean(row.get("Cost", ""))
        permalink   = clean(row.get("Permalink", ""))

        location_parts = [p for p in [building, room, location] if p]
        full_location = ", ".join(dict.fromkeys(location_parts))

        doc_parts = [f"Event: {title}."]
        if subtitle:
            doc_parts.append(subtitle)
        if event_type:
            doc_parts.append(f"Type: {event_type}.")
        if description:
            doc_parts.append(description[:800])
        if tags:
            doc_parts.append(f"Tags: {tags}.")
        if sponsors:
            doc_parts.append(f"Hosted by: {sponsors}.")
        document = " ".join(doc_parts)

        records.append({
            "document": document,
            "metadata": {
                "title":       title,
                "subtitle":    subtitle,
                "type":        event_type,
                "description": description[:600] if description else "",
                "start":       start_dt,
                "end":         end_dt,
                "location":    full_location,
                "cost":        cost,
                "tags":        tags,
                "permalink":   permalink,
            },
        })

    return records


def main():
    path = "data/campus_events.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Events file not found: {path}")

    records = load_events(path)
    print(f"Loaded {len(records)} events from {path}")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Dropped existing events collection.")
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    ids, embeddings, documents, metadatas = [], [], [], []
    for i, record in enumerate(records):
        title = record["metadata"]["title"]
        print(f"  Embedding [{i+1}/{len(records)}]: {title[:60]}")
        embedding = get_embedding(record["document"])
        ids.append(str(i))
        embeddings.append(embedding)
        documents.append(record["document"])
        metadatas.append(record["metadata"])

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(f"\nDone! {len(ids)} events stored in ChromaDB at '{CHROMA_DIR}/'")


if __name__ == "__main__":
    main()
    