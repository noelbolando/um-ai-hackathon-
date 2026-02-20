"""
faculty_agent.py
----------------
The Faculty Catalog Agent.

Given a plain-English query, it:
  1. Embeds the query with nomic-embed-text via Ollama
  2. Performs a semantic similarity search against the ChromaDB faculty collection
  3. Returns the top-N matching faculty as a list of dicts
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "faculty"
DEFAULT_TOP_K   = 10
# ─────────────────────────────────────────────────────────────────────────────


def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)


def search_faculty(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Embed the query and return the top-k most semantically similar faculty.

    Returns a list of dicts with keys:
        name, titles, bio, profile_url, distance
    """
    collection = _get_collection()
    query_embedding = _embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances"],
    )

    faculty_list = []
    for metadata, distance in zip(
        results["metadatas"][0], results["distances"][0]
    ):
        faculty = dict(metadata)
        faculty["distance"] = round(distance, 4)
        faculty_list.append(faculty)

    return faculty_list


def format_faculty_for_llm(faculty_list: list[dict]) -> str:
    """Serialize retrieved faculty into a clean string for the LLM prompt."""
    if not faculty_list:
        return "No matching faculty found."

    lines = []
    for i, f in enumerate(faculty_list, 1):
        bio_snippet = f.get("bio", "")
        # Truncate very long bios for the prompt — LLM gets the gist
        if len(bio_snippet) > 600:
            bio_snippet = bio_snippet[:600] + "..."
        lines.append(
            f"{i}. {f['name']} — {f['titles']}\n"
            f"   Bio: {bio_snippet}\n"
            f"   Profile: {f.get('profile_url', 'N/A')}"
        )
    return "\n\n".join(lines)
