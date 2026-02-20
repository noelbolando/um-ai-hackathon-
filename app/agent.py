"""
agent.py
--------
The Course Catalog Agent.

Given a plain-English query, it:
  1. Embeds the query with nomic-embed-text via Ollama
  2. Performs a semantic similarity search against the ChromaDB courses collection
  3. Returns the top-N matching courses as a list of dicts

Works with the unified course collection containing Ross, SEAS, and custom courses.
Each result includes a 'source' field indicating which catalog it came from.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "courses"
DEFAULT_TOP_K   = 10
# ─────────────────────────────────────────────────────────────────────────────


def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)


def search_courses(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Embed the query and return the top-k most semantically similar courses.

    Returns a list of dicts with keys:
        course code, course description, semester taught, taught by,
        prerequisites, meeting times, credits, source, distance
    """
    collection = _get_collection()
    query_embedding = _embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances"],
    )

    courses = []
    for metadata, distance in zip(
        results["metadatas"][0], results["distances"][0]
    ):
        course = dict(metadata)
        course["distance"] = round(distance, 4)
        courses.append(course)

    return courses


def format_courses_for_llm(courses: list[dict]) -> str:
    """Serialize retrieved courses into a clean string for the LLM prompt."""
    if not courses:
        return "No matching courses found."

    lines = []
    for i, c in enumerate(courses, 1):
        parts = [
            f"{i}. [{c.get('source', '')} | {c['course code']}] {c['course description']}"
        ]
        if c.get("semester taught"):
            parts.append(f"Semester: {c['semester taught']}")
        if c.get("taught by"):
            parts.append(f"Instructor: {c['taught by']}")
        if c.get("prerequisites"):
            parts.append(f"Prerequisites: {c['prerequisites']}")
        if c.get("meeting times"):
            parts.append(f"Schedule: {c['meeting times']}")
        if c.get("credits"):
            parts.append(f"Credits: {c['credits']}")
        lines.append(" | ".join(parts))

    return "\n".join(lines)
