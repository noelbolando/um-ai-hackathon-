"""
agent.py
--------
The Catalog Agent.

Given a search query (plain English), it:
  1. Embeds the query with nomic-embed-text via Ollama
  2. Performs a semantic similarity search against the ChromaDB course collection
  3. Returns the top-N matching courses as a list of dicts
"""

import chromadb
import ollama

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR      = "chroma_db"
COLLECTION_NAME = "courses"
EMBED_MODEL     = "nomic-embed-text"
DEFAULT_TOP_K   = 5
# ─────────────────────────────────────────────────────────────────────────────


def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)


def embed_query(query: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=query)
    return response["embedding"]


def search_courses(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Embed the query and return the top-k most semantically similar courses.

    Returns a list of dicts, each with keys:
        course code, course description, semester taught, taught by, distance
    """
    collection = _get_collection()
    query_embedding = embed_query(query)

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
        lines.append(
            f"{i}. [{c['course code']}] {c['course description']} "
            f"| Semester: {c['semester taught']} "
            f"| Instructor: {c['taught by']}"
        )
    return "\n".join(lines)
