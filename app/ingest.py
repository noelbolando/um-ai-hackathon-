"""
ingest.py
---------
Unified course ingestion script.

Reads and normalizes courses from multiple sources:
  - data/ross_courses.xlsx   (Ross School of Business)
  - data/seas_courses.csv    (School for Environment & Sustainability)
  - data/courses.csv         (sample/custom catalog, if present)

All courses are embedded with nomic-embed-text via Ollama and stored
in a single ChromaDB "courses" collection. Each record is tagged with
a 'source' field so the app knows where it came from.

Run once (or re-run whenever your catalog changes):
    python ingest.py
"""

import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR      = "chroma_db"
COLLECTION_NAME = "courses"
# ─────────────────────────────────────────────────────────────────────────────


def get_embedding(text: str) -> list[float]:
    return _embedder.encode(text).tolist()


# ── Source loaders ────────────────────────────────────────────────────────────

def load_ross_courses(path: str) -> list[dict]:
    """
    Load Ross MBA course schedule from xlsx.

    Relevant columns:
        SUBJECT, CATALOG NBR, COURSE TITLE, COURSE PREREQUISITES,
        MEETING TIMES, INSTRUCTOR, CREDITS, SESSION DESC, START DT, END DT
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Drop cancelled courses (none currently, but future-proof)
    if "CANCELLED" in df.columns:
        df = df[df["CANCELLED"].str.upper() != "Y"]

    records = []
    for _, row in df.iterrows():
        subject     = str(row.get("SUBJECT", "")).strip()
        catalog_nbr = str(row.get("CATALOG NBR", "")).strip()
        course_code = f"{subject} {catalog_nbr}".strip()
        title       = str(row.get("COURSE TITLE", "")).strip()
        prereqs     = str(row.get("COURSE PREREQUISITES", "")).strip()
        prereqs     = "" if prereqs.lower() == "nan" else prereqs
        meeting     = str(row.get("MEETING TIMES", "")).strip()
        meeting     = "" if meeting.lower() == "nan" else meeting
        instructor  = str(row.get("INSTRUCTOR", "")).strip()
        instructor  = "" if instructor.lower() == "nan" else instructor
        credits     = str(row.get("CREDITS", "")).strip()
        session     = str(row.get("SESSION DESC", "")).strip()
        session     = "" if session.lower() == "nan" else session
        start_dt    = str(row.get("START DT", "")).strip()
        end_dt      = str(row.get("END DT", "")).strip()

        # Build semester string from session + dates
        semester = session
        if start_dt and start_dt.lower() != "nan":
            try:
                start_parsed = pd.to_datetime(start_dt)
                year = start_parsed.year
                month = start_parsed.month
                term = "Winter" if month <= 5 else ("Summer" if month <= 8 else "Fall")
                semester = f"{term} {year}" + (f" ({session})" if session else "")
            except Exception:
                pass

        # Embed document: combine all meaningful text
        doc_parts = [f"Course: {course_code} — {title}."]
        if prereqs:
            doc_parts.append(f"Prerequisites: {prereqs}.")
        if instructor:
            doc_parts.append(f"Instructor: {instructor}.")
        if meeting:
            doc_parts.append(f"Schedule: {meeting}.")
        if credits:
            doc_parts.append(f"Credits: {credits}.")
        document = " ".join(doc_parts)

        records.append({
            "document": document,
            "metadata": {
                "course code":        course_code,
                "course description": title,
                "semester taught":    semester,
                "taught by":          instructor,
                "prerequisites":      prereqs,
                "meeting times":      meeting,
                "credits":            credits,
                "source":             "Ross",
            },
        })

    return records


def load_seas_courses(path: str) -> list[dict]:
    """
    Load all SEAS courses from CSV — all 490 rows including multi-section listings.

    Columns: course_number, title, term, credit_hours, instructor, description

    Missing descriptions are handled gracefully: the embed document falls back
    to course number + title + term so every section is still searchable.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    records = []
    for _, row in df.iterrows():
        course_code = str(row.get("course_number", "")).strip()
        title       = str(row.get("title", "")).strip()
        term        = str(row.get("term", "")).strip()
        term        = "" if term.lower() == "nan" else term
        credits     = str(row.get("credit_hours", "")).strip()
        credits     = "" if credits.lower() == "nan" else credits
        instructor  = str(row.get("instructor", "")).strip()
        instructor  = "" if instructor.lower() == "nan" else instructor
        description = str(row.get("description", "")).strip()
        description = "" if description.lower() == "nan" else description

        # Build embed document — always include code + title, add rest when present
        doc_parts = [f"Course: {course_code} — {title}."]
        if description:
            doc_parts.append(description)
        if instructor:
            doc_parts.append(f"Instructor: {instructor}.")
        if term:
            doc_parts.append(f"Term: {term}.")
        document = " ".join(doc_parts)

        records.append({
            "document": document,
            "metadata": {
                "course code":        course_code,
                "course description": title + (f": {description}" if description else ""),
                "semester taught":    term,
                "taught by":          instructor,
                "prerequisites":      "",
                "meeting times":      "",
                "credits":            credits,
                "source":             "SEAS",
            },
        })

    return records



def load_psu_courses(path: str) -> list[dict]:
    """
    Load Penn State courses from CSV.

    Columns: key, prefix, number, suffix, title, description,
             minimum credits, maximum credits, other (prereqs/notes)
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    records = []
    for _, row in df.iterrows():
        course_code = str(row.get("key", "")).strip()
        course_code = "" if course_code.lower() == "nan" else course_code
        title       = str(row.get("title", "")).strip()
        title       = "" if title.lower() == "nan" else title
        description = str(row.get("description", "")).strip()
        description = "" if description.lower() == "nan" else description
        min_credits = str(row.get("minimum credits", "")).strip()
        max_credits = str(row.get("maximum credits", "")).strip()
        prereqs     = str(row.get("other", "")).strip()
        prereqs     = "" if prereqs.lower() == "nan" else prereqs

        # Build credits display
        if min_credits and max_credits and min_credits != max_credits:
            credits = f"{min_credits}-{max_credits}"
        elif min_credits:
            credits = min_credits
        else:
            credits = ""

        # Build embed document
        doc_parts = [f"Course: {course_code} — {title}."]
        if description:
            doc_parts.append(description)
        if prereqs:
            doc_parts.append(f"Notes: {prereqs}.")
        document = " ".join(doc_parts)

        records.append({
            "document": document,
            "metadata": {
                "course code":        course_code,
                "course description": title + (f": {description}" if description else ""),
                "semester taught":    "",
                "taught by":          "",
                "prerequisites":      prereqs,
                "meeting times":      "",
                "credits":            credits,
                "source":             "PSU",
            },
        })

    return records

def load_sample_courses(path: str) -> list[dict]:
    """
    Load the original sample/custom courses.csv if it exists.
    Expected columns: course code, course description, semester taught, taught by
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    records = []
    for _, row in df.iterrows():
        code        = str(row.get("course code", "")).strip()
        description = str(row.get("course description", "")).strip()
        semester    = str(row.get("semester taught", "")).strip()
        instructor  = str(row.get("taught by", "")).strip()

        document = (
            f"Course: {code}. Description: {description}. "
            f"Semester: {semester}. Instructor: {instructor}."
        )

        records.append({
            "document": document,
            "metadata": {
                "course code":        code,
                "course description": description,
                "semester taught":    semester,
                "taught by":          instructor,
                "prerequisites":      "",
                "meeting times":      "",
                "credits":            "",
                "source":             "Custom",
            },
        })

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_records = []

    # Load Ross courses
    ross_path = "data/ross_courses.csv"
    if os.path.exists(ross_path):
        ross = load_ross_courses(ross_path)
        print(f"Loaded {len(ross)} Ross courses from {ross_path}")
        all_records.extend(ross)
    else:
        print(f"Skipping Ross courses (not found: {ross_path})")

    # Load SEAS courses
    seas_path = "data/seas_courses.csv"
    if os.path.exists(seas_path):
        seas = load_seas_courses(seas_path)
        print(f"Loaded {len(seas)} SEAS courses from {seas_path}")
        all_records.extend(seas)
    else:
        print(f"Skipping SEAS courses (not found: {seas_path})")

    # Load sample/custom courses (optional)
    sample_path = "data/courses.csv"
    if os.path.exists(sample_path):
        sample = load_sample_courses(sample_path)
        print(f"Loaded {len(sample)} custom courses from {sample_path}")
        all_records.extend(sample)


    # Load PSU courses
    psu_path = "data/PSUcourses.csv"
    if os.path.exists(psu_path):
        psu = load_psu_courses(psu_path)
        print(f"Loaded {len(psu)} PSU courses from {psu_path}")
        all_records.extend(psu)
    else:
        print(f"Skipping PSU courses (not found: {psu_path})")

    if not all_records:
        raise RuntimeError(
            "No course data found. Place at least one of these in data/:\n"
            "  ross_courses.xlsx, seas_courses.csv, courses.csv"
        )

    print(f"\nTotal courses to embed: {len(all_records)}")

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Dropped existing courses collection.")
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    # Embed and insert
    ids, embeddings, documents, metadatas = [], [], [], []

    for i, record in enumerate(all_records):
        source = record["metadata"].get("source", "")
        code   = record["metadata"].get("course code", f"#{i}")
        print(f"  Embedding [{i+1}/{len(all_records)}] [{source}]: {code}")

        embedding = get_embedding(record["document"])

        ids.append(str(i))
        embeddings.append(embedding)
        documents.append(record["document"])
        metadatas.append(record["metadata"])

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"\nDone! {len(ids)} courses stored in ChromaDB at '{CHROMA_DIR}/'")
    sources = {}
    for r in all_records:
        s = r["metadata"].get("source", "Unknown")
        sources[s] = sources.get(s, 0) + 1
    for s, count in sources.items():
        print(f"  {s}: {count} courses")


if __name__ == "__main__":
    main()
    