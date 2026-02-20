# 🎓 Course Finder

An AI-powered course recommendation tool that uses a local LLM (Mistral via Ollama) and a semantic vector search (ChromaDB + nomic-embed-text) to help students find relevant courses from a catalog.

## Architecture

```
User Query
    │
    ▼
Mistral (extract search intent)
    │
    ▼
Catalog Agent → ChromaDB (semantic search)
    │
    ▼
Mistral (generate recommendation)
    │
    ▼
Streamlit UI
```

## Prerequisites

1. **Ollama** installed and running — https://ollama.com
2. Pull the required models:
   ```bash
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

## Setup

1. **Clone the repo and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your course catalog:**
   - Place your CSV in `data/courses.csv`
   - Required columns: `course code`, `course description`, `semester taught`, `taught by`
   - A sample catalog is already included for testing

3. **Ingest the catalog into the vector database:**
   ```bash
   python ingest.py
   ```
   This reads your CSV, generates embeddings, and stores them in ChromaDB. Re-run this whenever your catalog changes.

4. **Launch the app:**
   ```bash
   streamlit run app.py
   ```

## File Structure

```
course-finder/
├── README.md
├── requirements.txt
├── data/
│   └── courses.csv        # Your course catalog
├── ingest.py              # CSV → embeddings → ChromaDB
├── agent.py               # Catalog Agent: semantic search
├── llm.py                 # Mistral interface (query extraction + response)
└── app.py                 # Streamlit web UI
```

## Usage

Just describe what you want to learn in plain English:

- *"I want to take a course in negotiations next semester"*
- *"Looking for something related to data science or machine learning"*
- *"Are there any ethics or philosophy courses available?"*

The app will extract your intent, search the catalog semantically, and return a personalized list of matching courses.

## Customizing

- **Change the number of results** using the sidebar slider in the app
- **Swap in your own catalog** by replacing `data/courses.csv` and re-running `ingest.py`
- **Adjust the LLM prompts** in `llm.py` to change the tone or format of responses