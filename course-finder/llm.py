"""
llm.py
------
Handles all communication with Mistral via Ollama.

Two responsibilities:
  1. extract_search_query()  — ask Mistral to distill the user's message
                               into a clean semantic search query
  2. generate_response()     — given the user's original message + retrieved
                               courses, ask Mistral to write a helpful reply
"""

import ollama

LLM_MODEL = "mistral"


def extract_search_query(user_message: str) -> str:
    """
    Ask Mistral to extract the core search intent from the user's message.
    Returns a concise search query string to send to the catalog agent.
    """
    prompt = f"""You are a helpful assistant that extracts course search queries.

A student said: "{user_message}"

Extract the core academic topic(s) or subject(s) they want to study and return ONLY a short search query (no explanation, no punctuation, just the key terms). For example: "negotiations conflict resolution" or "data structures algorithms".

Search query:"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


def generate_response(user_message: str, courses_text: str) -> str:
    """
    Given the user's original message and a formatted list of matching courses,
    ask Mistral to write a friendly, helpful recommendation.
    """
    prompt = f"""You are a friendly academic advisor helping a student find courses.

The student asked: "{user_message}"

Based on a search of the course catalog, here are the most relevant courses found:

{courses_text}

Write a helpful, conversational response that:
- Briefly acknowledges what the student is looking for
- Presents the matching courses in a clear, readable way
- Highlights why each course might be relevant to their interest
- Encourages them to ask follow-up questions

Keep your tone warm and encouraging."""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()
