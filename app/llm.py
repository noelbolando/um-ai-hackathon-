"""
llm.py
------
Handles all communication with llama3.2 via Ollama.

Responsibilities:
  1. extract_search_query()       — distill user message into a search query
  2. explain_course_match()       — why does this course help meet the goal?
  3. explain_faculty_match()      — why is this faculty relevant to the goal?
  4. explain_event_match()        — why does this event help meet the goal?
  5. explain_all_parallel()       — run all explanations concurrently (fast path)
  6. generate_goal_response()     — unified response covering courses, faculty + events
"""

import os
import streamlit as st
from groq import Groq
from concurrent.futures import ThreadPoolExecutor, as_completed

LLM_MODEL = "llama-3.1-8b-instant"
_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
_client = Groq(api_key=_api_key)


def _chat(prompt: str) -> str:
    """Single Ollama call — short helper used by parallel workers."""
    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
    )
    return response.choices[0].message.content.strip()


def _history_text(history: list[dict], n: int = 4) -> str:
    if not history:
        return ""
    lines = []
    for msg in history[-n:]:
        role = "Student" if msg["role"] == "user" else "Advisor"
        lines.append(f"{role}: {msg['content']}")
    return "Conversation so far:\n" + "\n".join(lines) + "\n\n"


def extract_search_query(user_message: str, history: list[dict]) -> str:
    """Extract a concise search query from the student's learning goal."""
    hist = ""
    if history:
        lines = []
        for m in history[-6:]:
            role = "Student" if m["role"] == "user" else "Advisor"
            lines.append(f"{role}: {m['content']}")
        hist = "\n".join(lines) + "\n"

    prompt = (
        f"Previous conversation:\n{hist}"
        f"Student's goal: \"{user_message}\"\n\n"
        "Extract the core academic topic(s). Return ONLY 2-6 keywords, no punctuation. "
        "Examples: \"negotiations conflict resolution\", \"machine learning neural networks\".\n"
        "Search query:"
    )
    return _chat(prompt)


def _course_prompt(course: dict, user_goal: str, history: list[dict]) -> str:
    return (
        f"{_history_text(history)}"
        f"Student's goal: \"{user_goal}\"\n\n"
        f"Course: {course['course code']} — {course['course description'][:200]}\n"
        f"Instructor: {course.get('taught by', '')}\n\n"
        "In 1-2 sentences, explain how this course helps the student's goal. "
        "Be specific. Do not start with \"This course\".\nExplanation:"
    )


def _faculty_prompt(faculty: dict, user_goal: str, history: list[dict]) -> str:
    bio = faculty.get("bio", "")[:300]
    return (
        f"{_history_text(history)}"
        f"Student's goal: \"{user_goal}\"\n\n"
        f"Faculty: {faculty['name']} — {faculty['titles']}\n"
        f"Bio: {bio}\n\n"
        "In 1-2 sentences, explain why connecting with this person helps the student's goal. "
        "Do not start with \"This professor\".\nExplanation:"
    )


def _event_prompt(event: dict, user_goal: str, history: list[dict]) -> str:
    desc = event.get("description", "")[:300]
    return (
        f"{_history_text(history)}"
        f"Student's goal: \"{user_goal}\"\n\n"
        f"Event: {event['title']} ({event.get('type', '')})\n"
        f"Description: {desc}\n\n"
        "In 1-2 sentences, explain why attending this event helps the student's goal. "
        "Do not start with \"This event\".\nExplanation:"
    )


def explain_course_match(course: dict, user_goal: str, history: list[dict]) -> str:
    return _chat(_course_prompt(course, user_goal, history))


def explain_faculty_match(faculty: dict, user_goal: str, history: list[dict]) -> str:
    return _chat(_faculty_prompt(faculty, user_goal, history))


def explain_event_match(event: dict, user_goal: str, history: list[dict]) -> str:
    return _chat(_event_prompt(event, user_goal, history))


def explain_all_parallel(
    courses: list[dict],
    faculty: list[dict],
    events: list[dict],
    user_goal: str,
    history: list[dict],
    max_workers: int = 8,
) -> None:
    """
    Run all explanation LLM calls concurrently using a thread pool.
    Mutates each item in-place by setting item["explanation"].
    Much faster than sequential calls — 30 calls in ~parallel vs series.
    """
    tasks = []

    for item in courses:
        tasks.append(("course", item, _course_prompt(item, user_goal, history)))
    for item in faculty:
        tasks.append(("faculty", item, _faculty_prompt(item, user_goal, history)))
    for item in events:
        tasks.append(("event", item, _event_prompt(item, user_goal, history)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(_chat, prompt): item
            for (_, item, prompt) in tasks
        }
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                item["explanation"] = future.result()
            except Exception:
                item["explanation"] = ""


def generate_goal_response(
    user_goal: str,
    courses: list[dict],
    faculty: list[dict],
    history: list[dict],
    events: list[dict] = None,
) -> str:
    """
    Generate a unified conversational response covering courses, faculty, and events.
    """
    def fmt_courses():
        if not courses:
            return "No matching courses found."
        return "\n".join(
            f"{i}. [{c['course code']}] {c['course description'][:80]}"
            for i, c in enumerate(courses[:5], 1)
        )

    def fmt_faculty():
        if not faculty:
            return "No matching faculty found."
        return "\n".join(
            f"{i}. {f['name']} — {f['titles'][:60]}"
            for i, f in enumerate(faculty[:5], 1)
        )

    def fmt_events():
        if not events:
            return "No matching events found."
        return "\n".join(
            f"{i}. {e['title']} ({e.get('type', '')})"
            for i, e in enumerate(events[:5], 1)
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a warm academic advisor. Given a student's learning goal and "
                "matched courses, faculty, and campus events, write a concise 3-4 sentence "
                "response that acknowledges their goal, highlights the best resources, and "
                "invites them to refine. No bullet lists."
            ),
        }
    ]
    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": (
            f"Goal: {user_goal}\n\n"
            f"Courses:\n{fmt_courses()}\n\n"
            f"Faculty:\n{fmt_faculty()}\n\n"
            f"Events:\n{fmt_events()}"
        ),
    })

    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()
