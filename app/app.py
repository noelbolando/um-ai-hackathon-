"""
app.py
------
Ross Advisor — Learning Goal Dashboard.

Layout:
  [Top bar: learning goal input]
  [Left: Preferences] [Center-Left: Courses] [Center-Right: Faculty] [Right: Chat]

Run with:
    streamlit run app.py
"""

import streamlit as st
from agent import search_courses
from faculty_agent import search_faculty
from events_agent import search_events
from llm import (
    extract_search_query,
    explain_all_parallel,
    generate_goal_response,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Curio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #080C14;
    color: #E2E8F4;
}
.stApp {
    background:
        radial-gradient(ellipse at 20% 20%, rgba(74,158,255,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(120,80,255,0.04) 0%, transparent 50%),
        #080C14;
}
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none; }

/* ── Navbar ── */
.navbar {
    background: rgba(8,12,20,0.95);
    border-bottom: 1px solid #1E2A42;
    padding: 0 1.5rem;
    height: 80px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.2rem;
    margin-bottom: 0;
    backdrop-filter: blur(12px);
    position: relative;
}
.navbar-logo {
    width: 34px;
    height: 34px;
    border-radius: 8px;
    background: linear-gradient(135deg, #4A9EFF 0%, #7B5FFF 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 1rem;
}
.navbar-title .star { font-size: 0.6em; opacity: 0.8; vertical-align: middle; position: relative; top: -2px; color: #C4A8FF; }
.navbar-title {
    color: #B0B8C8;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    text-align: center;
    line-height: 1;
}
.navbar-tagline {
    color: #4A6080;
    font-size: 1rem;
    font-weight: 400;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    text-align: center;
}

/* ── Goal bar ── */
.goal-bar {
    background: rgba(14,21,37,0.9);
    border-bottom: 1px solid #1E2A42;
    padding: 1rem 1.5rem 1.1rem 1.5rem;
    margin-bottom: 0;
    width: 100%;
    backdrop-filter: blur(12px);
}
.goal-label {
    color: #E2E8F4;
    font-size: 1.35rem;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.01em;
    margin-bottom: 0.6rem;
}

/* ── Message history bar ── */
.history-bar {
    background: rgba(14,21,37,0.6);
    border-bottom: 1px solid #1E2A42;
    padding: 0.6rem 1.5rem;
    margin-bottom: 0;
}
.history-entry {
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
    padding: 0.45rem 0;
    border-bottom: 1px solid #141E30;
}
.history-entry:last-child { border-bottom: none; }
.history-role-user {
    font-size: 0.7rem;
    font-weight: 600;
    color: #4A9EFF;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    white-space: nowrap;
    padding-top: 0.1rem;
    min-width: 48px;
}
.history-role-ai {
    font-size: 0.7rem;
    font-weight: 600;
    color: #7B5FFF;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    white-space: nowrap;
    padding-top: 0.1rem;
    min-width: 48px;
}
.history-text {
    font-size: 0.9rem;
    color: #8899BB;
    line-height: 1.5;
}

/* ── Panel ── */
.panel {
    background: #0E1525;
    border: 1px solid #1E2A42;
    border-radius: 12px;
    overflow: hidden;
    height: 100%;
}
.panel-header {
    padding: 0.8rem 1.1rem;
    border-bottom: 1px solid #1E2A42;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255,255,255,0.02);
    position: relative;
}
.panel-header .panel-count {
    position: absolute;
    right: 1.1rem;
}
.panel-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #8899BB;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.panel-count {
    font-size: 0.7rem;
    background: rgba(74,158,255,0.12);
    color: #4A9EFF;
    padding: 0.15rem 0.55rem;
    border-radius: 20px;
    font-weight: 600;
    border: 1px solid rgba(74,158,255,0.2);
}

/* ── Course card ── */
.course-card {
    padding: 0.85rem 1.1rem;
    border-bottom: 1px solid #141E30;
    animation: fadeUp 0.3s ease forwards;
    opacity: 0;
    transition: background 0.15s ease;
}
.course-card:hover { background: rgba(74,158,255,0.04); }
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.course-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #E2E8F4;
    margin-bottom: 0.3rem;
    line-height: 1.35;
}
.instructor-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(74,158,255,0.1);
    color: #4A9EFF;
    border: 1px solid rgba(74,158,255,0.2);
    border-radius: 4px;
    padding: 0.1rem 0.45rem;
    font-size: 0.82rem;
    font-weight: 500;
    margin-bottom: 0.35rem;
}
.why-text {
    font-size: 0.88rem;
    color: #7A8FA8;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: flex-start;
    gap: 0.35rem;
    line-height: 1.45;
}
.why-text::before {
    content: '→';
    color: #4A9EFF;
    font-weight: 600;
    flex-shrink: 0;
    opacity: 0.8;
}
.card-meta {
    font-size: 0.82rem;
    color: #4A6080;
    margin-bottom: 0.3rem;
}

/* ── Faculty card ── */
.faculty-card {
    padding: 0.85rem 1.1rem;
    border-bottom: 1px solid #141E30;
    animation: fadeUp 0.3s ease forwards;
    opacity: 0;
}
.faculty-card:hover { background: rgba(123,95,255,0.04); }
.faculty-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #E2E8F4;
    margin-bottom: 0.15rem;
}
.faculty-title-text {
    font-size: 0.82rem;
    color: #4A6080;
    margin-bottom: 0.4rem;
    line-height: 1.4;
}
.why-faculty {
    font-size: 0.88rem;
    color: #7A8FA8;
    background: rgba(123,95,255,0.06);
    border-left: 2px solid #7B5FFF;
    padding: 0.4rem 0.65rem;
    border-radius: 0 6px 6px 0;
    margin-bottom: 0.4rem;
    line-height: 1.45;
}
.profile-link {
    font-size: 0.84rem;
    color: #4A9EFF;
    font-weight: 500;
    text-decoration: none;
    opacity: 0.8;
}
.profile-link:hover { opacity: 1; text-decoration: underline; }

/* ── Event card ── */
.event-card {
    padding: 0.85rem 1.1rem;
    border-bottom: 1px solid #141E30;
    animation: fadeUp 0.3s ease forwards;
    opacity: 0;
}
.event-card:hover { background: rgba(74,255,178,0.03); }
.event-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #E2E8F4;
    margin-bottom: 0.25rem;
    line-height: 1.35;
}
.event-type {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    background: rgba(74,255,178,0.08);
    color: #4AFFB2;
    border: 1px solid rgba(74,255,178,0.15);
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    margin-bottom: 0.35rem;
}
.event-why {
    font-size: 0.88rem;
    color: #7A8FA8;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: flex-start;
    gap: 0.35rem;
    line-height: 1.45;
}
.event-why::before {
    content: '→';
    color: #4AFFB2;
    font-weight: 600;
    flex-shrink: 0;
    opacity: 0.7;
}
.event-meta {
    font-size: 0.82rem;
    color: #4A6080;
    margin-bottom: 0.3rem;
}
.event-link {
    font-size: 0.84rem;
    color: #4AFFB2;
    font-weight: 500;
    text-decoration: none;
    opacity: 0.8;
}
.event-link:hover { opacity: 1; text-decoration: underline; }
.event-cost {
    display: inline-block;
    font-size: 0.68rem;
    background: rgba(74,255,178,0.08);
    color: #4AFFB2;
    border: 1px solid rgba(74,255,178,0.15);
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    margin-left: 0.4rem;
}

/* ── Empty state ── */
.empty-state {
    padding: 3rem 1.2rem;
    text-align: center;
}
.empty-icon { font-size: 1.8rem; margin-bottom: 0.7rem; opacity: 0.4; }
.empty-text {
    font-size: 0.92rem;
    color: #FFFFFF;
    line-height: 1.5;
}

/* ── Streamlit overrides ── */
div[data-testid="stTextInput"] > div > div > input {
    border-radius: 8px !important;
    border: 1px solid #6B4FA0 !important;
    background: #2A1F3D !important;
    color: #E2E8F4 !important;
    caret-color: #C4A8FF !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 0.9rem !important;
}
div[data-testid="stTextInput"] > div > div > input::placeholder {
    color: #FFFFFF !important;
    opacity: 0.6 !important;
}
div[data-testid="stTextInput"] > div > div > input:focus {
    border-color: #C4A8FF !important;
    box-shadow: 0 0 0 3px rgba(196,168,255,0.15) !important;
}
div[data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }

.stButton > button {
    background: linear-gradient(135deg, #4A9EFF, #7B5FFF) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.1rem !important;
    font-size: 0.88rem !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #4A9EFF, #7B5FFF) !important;
}
.stProgress > div > div > div {
    background: #1E2A42 !important;
    border-radius: 0 !important;
}
.stProgress {
    position: sticky;
    top: 0;
    z-index: 999;
    margin: 0 !important;
    padding: 0 !important;
}
div[data-testid="stStatusWidget"] { display: none; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1E2A42; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "courses": [],
    "faculty": [],
    "events": [],
    "time_pref": "Morning",
    "credit_target": 12,
    "top_k": 10,
    "pending_query": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── NAVBAR ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <span class="navbar-title"><span class="star">✦</span> Curio <span class="star">✦</span></span>
    <span class="navbar-tagline">your academic pathfinder</span>
</div>
""", unsafe_allow_html=True)

# ── GOAL BAR ──────────────────────────────────────────────────────────────────
st.markdown('<div class="goal-bar">', unsafe_allow_html=True)
st.markdown('<div class="goal-label" style="font-size:1.15rem; letter-spacing:0.01em; text-transform:none; font-family: Crimson Pro, serif; font-weight:600; color:#000000;">What do you want to learn or achieve?</div>', unsafe_allow_html=True)

with st.form("goal_form", clear_on_submit=True):
    goal_cols = st.columns([8, 1])
    with goal_cols[0]:
        goal_input = st.text_input(
            "goal",
            placeholder="e.g. I want to break into consulting, or I want to understand behavioral economics...",
            label_visibility="collapsed",
        )
    with goal_cols[1]:
        go = st.form_submit_button("Find →", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── MESSAGE HISTORY ──────────────────────────────────────────────────────────
if st.session_state.messages:
    st.markdown('<div class="history-bar">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role_class = "history-role-user" if msg["role"] == "user" else "history-role-ai"
        role_label = "You" if msg["role"] == "user" else "Curio"
        text = msg["content"]
        # Truncate long AI responses in the history view
        if msg["role"] == "assistant" and len(text) > 300:
            text = text[:300] + "..."
        st.markdown(
            f'<div class="history-entry">' +
            f'<span class="{role_class}">{role_label}</span>' +
            f'<span class="history-text">{text}</span>' +
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ── MAIN LAYOUT ───────────────────────────────────────────────────────────────
courses_col, faculty_col, events_col = st.columns([1, 1, 1], gap="medium")

# ═══════════════════════════════
# CENTER-LEFT — Courses
# ═══════════════════════════════
with courses_col:
    courses = st.session_state.courses
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    course_count_html = f'<span class="panel-count">{len(courses)} found</span>' if courses else ""
    st.markdown(
        f'<div class="panel-header">'
        f'<span class="panel-title">📚 Recommended Courses</span>'
        f'{course_count_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not courses:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-text">Enter a learning goal to see course recommendations</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for i, c in enumerate(courses):
            delay = i * 0.06
            instructor = c.get("taught by", "")
            semester = c.get("semester taught", "")
            credits_val = c.get("credits", "")
            meeting = c.get("meeting times", "")
            prereqs = c.get("prerequisites", "")
            why = c.get("explanation", "")
            code = c.get("course code", "")
            desc = c.get("course description", "")
            short_desc = desc[:55] + ("..." if len(desc) > 55 else "")

            meta_parts = []
            if credits_val:
                meta_parts.append(f"{credits_val} cr")
            if meeting:
                meta_parts.append(meeting)
            if semester:
                meta_parts.append(semester)
            meta_str = "  ·  ".join(meta_parts)

            instructor_html = f'<div><span class="instructor-tag">👤 {instructor}</span></div>' if instructor else ""
            why_html = f'<div class="why-text">{why}</div>' if why else ""
            meta_html = f'<div class="card-meta">{meta_str}</div>' if meta_str else ""
            prereq_html = f'<div class="card-meta" style="font-size:0.73rem;">📋 {prereqs}</div>' if prereqs else ""

            st.markdown(f"""
            <div class="course-card" style="animation-delay:{delay}s">
                <div class="course-title">{code}: {short_desc}</div>
                {instructor_html}{why_html}{meta_html}{prereq_html}
            </div>
            """, unsafe_allow_html=True)



    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════
# CENTER-RIGHT — Faculty
# ═══════════════════════════════
with faculty_col:
    faculty = st.session_state.faculty
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    faculty_count_html = f'<span class="panel-count">{len(faculty)} found</span>' if faculty else ""
    st.markdown(
        f'<div class="panel-header">'
        f'<span class="panel-title">👤 Faculty to Connect With</span>'
        f'{faculty_count_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not faculty:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-text">Enter a learning goal to find faculty to network with</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for i, f in enumerate(faculty):
            delay = i * 0.06
            why = f.get("explanation", "")
            profile = f.get("profile_url", "")

            why_html = f'<div class="why-faculty">{why}</div>' if why else ""
            profile_html = f'<a class="profile-link" href="{profile}" target="_blank">View Profile →</a>' if profile else ""

            st.markdown(f"""
            <div class="faculty-card" style="animation-delay:{delay}s">
                <div class="faculty-name">{f.get('name','')}</div>
                <div class="faculty-title-text">{f.get('titles','')}</div>
                {why_html}{profile_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════
# RIGHT — Events
# ═══════════════════════════════
with events_col:
    events = st.session_state.events
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    event_count_html = f'<span class="panel-count">{len(events)} found</span>' if events else ""
    st.markdown(
        f'<div class="panel-header">'
        f'<span class="panel-title">📅 Campus Events</span>'
        f'{event_count_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not events:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-text">Enter a learning goal to discover relevant campus events</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for i, e in enumerate(events):
            delay = i * 0.06
            why = e.get("explanation", "")
            permalink = e.get("permalink", "")
            start = e.get("start", "")
            location = e.get("location", "")
            cost = e.get("cost", "")
            event_type = e.get("type", "")

            why_html = f'<div class="event-why">{why}</div>' if why else ""
            link_html = f'<a class="event-link" href="{permalink}" target="_blank">View Event →</a>' if permalink else ""

            meta_parts = []
            if start:
                meta_parts.append(f"📅 {start}")
            if location:
                location_short = location[:40] + ("..." if len(location) > 40 else "")
            meta_parts.append(f"📍 {location_short}")
            meta_str = "  ·  ".join(meta_parts)
            meta_html = f'<div class="event-meta">{meta_str}</div>' if meta_str else ""

            cost_html = ""
            if cost:
                short_cost = cost if len(cost) < 30 else "See event"
                cost_html = f'<span class="event-cost">{short_cost}</span>'

            st.markdown(f"""
            <div class="event-card" style="animation-delay:{delay}s">
                <div class="event-title">{e.get("title", "")}</div>
                <span class="event-type">{event_type}</span>{cost_html}
                {why_html}{meta_html}{link_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Trigger from goal bar ─────────────────────────────────────────────────────
if go and goal_input.strip():
    st.session_state.pending_query = goal_input.strip()
    st.rerun()

# ── Process query ─────────────────────────────────────────────────────────────
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": query})

    progress = st.progress(0)
    try:
        progress.progress(15)
        search_query = extract_search_query(query, history)
        progress.progress(35)
        courses = search_courses(search_query, top_k=10)
        progress.progress(50)
        faculty = search_faculty(search_query, top_k=10)
        progress.progress(60)
        events = search_events(search_query, top_k=10)
        progress.progress(70)
        explain_all_parallel(courses, faculty, events, query, history)
        progress.progress(95)
        response = generate_goal_response(query, courses, faculty, history, events)
        progress.progress(100)
        progress.empty()
        st.session_state.courses = courses
        st.session_state.faculty = faculty
        st.session_state.events = events
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        progress.empty()
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ Error: {str(e)}\n\nMake sure all ingest scripts have been run and chroma_db/ is committed to your repo."
        })
        st.session_state.courses = []
        st.session_state.faculty = []
        st.session_state.events = []

    st.rerun()
