"""
app.py
------
Streamlit web UI for the Course Finder.

Run with:
    streamlit run app.py

Make sure you have run `python ingest.py` first to populate the vector DB.
"""

import streamlit as st
from agent import search_courses, format_courses_for_llm
from llm import extract_search_query, generate_response

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Course Finder",
    page_icon="🎓",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Page background */
.stApp {
    background-color: #F7F4EF;
}

/* Header */
.header-block {
    background: #1B2A4A;
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 2rem;
    color: white;
}
.header-block h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    margin: 0 0 0.4rem 0;
    color: #F0E6C8;
    letter-spacing: -0.5px;
}
.header-block p {
    font-size: 1rem;
    color: #A8BFDC;
    margin: 0;
    font-weight: 300;
}

/* Chat messages */
.user-bubble {
    background: #1B2A4A;
    color: #F7F4EF;
    border-radius: 18px 18px 4px 18px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 0 0.5rem 3rem;
    font-size: 0.95rem;
    line-height: 1.5;
}
.assistant-bubble {
    background: white;
    color: #1B2A4A;
    border-radius: 18px 18px 18px 4px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 3rem 0.5rem 0;
    font-size: 0.95rem;
    line-height: 1.6;
    border: 1px solid #E2DDD6;
    box-shadow: 0 2px 8px rgba(27,42,74,0.06);
}
.role-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    color: #8A9BB5;
}
.role-label-user {
    text-align: right;
    color: #A8BFDC;
}

/* Course pills shown in expander */
.course-pill {
    display: inline-block;
    background: #EBF0F8;
    color: #1B2A4A;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.8rem;
    font-weight: 500;
    margin-right: 0.4rem;
}

/* Result count badge */
.badge {
    background: #F0E6C8;
    color: #7A5C1E;
    border-radius: 20px;
    padding: 0.15rem 0.7rem;
    font-size: 0.78rem;
    font-weight: 500;
}

/* Input area */
.stTextInput > div > div > input {
    border-radius: 12px !important;
    border: 1.5px solid #D6D0C8 !important;
    background: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1B2A4A !important;
    box-shadow: 0 0 0 3px rgba(27,42,74,0.1) !important;
}

/* Button */
.stButton > button {
    background: #1B2A4A !important;
    color: #F0E6C8 !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.4rem !important;
    font-size: 0.95rem !important;
    transition: background 0.2s ease !important;
}
.stButton > button:hover {
    background: #2A3F6A !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #1B2A4A;
}
section[data-testid="stSidebar"] * {
    color: #A8BFDC !important;
}
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: #F0E6C8 !important;
    font-family: 'DM Serif Display', serif !important;
}
.stSlider > div {
    color: #F0E6C8 !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #1B2A4A !important;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    top_k = st.slider("Courses to retrieve", min_value=1, max_value=10, value=5)
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
1. You describe what you want to study  
2. **Mistral** extracts your search intent  
3. The **Catalog Agent** finds semantically similar courses  
4. **Mistral** crafts a personalized recommendation  
    """)
    st.markdown("---")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
    <h1>🎓 Course Finder</h1>
    <p>Describe what you want to learn and I'll find the right courses for you.</p>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="role-label role-label-user">You</div>
        <div class="user-bubble">{msg["content"]}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="role-label">Course Advisor</div>
        <div class="assistant-bubble">{msg["content"]}</div>
        """, unsafe_allow_html=True)

        # Show raw course matches in an expander
        if "courses" in msg:
            with st.expander(f"📋 View matched courses ({len(msg['courses'])})"):
                for c in msg["courses"]:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"**{c['course code']}**")
                        st.caption(f"Similarity: {1 - c['distance']:.2%}")
                    with col2:
                        st.write(c["course description"])
                        st.caption(f"🗓 {c['semester taught']}  •  👤 {c['taught by']}")
                    st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
with st.form(key="query_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            label="query",
            placeholder="e.g. I want to take a course in negotiations next semester...",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Search")

# ── Handle submission ─────────────────────────────────────────────────────────
if submitted and user_input.strip():
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Searching the catalog..."):
        try:
            # Step 1: Extract search query via Mistral
            search_query = extract_search_query(user_input)

            # Step 2: Catalog agent retrieves matching courses
            courses = search_courses(search_query, top_k=top_k)
            courses_text = format_courses_for_llm(courses)

            # Step 3: Mistral generates a response
            response = generate_response(user_input, courses_text)

        except Exception as e:
            response = f"⚠️ Something went wrong: {str(e)}\n\nMake sure Ollama is running and you've run `python ingest.py` first."
            courses = []

    # Save assistant message with course data attached
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "courses": courses,
    })

    st.rerun()
    