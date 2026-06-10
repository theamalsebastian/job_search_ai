"""
app.py
Phase 4: Streamlit UI for the AI Job Search Assistant.

Run: streamlit run app.py
"""

import os
import json
import tempfile
import logging
from pathlib import Path
from typing import Optional

import streamlit as st

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Job Search Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── lazy imports (heavy deps load after page config) ─────────────────────────
@st.cache_resource(show_spinner="Loading job index...")
def load_job_index():
    from utils.pipeline import load_index
    return load_index()

@st.cache_resource(show_spinner="Loading embedding model...")
def get_rag_chain(resume_json: Optional[str] = None):
    from utils.rag_chain import JobRAGChain
    idx = load_job_index()
    if idx is None:
        return None
    profile = json.loads(resume_json) if resume_json else None
    return JobRAGChain(idx, resume_profile=profile)


# ── session state defaults ────────────────────────────────────────────────────
def init_session():
    defaults = {
        "messages": [],
        "resume_profile": None,
        "resume_json": None,
        "chain": None,
        "index_stats": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Job Search AI")
    st.caption("RAG-powered career assistant")
    st.divider()

    # ── index status ──────────────────────────────────────────────────────────
    st.subheader("📦 Job Index")
    idx = load_job_index()
    if idx:
        stats = idx.stats()
        st.success(f"✅ {stats['total_jobs']} jobs indexed")
        st.caption(f"Sources: {stats['by_source']}")
        if st.button("🔄 Refresh Index", use_container_width=True):
            with st.spinner("Scraping fresh jobs..."):
                from utils.pipeline import run_pipeline
                from config import DEFAULT_QUERIES, DEFAULT_LOCATION, MAX_JOBS_PER_QUERY, SCRAPE_SOURCES
                run_pipeline(
                    queries=DEFAULT_QUERIES,
                    location=DEFAULT_LOCATION,
                    max_per_query=MAX_JOBS_PER_QUERY,
                    sources=SCRAPE_SOURCES,
                    incremental=True,
                )
                load_job_index.clear()
                st.rerun()
    else:
        st.warning("⚠️ No index found")
        if st.button("🚀 Build Index Now", use_container_width=True):
            with st.spinner("Scraping and indexing jobs (this takes ~2 min)..."):
                from utils.pipeline import run_pipeline
                from config import DEFAULT_QUERIES, DEFAULT_LOCATION, MAX_JOBS_PER_QUERY, SCRAPE_SOURCES
                run_pipeline(
                    queries=DEFAULT_QUERIES,
                    location=DEFAULT_LOCATION,
                    max_per_query=MAX_JOBS_PER_QUERY,
                    sources=SCRAPE_SOURCES,
                )
                load_job_index.clear()
                st.rerun()

    st.divider()

    # ── resume upload ─────────────────────────────────────────────────────────
    st.subheader("📄 Your Resume")
    uploaded = st.file_uploader(
        "Upload PDF resume",
        type=["pdf"],
        help="Upload your resume to enable personalized job matching",
    )

    if uploaded:
        with st.spinner("Parsing resume..."):
            from scraper.resume_parser import parse_resume, save_profile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                profile = parse_resume(tmp_path)
                save_profile(profile, "data/resume_profile.json")
                # Store serializable version (no raw_text)
                serializable = {k: v for k, v in profile.items() if k != "raw_text"}
                st.session_state.resume_profile = serializable
                st.session_state.resume_json = json.dumps(serializable)
                st.session_state.chain = None  # force chain rebuild
            finally:
                os.unlink(tmp_path)

    if st.session_state.resume_profile:
        p = st.session_state.resume_profile
        st.success(f"✅ {p['source_file']}")
        st.metric("Skills detected", p["skill_count"])
        if p.get("experience_years"):
            st.metric("Est. experience", f"~{p['experience_years']} yrs")
        with st.expander("Skills breakdown"):
            for cat, skills in p.get("skills_by_category", {}).items():
                st.caption(f"**{cat}**: {', '.join(skills)}")
    else:
        st.info("No resume uploaded — matching will use semantic search only")

    st.divider()

    # ── quick actions ─────────────────────────────────────────────────────────
    st.subheader("⚡ Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏆 Top Matches", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Show me my top 5 job matches with scores and key matched skills."
            })
            st.rerun()
    with col2:
        if st.button("📊 Market Summary", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Summarize the most in-demand skills in current job postings."
            })
            st.rerun()

    if st.button("🎯 Skill Gap Analysis", use_container_width=True):
        st.session_state.messages.append({
            "role": "user",
            "content": "What are the biggest skill gaps between my resume and the top job requirements? Give me a prioritized learning plan."
        })
        st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.chain:
            st.session_state.chain.reset()
        st.rerun()


# ── main content area ─────────────────────────────────────────────────────────
st.title("AI Job Search Assistant 🤖")
st.caption("Ask anything about jobs, your resume match, or career strategy")

# ── example prompts (shown when chat is empty) ────────────────────────────────
if not st.session_state.messages:
    st.markdown("### 💡 Try asking:")
    examples = [
        "What jobs match my Python and machine learning skills?",
        "Find remote ML engineer roles posted this week",
        "How do I improve my match for senior AI engineer roles?",
        "Compare my skills against FAANG job requirements",
        "What skills should I learn for an MLOps position?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(ex, use_container_width=True, key=f"ex_{i}"):
                st.session_state.messages.append({"role": "user", "content": ex})
                st.rerun()
    st.divider()

# ── chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about jobs, skills, or your resume match..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# ── process latest user message ───────────────────────────────────────────────
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_msg = st.session_state.messages[-1]["content"]

    # Build/reuse chain
    if not st.session_state.chain:
        chain = get_rag_chain(st.session_state.resume_json)
        st.session_state.chain = chain
    else:
        chain = st.session_state.chain

    if chain is None:
        with st.chat_message("assistant"):
            st.warning("Job index not loaded. Build the index first using the sidebar.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching jobs and thinking..."):
                response = chain.query(user_msg)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


# ── job browser tab (below chat) ──────────────────────────────────────────────
if idx and idx.total > 0:
    with st.expander(f"📋 Browse All Jobs ({idx.total} indexed)", expanded=False):
        all_jobs = idx.get_all_jobs()

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            sources = ["All"] + list({j.get("source", "unknown") for j in all_jobs})
            source_filter = st.selectbox("Source", sources)
        with col2:
            search_filter = st.text_input("Filter by keyword", placeholder="e.g. remote, senior")
        with col3:
            sort_by = st.selectbox("Sort by", ["Recent", "Title", "Company"])

        # Apply filters
        filtered = all_jobs
        if source_filter != "All":
            filtered = [j for j in filtered if j.get("source") == source_filter]
        if search_filter:
            kw = search_filter.lower()
            filtered = [j for j in filtered if kw in (j.get("title","") + j.get("company","") + j.get("description","")).lower()]

        # Sort
        if sort_by == "Title":
            filtered = sorted(filtered, key=lambda j: j.get("title",""))
        elif sort_by == "Company":
            filtered = sorted(filtered, key=lambda j: j.get("company",""))
        else:
            filtered = sorted(filtered, key=lambda j: j.get("posted_date",""), reverse=True)

        st.caption(f"Showing {len(filtered)} jobs")

        for job in filtered[:50]:  # cap at 50 for performance
            score_badge = ""
            if st.session_state.resume_profile:
                from utils.skill_matcher import SkillMatcher, _score_to_verdict
                m = SkillMatcher(st.session_state.resume_profile)
                s = m.score_job(job)
                score_badge = f" · **{s['match_score']:.0f}% match**"

            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(
                        f"**{job.get('title','N/A')}** at {job.get('company','N/A')}"
                        f"{score_badge}  \n"
                        f"📍 {job.get('location','N/A')} · "
                        f"🗓 {str(job.get('posted_date',''))[:10]} · "
                        f"🔗 [{job.get('source','')}]({job.get('url','')})"
                    )
                with col2:
                    if job.get("url"):
                        st.link_button("Apply →", job["url"])
            st.divider()
