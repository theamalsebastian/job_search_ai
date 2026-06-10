"""
config.py
Central config. Edit or use .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Anthropic ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Paths ────────────────────────────────────────────────────────────────────
INDEX_PATH = os.getenv("INDEX_PATH", "data/jobs_index")
RAW_JOBS_PATH = os.getenv("RAW_JOBS_PATH", "data/raw_jobs.json")
RESUME_UPLOAD_DIR = os.getenv("RESUME_UPLOAD_DIR", "data/resumes")

# ── Scraper ───────────────────────────────────────────────────────────────────
DEFAULT_QUERIES = [
    "machine learning engineer",
    "python backend engineer",
    "data scientist",
    "MLOps engineer",
    "AI engineer LLM",
]
DEFAULT_LOCATION = ""          # empty = remote/any
MAX_JOBS_PER_QUERY = 25
SCRAPE_SOURCES = ["indeed"]   # add "linkedin" once tested

# ── Embedding ────────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K_RESULTS = 10
MIN_SIMILARITY = 0.25

# ── RAG (Phase 3) ────────────────────────────────────────────────────────────
LLM_MODEL = "llama-3.3-70b-versatile"
MAX_CONTEXT_JOBS = 5           # jobs passed to LLM per query
