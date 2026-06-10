"""
utils/pipeline.py
Orchestrates scrape → embed → index pipeline.
Run directly to populate the FAISS index from fresh job data.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from scraper.indeed_scraper import scrape_jobs
from embeddings.faiss_indexer import JobIndex

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

INDEX_PATH = "data/jobs_index"
RAW_JOBS_PATH = "data/raw_jobs.json"


def run_pipeline(
    queries: List[str],
    location: str = "",
    max_per_query: int = 25,
    sources: Optional[List[str]] = None,
    incremental: bool = True,
) -> JobIndex:
    """
    Full pipeline: scrape → deduplicate → embed → index → save.

    Args:
        queries: Search queries to run
        location: Location string
        max_per_query: Jobs per query per source
        sources: ["indeed", "linkedin"] or subset
        incremental: If True and index exists, load and add new jobs only

    Returns:
        Populated JobIndex instance
    """
    if sources is None:
        sources = ["indeed", "linkedin"]

    # ── 1. load or create index ──────────────────────────────────────────────
    if incremental and Path(f"{INDEX_PATH}.index").exists():
        logger.info("Loading existing index for incremental update...")
        idx = JobIndex.load(INDEX_PATH)
        logger.info(f"Existing index: {idx.total} jobs")
    else:
        logger.info("Creating fresh index...")
        idx = JobIndex()

    # ── 2. scrape ────────────────────────────────────────────────────────────
    logger.info(f"Scraping: queries={queries}, location='{location}', sources={sources}")
    jobs = scrape_jobs(
        queries=queries,
        location=location,
        max_per_query=max_per_query,
        sources=sources,
    )
    logger.info(f"Scraped {len(jobs)} raw jobs")

    if not jobs:
        logger.warning("No jobs scraped — check network / queries")
        return idx

    # ── 3. save raw (for debugging / audit) ─────────────────────────────────
    Path("data").mkdir(exist_ok=True)
    existing_raw = []
    if Path(RAW_JOBS_PATH).exists():
        with open(RAW_JOBS_PATH) as f:
            existing_raw = json.load(f)

    merged = {j["url"]: j for j in existing_raw}
    for j in jobs:
        if j.get("url"):
            merged[j["url"]] = j
    with open(RAW_JOBS_PATH, "w") as f:
        json.dump(list(merged.values()), f, indent=2)
    logger.info(f"Raw jobs saved: {len(merged)} total in {RAW_JOBS_PATH}")

    # ── 4. embed + index ─────────────────────────────────────────────────────
    added = idx.add_jobs(jobs)
    logger.info(f"Added {added} new jobs to index (skipped {len(jobs) - added} duplicates)")

    # ── 5. persist ───────────────────────────────────────────────────────────
    idx.save(INDEX_PATH)
    logger.info(f"Index saved. Stats: {idx.stats()}")

    return idx


def load_index() -> Optional[JobIndex]:
    """Load existing index, return None if not found."""
    try:
        return JobIndex.load(INDEX_PATH)
    except FileNotFoundError:
        logger.warning("No saved index found. Run pipeline first.")
        return None


def quick_search(query: str, top_k: int = 5) -> None:
    """CLI helper: load index and print top results for a query."""
    idx = load_index()
    if not idx:
        print("Index not found. Run: python -m utils.pipeline")
        return

    print(f"\nSearching: '{query}' (top {top_k})\n" + "─" * 50)
    results = idx.search(query, top_k=top_k)

    if not results:
        print("No results found.")
        return

    for i, (job, score) in enumerate(results, 1):
        print(f"{i}. [{score:.3f}] {job['title']} @ {job['company']}")
        print(f"   📍 {job['location']}  |  🔗 {job['source']}  |  📅 {job.get('posted_date','')[:10]}")
        print(f"   {job['url']}")
        desc = job.get("description", "")[:150]
        if desc:
            print(f"   {desc}...")
        print()


if __name__ == "__main__":
    import sys

    # Default queries — edit to your target roles
    DEFAULT_QUERIES = [
        "machine learning engineer",
        "python backend engineer",
        "data scientist",
        "MLOps engineer",
        "AI engineer",
    ]

    if len(sys.argv) > 1 and sys.argv[1] == "search":
        query = " ".join(sys.argv[2:]) or "python ML engineer"
        quick_search(query)
    else:
        run_pipeline(
            queries=DEFAULT_QUERIES,
            location="",           # empty = remote/any
            max_per_query=20,
            sources=["indeed"],    # start with indeed only (more reliable)
            incremental=True,
        )
