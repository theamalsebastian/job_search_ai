"""
scraper/indeed_scraper.py
Fetches jobs from Indeed via RSS feed.
Indeed RSS: https://www.indeed.com/rss?q=<query>&l=<location>&sort=date
"""

import feedparser
import requests
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_indeed_rss(
    query: str,
    location: str = "",
    max_results: int = 50,
    fromage: int = 7,         # days old max
) -> List[Dict]:
    """
    Fetch jobs from Indeed RSS feed.
    
    Args:
        query: Job search query (e.g. "machine learning engineer")
        location: City/state (e.g. "San Francisco, CA") — leave empty for remote
        max_results: Max jobs to return
        fromage: Max age in days
    
    Returns:
        List of job dicts with keys: title, company, location, url, description, posted_date, source
    """
    base_url = "https://www.indeed.com/rss"
    params = {
        "q": query,
        "l": location,
        "sort": "date",
        "fromage": fromage,
        "limit": min(max_results, 50),  # Indeed caps at 50/request
    }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }

    # Build URL manually for feedparser (handles auth headers better)
    query_string = "&".join(f"{k}={requests.utils.quote(str(v))}" for k, v in params.items())
    url = f"{base_url}?{query_string}"

    logger.info(f"Fetching Indeed RSS: {url}")

    try:
        feed = feedparser.parse(url, request_headers=headers)
    except Exception as e:
        logger.error(f"RSS parse error: {e}")
        return []

    if feed.bozo:
        logger.warning(f"Feed parse warning: {feed.bozo_exception}")

    jobs = []
    for entry in feed.entries[:max_results]:
        description = _clean_html(entry.get("summary", ""))
        jobs.append({
            "title": entry.get("title", "").strip(),
            "company": _extract_company(entry),
            "location": entry.get("location", location),
            "url": entry.get("link", ""),
            "description": description,
            "posted_date": _parse_date(entry.get("published", "")),
            "source": "indeed",
            "query_used": query,
        })

    logger.info(f"Indeed: fetched {len(jobs)} jobs for '{query}'")
    return jobs


def fetch_linkedin_rss(
    query: str,
    location: str = "",
    max_results: int = 50,
) -> List[Dict]:
    """
    Fetch jobs from LinkedIn Jobs RSS (public feed).
    LinkedIn RSS: https://www.linkedin.com/jobs/search/?keywords=<q>&location=<loc>&f_TPR=r604800
    
    NOTE: LinkedIn's RSS support is limited. This uses their public job search
    URL and parses the page. Rate-limit aggressively to avoid blocks.
    """
    base_url = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
    params = {
        "keywords": query,
        "location": location,
        "f_TPR": "r604800",   # last 7 days
        "position": 1,
        "pageNum": 0,
        "start": 0,
    }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    jobs = []
    start = 0
    batch = 25

    while len(jobs) < max_results:
        params["start"] = start
        try:
            resp = requests.get(base_url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"LinkedIn fetch error at start={start}: {e}")
            break

        soup = BeautifulSoup(resp.text, "lxml")
        cards = soup.find_all("div", class_="base-card")

        if not cards:
            logger.info("LinkedIn: no more cards found")
            break

        for card in cards:
            job = _parse_linkedin_card(card, query)
            if job:
                jobs.append(job)

        logger.info(f"LinkedIn: fetched {len(jobs)} jobs so far for '{query}'")
        start += batch
        time.sleep(1.5)  # polite delay

    logger.info(f"LinkedIn: total {len(jobs)} jobs for '{query}'")
    return jobs[:max_results]


# ── helpers ─────────────────────────────────────────────────────────────────

def _clean_html(raw: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    if not raw:
        return ""
    soup = BeautifulSoup(raw, "lxml")
    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def _extract_company(entry) -> str:
    """Try to pull company name from Indeed RSS entry."""
    # Indeed embeds company in title as "Job Title - Company"
    title = entry.get("title", "")
    if " - " in title:
        parts = title.rsplit(" - ", 1)
        if len(parts) == 2:
            return parts[1].strip()
    return entry.get("author", "Unknown")


def _parse_date(date_str: str) -> str:
    """Normalize RSS date string to ISO format."""
    if not date_str:
        return datetime.utcnow().isoformat()
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(date_str).isoformat()
    except Exception:
        return date_str


def _parse_linkedin_card(card, query: str) -> Optional[Dict]:
    """Parse a LinkedIn job card HTML element."""
    try:
        title_el = card.find("h3", class_="base-search-card__title")
        company_el = card.find("h4", class_="base-search-card__subtitle")
        location_el = card.find("span", class_="job-search-card__location")
        link_el = card.find("a", class_="base-card__full-link")
        date_el = card.find("time")

        title = title_el.get_text(strip=True) if title_el else ""
        company = company_el.get_text(strip=True) if company_el else ""
        location = location_el.get_text(strip=True) if location_el else ""
        url = link_el["href"].split("?")[0] if link_el else ""
        posted_date = date_el.get("datetime", "") if date_el else ""

        if not title:
            return None

        return {
            "title": title,
            "company": company,
            "location": location,
            "url": url,
            "description": "",  # requires separate fetch — done in enricher
            "posted_date": posted_date,
            "source": "linkedin",
            "query_used": query,
        }
    except Exception as e:
        logger.warning(f"LinkedIn card parse error: {e}")
        return None


# ── multi-query runner ───────────────────────────────────────────────────────

def scrape_jobs(
    queries: List[str],
    location: str = "",
    max_per_query: int = 25,
    sources: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Run multiple queries across enabled sources.
    
    Args:
        queries: List of job search terms
        location: Location string (empty = remote/any)
        max_per_query: Jobs per query per source
        sources: ["indeed", "linkedin"] or subset
    
    Returns:
        Deduplicated list of all jobs
    """
    if sources is None:
        sources = ["indeed", "linkedin"]

    all_jobs = []
    seen_urls = set()

    for query in queries:
        if "indeed" in sources:
            jobs = fetch_indeed_rss(query, location, max_per_query)
            for job in jobs:
                if job["url"] not in seen_urls:
                    seen_urls.add(job["url"])
                    all_jobs.append(job)
            time.sleep(1)

        if "linkedin" in sources:
            jobs = fetch_linkedin_rss(query, location, max_per_query)
            for job in jobs:
                if job["url"] not in seen_urls:
                    seen_urls.add(job["url"])
                    all_jobs.append(job)
            time.sleep(2)

    logger.info(f"Total unique jobs scraped: {len(all_jobs)}")
    return all_jobs


if __name__ == "__main__":
    # Quick test run
    queries = ["machine learning engineer", "python backend engineer"]
    jobs = scrape_jobs(queries, location="", max_per_query=10, sources=["indeed"])
    for j in jobs[:3]:
        print(f"\n{j['title']} @ {j['company']}")
        print(f"  {j['location']} | {j['source']} | {j['posted_date'][:10]}")
        print(f"  {j['url']}")
        print(f"  Desc: {j['description'][:120]}...")
