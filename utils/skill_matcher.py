"""
utils/skill_matcher.py
Phase 2: Compare resume skills against job descriptions.
Produces match scores, gap analysis, and ranked job recommendations.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


# ── Core matcher ──────────────────────────────────────────────────────────────

class SkillMatcher:
    """
    Compares a parsed resume profile against job postings.
    
    Two scoring methods:
      1. Keyword overlap  → fast, deterministic, used for bulk ranking
      2. Semantic (FAISS) → used for final top-N results (Phase 3)
    
    Usage:
        matcher = SkillMatcher(resume_profile)
        results = matcher.rank_jobs(jobs_list, top_k=10)
        gap = matcher.gap_analysis(job)
    """

    # Weight multipliers per skill category (skills matter more than tools)
    CATEGORY_WEIGHTS = {
        "ml_ai":        2.5,
        "ml_frameworks":2.0,
        "languages":    1.8,
        "data":         1.5,
        "backend":      1.3,
        "cloud_devops": 1.2,
        "tools":        0.8,
    }

    def __init__(self, resume_profile: Dict):
        """
        Args:
            resume_profile: Output of resume_parser.parse_resume()
        """
        self.profile = resume_profile
        self.resume_skills = set(s.lower() for s in resume_profile.get("all_skills", []))
        self.skills_by_cat = resume_profile.get("skills_by_category", {})
        self.exp_years = resume_profile.get("experience_years") or 0

    # ── job scoring ───────────────────────────────────────────────────────────

    def score_job(self, job: Dict) -> Dict:
        """
        Score a single job against the resume.
        
        Returns:
        {
            "match_score":      float (0-100),
            "matched_skills":   List[str],
            "missing_skills":   List[str],
            "bonus_skills":     List[str],   # resume skills NOT in job desc (extras)
            "seniority_match":  bool,
            "breakdown":        Dict,
        }
        """
        job_text = _job_to_text(job).lower()
        job_skills = _extract_skills_from_text(job_text)

        matched = sorted(self.resume_skills & job_skills)
        missing = sorted(job_skills - self.resume_skills)
        bonus   = sorted(self.resume_skills - job_skills)

        # Weighted overlap score
        weighted_match = 0.0
        weighted_total = 0.0

        for skill in job_skills:
            cat = _skill_category(skill)
            w = self.CATEGORY_WEIGHTS.get(cat, 1.0)
            weighted_total += w
            if skill in self.resume_skills:
                weighted_match += w

        raw_score = (weighted_match / weighted_total * 100) if weighted_total > 0 else 0.0

        # Seniority check
        seniority_ok = _check_seniority(job.get("title", ""), self.exp_years)

        # Small boost for seniority alignment
        final_score = raw_score * (1.05 if seniority_ok else 0.95)
        final_score = min(final_score, 100.0)

        return {
            "match_score":    round(final_score, 1),
            "matched_skills": matched,
            "missing_skills": missing[:10],   # top 10 gaps only
            "bonus_skills":   bonus[:10],
            "seniority_match": seniority_ok,
            "breakdown": {
                "weighted_match": round(weighted_match, 2),
                "weighted_total": round(weighted_total, 2),
                "raw_score":      round(raw_score, 1),
            },
        }

    def rank_jobs(
        self,
        jobs: List[Dict],
        top_k: int = 10,
        min_score: float = 20.0,
    ) -> List[Tuple[Dict, Dict]]:
        """
        Rank a list of jobs by match score.
        
        Returns:
            List of (job_dict, score_dict) sorted by match_score descending
        """
        scored = []
        for job in jobs:
            score_info = self.score_job(job)
            if score_info["match_score"] >= min_score:
                scored.append((job, score_info))

        scored.sort(key=lambda x: x[1]["match_score"], reverse=True)
        logger.info(f"Ranked {len(scored)}/{len(jobs)} jobs above {min_score}% match")
        return scored[:top_k]

    def gap_analysis(self, job: Dict) -> Dict:
        """
        Detailed gap analysis for a single job.
        Useful for "how do I improve my match for this role?" feature.
        
        Returns:
        {
            "score":            float,
            "verdict":          str,      # "Strong match" / "Good match" / etc.
            "strengths":        List[str],
            "gaps":             List[str],
            "learning_priority": List[str],  # top 3 skills to learn
            "summary":          str,
        }
        """
        score_info = self.score_job(job)
        score = score_info["match_score"]
        matched = score_info["matched_skills"]
        missing = score_info["missing_skills"]

        verdict = _score_to_verdict(score)

        # Prioritize gaps by category weight
        learning_priority = sorted(
            missing,
            key=lambda s: self.CATEGORY_WEIGHTS.get(_skill_category(s), 1.0),
            reverse=True,
        )[:3]

        summary = (
            f"{verdict} ({score:.0f}% match). "
            f"You have {len(matched)} of the key skills. "
        )
        if missing:
            summary += f"Top gaps: {', '.join(learning_priority)}."
        else:
            summary += "No critical skill gaps detected."

        return {
            "score":             score,
            "verdict":           verdict,
            "strengths":         matched[:10],
            "gaps":              missing,
            "learning_priority": learning_priority,
            "summary":           summary,
            "job_title":         job.get("title", ""),
            "company":           job.get("company", ""),
        }

    def resume_summary(self) -> str:
        """One-paragraph summary of the resume profile for display."""
        cats = sorted(
            self.skills_by_cat.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )
        top_cats = [f"{cat} ({', '.join(skills[:3])})" for cat, skills in cats[:3]]
        exp_str = f"~{self.exp_years} years experience" if self.exp_years else "experience unknown"

        return (
            f"Resume has {self.profile['skill_count']} skills detected across "
            f"{len(self.skills_by_cat)} categories. {exp_str}. "
            f"Strongest areas: {'; '.join(top_cats)}."
        )


# ── helpers ───────────────────────────────────────────────────────────────────

from scraper.resume_parser import SKILL_TO_CATEGORY   # reuse ontology

def _extract_skills_from_text(text: str) -> set:
    """Extract known skills from any text (job description etc)."""
    found = set()
    for skill in SKILL_TO_CATEGORY:
        if " " in skill:
            if skill in text:
                found.add(skill)
        else:
            if re.search(r'\b' + re.escape(skill) + r'\b', text):
                found.add(skill)
    return found

def _skill_category(skill: str) -> str:
    return SKILL_TO_CATEGORY.get(skill.lower(), "tools")

def _job_to_text(job: Dict) -> str:
    return " ".join([
        job.get("title", ""),
        job.get("company", ""),
        job.get("description", ""),
    ])

def _check_seniority(title: str, exp_years: int) -> bool:
    title_lower = title.lower()
    if any(w in title_lower for w in ["senior", "sr.", "lead", "principal", "staff"]):
        return exp_years >= 4
    if any(w in title_lower for w in ["junior", "jr.", "entry", "associate", "intern"]):
        return exp_years <= 3
    return True  # mid-level or unspecified → always ok

def _score_to_verdict(score: float) -> str:
    if score >= 75:  return "Strong match"
    if score >= 55:  return "Good match"
    if score >= 35:  return "Partial match"
    return "Weak match"


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Smoke test with mock data
    mock_profile = {
        "all_skills": ["python", "pytorch", "machine learning", "fastapi",
                       "docker", "aws", "sql", "pandas", "langchain", "rag"],
        "skills_by_category": {
            "languages": ["python", "sql"],
            "ml_ai": ["machine learning", "rag", "langchain"],
            "ml_frameworks": ["pytorch"],
            "data": ["pandas"],
            "backend": ["fastapi"],
            "cloud_devops": ["docker", "aws"],
        },
        "experience_years": 3,
        "skill_count": 10,
    }

    mock_jobs = [
        {
            "title": "ML Engineer",
            "company": "OpenAI",
            "description": "Python, PyTorch, machine learning, LLM, RAG, FAISS, AWS, Docker",
        },
        {
            "title": "Senior Backend Engineer",
            "company": "Stripe",
            "description": "Python, FastAPI, PostgreSQL, Redis, Kubernetes, Go",
        },
        {
            "title": "Data Scientist",
            "company": "Airbnb",
            "description": "Python, pandas, scikit-learn, SQL, Spark, A/B testing, R",
        },
    ]

    matcher = SkillMatcher(mock_profile)
    print(matcher.resume_summary())
    print("\n── Ranked jobs ──")
    for job, score_info in matcher.rank_jobs(mock_jobs, top_k=5):
        print(f"\n[{score_info['match_score']:.0f}%] {job['title']} @ {job['company']}")
        print(f"  Matched : {score_info['matched_skills']}")
        print(f"  Missing : {score_info['missing_skills']}")

    print("\n── Gap analysis: ML Engineer ──")
    gap = matcher.gap_analysis(mock_jobs[0])
    print(gap["summary"])
    print(f"  Learn: {gap['learning_priority']}")
