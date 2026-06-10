"""
scraper/resume_parser.py
Phase 2: Parse resume PDF → extract text → extract skills → return structured profile.

Uses pdfplumber for text extraction (handles multi-column layouts better than PyPDF2).
Skill extraction via two methods:
  1. Keyword matching against curated tech skill ontology
  2. Claude API for semantic extraction (sections, experience, summary)
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Tech skill ontology ───────────────────────────────────────────────────────
# Curated by category — used for fast keyword matching
SKILL_ONTOLOGY = {
    "languages": [
        "python", "javascript", "typescript", "java", "c++", "c#", "go", "golang",
        "rust", "scala", "kotlin", "swift", "ruby", "php", "r", "matlab", "julia",
        "bash", "shell", "sql", "html", "css", "dart",
    ],
    "ml_ai": [
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "reinforcement learning", "llm", "large language model",
        "transformer", "bert", "gpt", "rag", "retrieval augmented generation",
        "fine-tuning", "finetuning", "embeddings", "vector database", "langchain",
        "hugging face", "huggingface", "openai", "anthropic", "claude", "diffusion",
        "gan", "generative ai", "prompt engineering", "mlops",
    ],
    "ml_frameworks": [
        "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "xgboost",
        "lightgbm", "catboost", "jax", "flax", "spacy", "nltk", "gensim",
        "sentence-transformers", "faiss", "chromadb", "weaviate", "pinecone",
        "onnx", "triton", "torchserve",
    ],
    "data": [
        "pandas", "numpy", "polars", "dask", "spark", "pyspark", "hadoop",
        "kafka", "airflow", "dbt", "great expectations", "feast",
        "sql", "postgresql", "mysql", "sqlite", "mongodb", "redis",
        "elasticsearch", "cassandra", "bigquery", "snowflake", "redshift",
        "databricks", "delta lake",
    ],
    "backend": [
        "fastapi", "flask", "django", "fastify", "express", "spring boot",
        "graphql", "rest", "grpc", "websocket", "celery", "rabbitmq",
        "microservices", "api design", "system design",
    ],
    "cloud_devops": [
        "aws", "gcp", "google cloud", "azure", "docker", "kubernetes", "k8s",
        "terraform", "ansible", "ci/cd", "github actions", "jenkins", "argocd",
        "helm", "prometheus", "grafana", "datadog", "sentry",
        "lambda", "ec2", "s3", "sagemaker", "vertex ai", "azure ml",
    ],
    "tools": [
        "git", "github", "gitlab", "jira", "linear", "notion",
        "jupyter", "vscode", "linux", "unix",
        "streamlit", "gradio", "plotly", "matplotlib", "seaborn",
    ],
}

# Flatten for quick lookup (skill → category)
SKILL_TO_CATEGORY: Dict[str, str] = {}
for cat, skills in SKILL_ONTOLOGY.items():
    for skill in skills:
        SKILL_TO_CATEGORY[skill.lower()] = cat


# ── PDF text extraction ───────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract full text from resume PDF.
    Uses pdfplumber — handles multi-column and tables better than PyPDF2.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pip install pdfplumber")

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    full_text = []
    with pdfplumber.open(path) as pdf:
        logger.info(f"PDF: {len(pdf.pages)} pages — {path.name}")
        for i, page in enumerate(pdf.pages):
            text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if text:
                full_text.append(text)
            else:
                logger.warning(f"Page {i+1}: no extractable text (scanned?)")

    combined = "\n".join(full_text)
    logger.info(f"Extracted {len(combined)} chars from {path.name}")
    return combined


# ── Section detector ──────────────────────────────────────────────────────────

SECTION_PATTERNS = {
    "contact":     r"(contact|email|phone|linkedin|github|location)",
    "summary":     r"(summary|objective|profile|about me|overview)",
    "experience":  r"(experience|work history|employment|career)",
    "education":   r"(education|degree|university|college|academic)",
    "skills":      r"(skills|technologies|tech stack|tools|competencies)",
    "projects":    r"(projects|portfolio|open.?source|side projects)",
    "certifications": r"(certifications?|certificates?|licenses?|credentials)",
    "publications": r"(publications?|papers?|research|patents?)",
}

def detect_sections(text: str) -> Dict[str, str]:
    """
    Split resume text into sections by heading detection.
    Returns dict of section_name → section_text.
    """
    lines = text.split("\n")
    sections: Dict[str, List[str]] = {"header": []}
    current_section = "header"

    for line in lines:
        stripped = line.strip()
        if not stripped:
            sections.setdefault(current_section, []).append("")
            continue

        # Check if line is a section heading
        matched_section = None
        for section_name, pattern in SECTION_PATTERNS.items():
            if re.search(pattern, stripped.lower()) and len(stripped) < 60:
                matched_section = section_name
                break

        if matched_section:
            current_section = matched_section
            sections.setdefault(current_section, [])
        else:
            sections.setdefault(current_section, []).append(stripped)

    # Join each section
    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


# ── Keyword skill extractor ───────────────────────────────────────────────────

def extract_skills_keyword(text: str) -> Dict[str, List[str]]:
    """
    Fast keyword matching against SKILL_ONTOLOGY.
    Returns dict of category → [skills found].
    Case-insensitive, handles multi-word skills.
    """
    text_lower = text.lower()
    found: Dict[str, List[str]] = {}

    for skill, category in SKILL_TO_CATEGORY.items():
        # Use word boundary for single-word skills, substring for multi-word
        if " " in skill:
            if skill in text_lower:
                found.setdefault(category, []).append(skill)
        else:
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                found.setdefault(category, []).append(skill)

    # Deduplicate and sort
    return {cat: sorted(set(skills)) for cat, skills in found.items()}


# ── Contact info extractor ────────────────────────────────────────────────────

def extract_contact_info(text: str) -> Dict[str, str]:
    """Extract email, phone, LinkedIn URL, GitHub URL from resume text."""
    contact = {}

    email = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email:
        contact["email"] = email.group()

    phone = re.search(r'(\+?\d[\d\s\-().]{7,}\d)', text)
    if phone:
        contact["phone"] = phone.group().strip()

    linkedin = re.search(r'linkedin\.com/in/[\w\-]+', text, re.IGNORECASE)
    if linkedin:
        contact["linkedin"] = "https://" + linkedin.group()

    github = re.search(r'github\.com/[\w\-]+', text, re.IGNORECASE)
    if github:
        contact["github"] = "https://" + github.group()

    return contact


# ── Experience year estimator ─────────────────────────────────────────────────

def estimate_experience_years(text: str) -> Optional[int]:
    """
    Rough heuristic: find year ranges in work experience section.
    e.g. "2019 - 2023", "Jan 2020 – Present" → adds up durations.
    """
    import datetime
    current_year = datetime.datetime.now().year

    year_range_pattern = re.compile(
        r'(20\d{2}|19\d{2})\s*[-–—to]+\s*(20\d{2}|19\d{2}|present|current|now)',
        re.IGNORECASE,
    )
    matches = year_range_pattern.findall(text)

    total_years = 0
    seen = set()
    for start_str, end_str in matches:
        try:
            start = int(start_str)
            end = current_year if re.search(r'present|current|now', end_str, re.IGNORECASE) else int(end_str)
            key = (start, end)
            if key not in seen and 1990 <= start <= current_year and start <= end:
                total_years += end - start
                seen.add(key)
        except ValueError:
            continue

    return total_years if total_years > 0 else None


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_resume(pdf_path: str) -> Dict:
    """
    Full resume parse pipeline.
    
    Returns:
    {
        "raw_text": str,
        "sections": {"experience": ..., "skills": ..., ...},
        "contact": {"email": ..., "linkedin": ..., ...},
        "skills_by_category": {"ml_ai": [...], "languages": [...], ...},
        "all_skills": [...],           # flat list
        "experience_years": int|None,
        "skill_count": int,
        "source_file": str,
    }
    """
    logger.info(f"Parsing resume: {pdf_path}")

    # 1. Extract text
    raw_text = extract_text_from_pdf(pdf_path)

    # 2. Detect sections
    sections = detect_sections(raw_text)

    # 3. Contact info
    contact = extract_contact_info(raw_text)

    # 4. Keyword skill extraction
    skills_by_cat = extract_skills_keyword(raw_text)
    all_skills = sorted({s for skills in skills_by_cat.values() for s in skills})

    # 5. Experience estimate
    exp_years = estimate_experience_years(
        sections.get("experience", raw_text)
    )

    profile = {
        "raw_text": raw_text,
        "sections": sections,
        "contact": contact,
        "skills_by_category": skills_by_cat,
        "all_skills": all_skills,
        "experience_years": exp_years,
        "skill_count": len(all_skills),
        "source_file": str(Path(pdf_path).name),
    }

    logger.info(
        f"Resume parsed: {len(all_skills)} skills found, "
        f"~{exp_years} yrs experience, "
        f"{len(sections)} sections detected"
    )
    return profile


def save_profile(profile: Dict, output_path: str) -> None:
    """Save parsed profile to JSON (excludes raw_text to keep file small)."""
    saveable = {k: v for k, v in profile.items() if k != "raw_text"}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(saveable, f, indent=2)
    logger.info(f"Profile saved: {output_path}")


def load_profile(profile_path: str) -> Dict:
    """Load a saved profile JSON."""
    with open(profile_path) as f:
        return json.load(f)


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m scraper.resume_parser <path_to_resume.pdf>")
        sys.exit(1)

    path = sys.argv[1]
    profile = parse_resume(path)

    print(f"\n{'='*50}")
    print(f"Resume: {profile['source_file']}")
    print(f"Skills: {profile['skill_count']} found")
    print(f"Experience: ~{profile['experience_years']} years")
    print(f"Contact: {profile['contact']}")
    print(f"\nSkills by category:")
    for cat, skills in profile["skills_by_category"].items():
        print(f"  {cat:20s}: {', '.join(skills)}")

    save_profile(profile, f"data/resume_profile.json")
    print(f"\nProfile saved to data/resume_profile.json")
