# 🚀 AI Job Search Assistant

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-green)
![FAISS](https://img.shields.io/badge/FAISS-Indexing-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **Live Demo:** [https://your-deployment-link.streamlit.app](https://jobsearchaibot.streamlit.app/)  
> Try it now — recruiters can search jobs in real time!

---

## 📌 Overview
The **AI Job Search Assistant** is a smart pipeline that scrapes job postings from **Indeed** and **LinkedIn**, embeds them using **Sentence Transformers**, and indexes them with **FAISS** for lightning‑fast semantic search.  

This project is designed as a **portfolio piece** to showcase:
- End‑to‑end AI pipelines (scraping → embedding → search).
- Practical use of **transformer models** in job search automation.
- Deployable, recruiter‑friendly apps with **Streamlit UI**.

---

## 🗂 Project Structure
job-search-ai/
├── scraper/                 # Indeed + LinkedIn scrapers
│   └── indeed_scraper.py
├── embeddings/              # FAISS index builder
│   └── faiss_indexer.py
├── utils/                   # Orchestration pipeline
│   └── pipeline.py
├── data/                    # Auto-created (ignored in git)
│   ├── jobs_index.index
│   ├── jobs_index.meta.json
│   └── raw_jobs.json
├── config.py                 # Settings + queries
├── requirements.txt
└── .env                      # API keys (never commit)


---

## ⚙️ Setup Instructions

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env → add ANTHROPIC_API_KEY=sk-ant-...


🔍 How It Works
1. Scraping
Indeed: RSS feed (/rss?q=...&sort=date) → reliable, structured.

LinkedIn: Guest job cards endpoint → rate‑limited, retries built in.

Deduplication by job URL across runs.

2. Embedding
Model: all-MiniLM-L6-v2 (384‑dim, ~80MB, CPU‑friendly).

Each job → title | company | location | description.

L2‑normalized embeddings → cosine similarity via FAISS.

3. Indexing
Persistent: data/jobs_index.index + .meta.json.

Incremental: adds only new jobs (deduped).

Search returns (job_dict, cosine_score) sorted by relevance.

📸 Screenshots
<img width="1919" height="964" alt="Screenshot 2026-06-10 203403" src="https://github.com/user-attachments/assets/103def64-391c-4bea-98e9-ca2d1c4785b5" />


🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change.

📄 License
MIT License — free to use, modify, and share.
