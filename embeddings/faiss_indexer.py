"""
embeddings/faiss_indexer.py
Embeds job postings with sentence-transformers and stores in FAISS index.
Supports incremental updates and persistent save/load.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


# ── lazy imports (heavy deps) ────────────────────────────────────────────────

def _get_faiss():
    import faiss
    return faiss

def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


# ── index manager ────────────────────────────────────────────────────────────

class JobIndex:
    """
    Wraps FAISS index + job metadata store.
    
    Flow:
        index = JobIndex()
        index.add_jobs(jobs_list)
        results = index.search("senior python ML engineer remote", top_k=5)
    
    Persistence:
        index.save("data/jobs")   → writes data/jobs.index + data/jobs.meta.json
        index = JobIndex.load("data/jobs")
    """

    EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality
    EMBED_DIM = 384

    def __init__(self, model_name: str = EMBED_MODEL):
        self.model_name = model_name
        self._model = None          # lazy load
        self._index = None          # FAISS index
        self._jobs: List[Dict] = [] # metadata store (parallel to index vectors)

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = _get_model(self.model_name)
        return self._model

    @property
    def index(self):
        if self._index is None:
            faiss = _get_faiss()
            # IndexFlatIP = inner-product (cosine after normalization)
            self._index = faiss.IndexFlatIP(self.EMBED_DIM)
            logger.info("Created new FAISS IndexFlatIP")
        return self._index

    # ── ingest ───────────────────────────────────────────────────────────────

    def _job_to_text(self, job: Dict) -> str:
        """
        Combine job fields into one rich text chunk for embedding.
        Title + company + location weighted at top for better semantic match.
        """
        parts = [
            job.get("title", ""),
            job.get("company", ""),
            job.get("location", ""),
            job.get("description", ""),
        ]
        return " | ".join(p for p in parts if p).strip()

    def add_jobs(self, jobs: List[Dict], batch_size: int = 64) -> int:
        """
        Embed and add jobs to index. Returns count added.
        Skips duplicates by URL.
        """
        existing_urls = {j.get("url") for j in self._jobs}
        new_jobs = [j for j in jobs if j.get("url") not in existing_urls]

        if not new_jobs:
            logger.info("No new jobs to add (all duplicates)")
            return 0

        texts = [self._job_to_text(j) for j in new_jobs]
        embeddings = self._embed_batch(texts, batch_size)

        # Normalize for cosine similarity via inner product
        faiss = _get_faiss()
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self._jobs.extend(new_jobs)

        logger.info(f"Added {len(new_jobs)} jobs. Index total: {self.index.ntotal}")
        return len(new_jobs)

    def _embed_batch(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Embed texts in batches. Returns float32 numpy array."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Embedding batch {i//batch_size + 1} ({len(batch)} texts)...")
            embs = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,  # we normalize manually
            )
            all_embeddings.append(embs)
        return np.vstack(all_embeddings).astype("float32")

    # ── search ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.3,
        filter_source: Optional[str] = None,
    ) -> List[Tuple[Dict, float]]:
        """
        Semantic search over indexed jobs.
        
        Args:
            query: Natural language query or resume snippet
            top_k: Number of results
            min_score: Cosine similarity threshold (0-1)
            filter_source: Optionally filter by "indeed" or "linkedin"
        
        Returns:
            List of (job_dict, score) tuples sorted by relevance
        """
        if self.index.ntotal == 0:
            logger.warning("Index empty — no jobs loaded yet")
            return []

        faiss = _get_faiss()
        query_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype("float32")
        faiss.normalize_L2(query_emb)

        # Search more than top_k if filtering, to get enough after filter
        k = min(top_k * 3 if filter_source else top_k, self.index.ntotal)
        scores, indices = self.index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < min_score:
                continue
            job = self._jobs[idx]
            if filter_source and job.get("source") != filter_source:
                continue
            results.append((job, float(score)))
            if len(results) >= top_k:
                break

        return results

    def search_by_skills(self, skills: List[str], top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Convenience: search using extracted skill list as query.
        """
        query = ", ".join(skills)
        return self.search(f"job requiring skills: {query}", top_k=top_k)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path_prefix: str) -> None:
        """
        Save index + metadata.
        Creates <path_prefix>.index and <path_prefix>.meta.json
        """
        faiss = _get_faiss()
        Path(path_prefix).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{path_prefix}.index")
        with open(f"{path_prefix}.meta.json", "w") as f:
            json.dump({
                "model_name": self.model_name,
                "total": len(self._jobs),
                "jobs": self._jobs,
            }, f, indent=2)
        logger.info(f"Saved index to {path_prefix}.index ({len(self._jobs)} jobs)")

    @classmethod
    def load(cls, path_prefix: str) -> "JobIndex":
        """Load previously saved index."""
        faiss = _get_faiss()
        meta_path = f"{path_prefix}.meta.json"
        index_path = f"{path_prefix}.index"

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}")

        with open(meta_path) as f:
            meta = json.load(f)

        instance = cls(model_name=meta.get("model_name", cls.EMBED_MODEL))
        instance._index = faiss.read_index(index_path)
        instance._jobs = meta.get("jobs", [])
        logger.info(f"Loaded index: {instance._jobs.__len__()} jobs")
        return instance

    # ── utils ────────────────────────────────────────────────────────────────

    @property
    def total(self) -> int:
        return len(self._jobs)

    def get_all_jobs(self) -> List[Dict]:
        return list(self._jobs)

    def stats(self) -> Dict:
        sources = {}
        for j in self._jobs:
            s = j.get("source", "unknown")
            sources[s] = sources.get(s, 0) + 1
        return {
            "total_jobs": self.total,
            "index_vectors": self.index.ntotal,
            "by_source": sources,
            "model": self.model_name,
        }


if __name__ == "__main__":
    # Smoke test with mock data
    mock_jobs = [
        {
            "title": "Senior ML Engineer",
            "company": "OpenAI",
            "location": "San Francisco, CA",
            "url": "https://example.com/job1",
            "description": "Python, PyTorch, transformers, LLM fine-tuning, RAG pipelines",
            "posted_date": "2024-06-01",
            "source": "indeed",
        },
        {
            "title": "Backend Python Engineer",
            "company": "Stripe",
            "location": "Remote",
            "url": "https://example.com/job2",
            "description": "Python, FastAPI, PostgreSQL, Redis, distributed systems",
            "posted_date": "2024-06-02",
            "source": "linkedin",
        },
        {
            "title": "Data Scientist",
            "company": "Airbnb",
            "location": "New York, NY",
            "url": "https://example.com/job3",
            "description": "Python, pandas, scikit-learn, SQL, A/B testing, Spark",
            "posted_date": "2024-06-03",
            "source": "indeed",
        },
    ]

    idx = JobIndex()
    added = idx.add_jobs(mock_jobs)
    print(f"Added: {added} jobs")
    print(f"Stats: {idx.stats()}")

    results = idx.search("machine learning python pytorch", top_k=3)
    print("\nSearch: 'machine learning python pytorch'")
    for job, score in results:
        print(f"  [{score:.3f}] {job['title']} @ {job['company']}")

    idx.save("data/test_index")
    print("\nSaved. Loading back...")
    idx2 = JobIndex.load("data/test_index")
    print(f"Loaded stats: {idx2.stats()}")
