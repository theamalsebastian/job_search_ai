"""
utils/rag_chain.py
Phase 3: RAG pipeline — natural language queries answered using Claude API.

Flow:
  user query
      ↓
  FAISS semantic search → top-K job chunks
      ↓
  (optional) resume context injected
      ↓
  Claude API prompt → grounded answer
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from groq import Groq

from embeddings.faiss_indexer import JobIndex
from utils.skill_matcher import SkillMatcher, _score_to_verdict
from config import LLM_MODEL, MAX_CONTEXT_JOBS, ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)


# ── prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert career advisor and job search assistant.
You help users find relevant job opportunities and understand how their skills align with open roles.

You have access to a curated set of job postings retrieved for the user's query.
Always ground your answers in the provided job data. Be specific, honest, and actionable.

When listing jobs:
- Always include: title, company, location, match indicators
- Highlight skill matches and gaps concisely
- Suggest concrete next steps when relevant

When answering questions about a resume vs jobs:
- Be encouraging but realistic
- Prioritize the most impactful skill gaps to address
- Keep answers focused and scannable"""


def _format_jobs_for_context(
    jobs_with_scores: List[Tuple[Dict, float]],
    resume_profile: Optional[Dict] = None,
    matcher: Optional[SkillMatcher] = None,
) -> str:
    """
    Format retrieved jobs into a context block for the LLM prompt.
    If resume + matcher provided, includes per-job match scores.
    """
    lines = []
    for i, (job, sem_score) in enumerate(jobs_with_scores, 1):
        lines.append(f"--- Job {i} ---")
        lines.append(f"Title:    {job.get('title', 'N/A')}")
        lines.append(f"Company:  {job.get('company', 'N/A')}")
        lines.append(f"Location: {job.get('location', 'N/A')}")
        lines.append(f"Source:   {job.get('source', 'N/A')}")
        lines.append(f"Posted:   {str(job.get('posted_date', ''))[:10]}")
        lines.append(f"URL:      {job.get('url', 'N/A')}")

        desc = job.get("description", "")
        if desc:
            lines.append(f"Description: {desc[:400]}{'...' if len(desc) > 400 else ''}")

        if matcher:
            gap = matcher.gap_analysis(job)
            lines.append(f"Match Score:    {gap['score']:.0f}% ({gap['verdict']})")
            if gap["strengths"]:
                lines.append(f"Skill Matches:  {', '.join(gap['strengths'][:6])}")
            if gap["gaps"]:
                lines.append(f"Skill Gaps:     {', '.join(gap['gaps'][:5])}")

        lines.append("")

    return "\n".join(lines)


def _build_user_message(
    query: str,
    jobs_context: str,
    resume_summary: Optional[str] = None,
) -> str:
    """Assemble the user message with context injected."""
    parts = []

    if resume_summary:
        parts.append(f"## My Resume Profile\n{resume_summary}\n")

    parts.append(f"## Retrieved Job Postings\n{jobs_context}")
    parts.append(f"## My Question\n{query}")

    return "\n\n".join(parts)


# ── RAG chain class ───────────────────────────────────────────────────────────

class JobRAGChain:
    """
    RAG chain for job search Q&A.
    
    Usage:
        chain = JobRAGChain(index)
        response = chain.query("What jobs match my Python and ML skills?")
        
        # With resume
        chain = JobRAGChain(index, resume_profile=profile)
        response = chain.query("What are my top matches this week?")
        response = chain.query("What skills should I learn to get a FAANG ML role?")
    """

    def __init__(
        self,
        index: JobIndex,
        resume_profile: Optional[Dict] = None,
        top_k: int = MAX_CONTEXT_JOBS,
    ):
        self.index = index
        self.resume_profile = resume_profile
        self.top_k = top_k
        self.matcher = SkillMatcher(resume_profile) if resume_profile else None
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.conversation_history: List[Dict] = []

    # ── main query entry point ────────────────────────────────────────────────

    def query(
        self,
        user_query: str,
        stream: bool = False,
        reset_history: bool = False,
    ) -> str:
        """
        Answer a job search question using RAG.
        
        Args:
            user_query: Natural language question
            stream: If True, prints tokens as they arrive and returns full text
            reset_history: Clear conversation history before this turn
        
        Returns:
            Assistant response string
        """
        if reset_history:
            self.conversation_history = []

        # 1. Semantic retrieval from FAISS
        logger.info(f"Query: {user_query!r}")
        raw_results = self.index.search(user_query, top_k=self.top_k)

        if not raw_results:
            return (
                "I couldn't find relevant jobs in the index. "
                "Try running the pipeline to refresh job data, "
                "or try a different search query."
            )

        # 2. Re-rank by skill match if resume available
        if self.matcher:
            jobs_list = [job for job, _ in raw_results]
            ranked = self.matcher.rank_jobs(jobs_list, top_k=self.top_k, min_score=0)
            # Merge semantic scores back in
            sem_score_map = {job["url"]: score for job, score in raw_results}
            jobs_with_scores = [
                (job, sem_score_map.get(job.get("url", ""), 0.0))
                for job, _ in ranked
            ]
        else:
            jobs_with_scores = raw_results

        # 3. Build context
        resume_summary = self.matcher.resume_summary() if self.matcher else None
        jobs_context = _format_jobs_for_context(
            jobs_with_scores,
            resume_profile=self.resume_profile,
            matcher=self.matcher,
        )

        user_message = _build_user_message(user_query, jobs_context, resume_summary)

        # 4. Add to conversation history (multi-turn support)
        self.conversation_history.append({"role": "user", "content": user_message})

        # 5. Call Claude API
        logger.info(f"Calling Claude API ({LLM_MODEL}) with {len(jobs_with_scores)} jobs in context")

        if stream:
            return self._stream_response()
        else:
            return self._get_response()

    def _get_response(self) -> str:
        """Non-streaming Claude API call."""
        response = self.client.chat.completions.create(            
            model=LLM_MODEL,
            max_tokens=1024,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.conversation_history,
            )
        assistant_text = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    def _stream_response(self) -> str:
        """Streaming Claude API call — prints tokens live, returns full text."""
        stream = self.client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=1024,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.conversation_history,
            stream=True,
            )
        full_text = ""
        
        for chunk in stream:
            text = chunk.choices[0].delta.content or ""
            print(text, end="", flush=True)
            full_text += text

        self.conversation_history.append({"role": "assistant", "content": full_text})
        return full_text

    # ── convenience queries ───────────────────────────────────────────────────

    def top_matches(self, n: int = 5) -> str:
        """Shortcut: 'What are my top N job matches?'"""
        return self.query(
            f"List my top {n} job matches with match scores, key matched skills, "
            f"and one sentence on why each is a good fit. Format as a numbered list.",
            reset_history=True,
        )

    def skill_gap_for_role(self, role: str) -> str:
        """Shortcut: 'What skills do I need for X role?'"""
        return self.query(
            f"What skills am I missing to be a strong candidate for a {role} role? "
            f"Use the job postings to identify the most common requirements I lack. "
            f"Suggest a prioritized learning plan.",
            reset_history=True,
        )

    def summarize_market(self) -> str:
        """Shortcut: 'What are the top requirements in current job postings?'"""
        return self.query(
            "Based on the job postings, what are the most in-demand skills and technologies "
            "right now? What patterns do you see in requirements across companies?",
            reset_history=True,
        )

    def reset(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from utils.pipeline import load_index

    idx = load_index()
    if not idx:
        print("No index found. Run: python -m utils.pipeline")
        sys.exit(1)

    # Load resume profile if available
    resume_profile = None
    try:
        from scraper.resume_parser import load_profile
        resume_profile = load_profile("data/resume_profile.json")
        print(f"Resume loaded: {resume_profile['skill_count']} skills")
    except FileNotFoundError:
        print("No resume profile found — running without resume context")

    chain = JobRAGChain(idx, resume_profile=resume_profile)

    print(f"\nIndex has {idx.total} jobs loaded.")
    print("Commands: 'top', 'market', 'gap <role>', or ask anything\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        elif user_input.lower() == "top":
            print("\nAssistant:", chain.top_matches())
        elif user_input.lower() == "market":
            print("\nAssistant:", chain.summarize_market())
        elif user_input.lower().startswith("gap "):
            role = user_input[4:].strip()
            print("\nAssistant:", chain.skill_gap_for_role(role))
        else:
            print("\nAssistant:", chain.query(user_input, stream=True))
        print()
