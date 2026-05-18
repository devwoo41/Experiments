"""Agent 1 — Paper Search Specialist (paper §2.2.1, K.1).

Per paper:
  1. Generate 20-30 diverse search queries including:
     - Core keyword as-is
     - Synonyms and variations
     - Related technical terms
     - Compound queries with AND/OR operators
     - Acronym expansions/contractions
  2. Search using both Semantic Scholar and arXiv APIs
  3. Deduplicate results (90% title similarity threshold)
  4. Filter for papers from 2020-2025
  5. Ensure abstracts are complete
  6. Target 100-150 papers for comprehensive coverage
"""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz

from ..llm import GeminiLLM
from ..state import Paper, SurveyState
from ..tools import search_arxiv, search_semantic_scholar


SEARCH_SYSTEM = (
    "You are the Paper Search Specialist subagent in the Agentic AutoSurvey "
    "pipeline (arXiv:2509.18661). You expand a topic into a diverse, "
    "high-recall query set and return structured JSON only."
)

# Verbatim from Appendix K.1, with literal placeholders substituted at call time.
QUERY_EXPANSION_PROMPT = """Search for academic papers on the topic: {topic}

Your task:
1. Generate {n_queries} diverse search queries including:
   - Core keyword as-is
   - Synonyms and variations
   - Related technical terms
   - Compound queries with AND/OR operators
   - Acronym expansions/contractions

Return STRICT JSON with this shape:
{{
  "search_queries": ["query 1", "query 2", ...]
}}
No commentary, no fences."""


def _expand_queries(llm: GeminiLLM, topic: str, n_queries: int) -> list[str]:
    parsed = llm.generate_json(
        QUERY_EXPANSION_PROMPT.format(topic=topic, n_queries=n_queries),
        tier="light",
        system=SEARCH_SYSTEM,
    )
    queries: list[str] = []
    if isinstance(parsed, dict):
        queries = [str(q) for q in parsed.get("search_queries", []) if isinstance(q, str)]
    elif isinstance(parsed, list):
        queries = [str(q) for q in parsed if isinstance(q, str)]
    if topic not in queries:
        queries.insert(0, topic)
    return queries[:n_queries]


def _dedupe_by_title_similarity(papers: list[Paper], threshold: int) -> list[Paper]:
    """Drop papers whose title fuzzy-ratio with an already-kept title >= threshold (paper: 90%)."""
    kept: list[Paper] = []
    kept_titles: list[str] = []
    for p in papers:
        title = (p.get("title") or "").strip().lower()
        if not title:
            continue
        dup_idx = -1
        for i, t in enumerate(kept_titles):
            if fuzz.token_set_ratio(title, t) >= threshold:
                dup_idx = i
                break
        if dup_idx == -1:
            kept.append(p)
            kept_titles.append(title)
        else:
            existing = kept[dup_idx]
            # Prefer the record with higher citations or longer abstract
            if (p.get("citations") or 0) > (existing.get("citations") or 0) or \
               len(p.get("abstract") or "") > len(existing.get("abstract") or ""):
                kept[dup_idx] = p
                kept_titles[dup_idx] = title
    return kept


def _filter(papers: list[Paper], year_min: int, year_max: int) -> list[Paper]:
    """Year range filter + abstract completeness check (paper §2.2.1)."""
    out: list[Paper] = []
    for p in papers:
        year = p.get("year")
        if year is not None and not (year_min <= year <= year_max):
            continue
        if not (p.get("abstract") or "").strip():
            continue
        out.append(p)
    return out


def _rank(papers: list[Paper]) -> list[Paper]:
    return sorted(
        papers,
        key=lambda p: ((p.get("citations") or 0), (p.get("year") or 0)),
        reverse=True,
    )


def search_node(state: SurveyState, llm: GeminiLLM) -> dict[str, Any]:
    cfg = state["config"]["search"]
    topic = state["topic"]

    queries = _expand_queries(llm, topic, cfg["num_query_expansions"])

    raw: list[Paper] = []
    per_query = cfg["per_query_limit"]
    year_min = cfg.get("year_min", 2020)
    year_max = cfg.get("year_max", 2025)
    for q in queries:
        if "arxiv" in cfg["sources"]:
            try:
                raw.extend(search_arxiv(q, limit=per_query, year_min=year_min))
            except Exception:
                pass
        if "semantic_scholar" in cfg["sources"]:
            try:
                raw.extend(search_semantic_scholar(q, limit=per_query, year_min=year_min))
            except Exception:
                pass

    filtered = _filter(raw, year_min=year_min, year_max=year_max)
    deduped = _dedupe_by_title_similarity(filtered, threshold=cfg["title_similarity_threshold"])
    ranked = _rank(deduped)

    target = cfg["target_papers"]
    upper = cfg["max_papers"]
    final = ranked[:upper]
    if len(final) > target:
        final = final[:target]

    stats = {
        "queries_used": len(queries),
        "raw_hits": len(raw),
        "after_filter": len(filtered),
        "after_dedupe": len(deduped),
        "final": len(final),
    }
    log = (
        f"[Search] {stats['queries_used']} queries -> {stats['raw_hits']} raw "
        f"-> {stats['after_filter']} year/abstract-filtered "
        f"-> {stats['after_dedupe']} unique (90% title sim) "
        f"-> kept {stats['final']}"
    )
    return {
        "search_queries": queries,
        "raw_papers": raw,
        "papers": final,
        "search_statistics": stats,
        "logs": state.get("logs", []) + [log],
    }
