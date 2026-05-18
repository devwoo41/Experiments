"""arXiv search wrapper. Uses the official `arxiv` package."""

from __future__ import annotations

import time
from typing import Any

import arxiv

from ..state import Paper


def search_arxiv(query: str, limit: int = 40, year_min: int | None = None) -> list[Paper]:
    client = arxiv.Client(page_size=min(100, limit), delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results: list[Paper] = []
    for r in client.results(search):
        year = r.published.year if r.published else None
        if year_min and year and year < year_min:
            continue
        paper_id = r.entry_id.split("/abs/")[-1]
        results.append(Paper(
            paper_id=f"arxiv:{paper_id}",
            title=(r.title or "").strip().replace("\n", " "),
            abstract=(r.summary or "").strip().replace("\n", " "),
            authors=[a.name for a in r.authors],
            year=year,
            venue=r.journal_ref or "arXiv preprint",
            url=r.entry_id,
            source="arxiv",
            citations=None,
        ))
    time.sleep(0.5)
    return results
