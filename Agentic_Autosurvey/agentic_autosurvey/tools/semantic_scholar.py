"""Semantic Scholar Graph API wrapper.

Unauthenticated calls are heavily rate-limited (~1 req/sec); a free
SEMANTIC_SCHOLAR_API_KEY raises that ceiling significantly.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ..state import Paper


BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,authors,year,venue,citationCount,externalIds,openAccessPdf,url"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    headers = {}
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if key:
        headers["x-api-key"] = key
    r = requests.get(url, params=params, headers=headers, timeout=30)
    if r.status_code == 429:
        # Honour rate-limit hint, then let tenacity retry
        time.sleep(5)
        r.raise_for_status()
    r.raise_for_status()
    return r.json()


def search_semantic_scholar(query: str, limit: int = 40, year_min: int | None = None) -> list[Paper]:
    params: dict[str, Any] = {"query": query, "limit": min(limit, 100), "fields": FIELDS}
    if year_min:
        params["year"] = f"{year_min}-"
    data = _get(BASE, params)
    out: list[Paper] = []
    for item in data.get("data", []):
        ext = item.get("externalIds") or {}
        doi = ext.get("DOI")
        arxiv_id = ext.get("ArXiv")
        if arxiv_id:
            pid = f"arxiv:{arxiv_id}"
        elif doi:
            pid = f"doi:{doi}"
        else:
            pid = f"s2:{item.get('paperId')}"
        abstract = item.get("abstract") or ""
        title = (item.get("title") or "").strip()
        if not abstract or not title:
            # Skip records without minimum content for downstream stages
            continue
        out.append(Paper(
            paper_id=pid,
            title=title,
            abstract=abstract,
            authors=[a.get("name", "") for a in (item.get("authors") or [])],
            year=item.get("year"),
            venue=item.get("venue"),
            url=item.get("url") or "",
            source="semantic_scholar",
            citations=item.get("citationCount"),
        ))
    # Conservative pacing for unauthenticated callers
    time.sleep(1.0 if not os.environ.get("SEMANTIC_SCHOLAR_API_KEY") else 0.1)
    return out
