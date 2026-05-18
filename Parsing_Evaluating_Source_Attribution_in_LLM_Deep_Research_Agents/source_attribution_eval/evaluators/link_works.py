"""Evaluator 1 — Link Works (paper §3.3.1).

  "Link Works assesses URL accessibility without LLM inference. For each
   cited URL, a web content extractor capable of handling JavaScript-rendered
   pages retrieves the content. The evaluator produces a binary score of 1 if
   the URL returns accessible content and 0 if the request fails due to HTTP
   errors (404, 403), timeouts, or blocked access."

Side-effect: when the link works, we also stash the extracted main text on
the Citation record so the LLM judges downstream do not need to re-fetch.
"""

from __future__ import annotations

from typing import Any

from ..state import Citation
from ..tools import extract_main_text, http_probe


def link_works(citation: Citation, *, timeout: int, user_agent: str) -> tuple[int, str, str]:
    """Returns (score, reason, extracted_text)."""
    url = citation.get("url") or ""
    if not url:
        return 0, "empty_url", ""
    try:
        status, html = http_probe(url, timeout=timeout, user_agent=user_agent)
    except Exception as e:                          # pragma: no cover
        return 0, f"exception:{type(e).__name__}", ""

    if status == 0:
        return 0, "connection_error_or_timeout", ""
    if status >= 400:
        return 0, f"http_{status}", ""

    # Trafilatura extraction
    text = extract_main_text(html, url=url)
    if not text.strip():
        # URL responded OK but no extractable content (paywall/JS-only/etc.)
        return 0, f"http_{status}_no_content", ""
    return 1, f"http_{status}", text
