"""Shared types for the pipeline.

Matches the paper's AttributionDocument structure (§3.2): citations,
attributions, and (later) per-pair evaluation results.
"""

from __future__ import annotations

from typing import Any, TypedDict


class Citation(TypedDict, total=False):
    """A unique cited source extracted from the report."""
    citation_id: str          # canonical id, e.g. "c1"
    raw_labels: list[str]     # e.g. ["[1]", "[1-3]"]
    url: str                  # canonical URL
    url_content: str | None   # populated by web fetch


class Attribution(TypedDict, total=False):
    """A single claim span with one or more citation references."""
    attribution_id: str       # e.g. "a7"
    text: str                 # original sentence text (citations included)
    text_nocite: str          # text with citation markers stripped
    citation_ids: list[str]   # ids referenced by this attribution


class PairEval(TypedDict, total=False):
    """Evaluation result for one (attribution, citation) pair."""
    attribution_id: str
    citation_id: str
    link_works: int                              # 0 or 1
    link_works_reason: str
    relevant_content: int                        # 0 or 1
    relevant_content_explanation: str
    fact_check: int                              # 0 or 1
    fact_check_explanation: str
    error: str | None


class AttributionDocument(TypedDict, total=False):
    """Output of Phase 1, populated through Phase 2 (paper §3.2)."""
    query_id: str
    query: str
    model: str
    depth: str
    raw_markdown: str
    citations: list[Citation]
    attributions: list[Attribution]
    evals: list[PairEval]


class PipelineState(TypedDict, total=False):
    """Top-level LangGraph state for a single (query, model, depth) cell."""
    query_id: str
    query: str
    model: str
    depth: str                 # "brief" | "moderate" | "extensive"
    config: dict[str, Any]
    document: AttributionDocument
    logs: list[str]
