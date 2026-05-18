"""Shared state schema for the LangGraph pipeline.

Field names track the paper's expected agent outputs (Appendix K).
"""

from __future__ import annotations

from typing import Any, TypedDict


class Paper(TypedDict, total=False):
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    year: int | None
    venue: str | None
    url: str
    source: str               # "arxiv" | "semantic_scholar"
    citations: int | None
    embedding: list[float] | None
    cluster_id: int | None
    cluster_confidence: float | None
    citation_key: str | None  # in-text [Author, Year] string


class Cluster(TypedDict, total=False):
    cluster_id: int
    name: str                 # TF-IDF generated
    key_terms: list[str]
    paper_ids: list[str]
    size: int


class ClusterRelationship(TypedDict):
    a: int
    b: int
    strength: float           # cosine similarity of centroids


class ClusterQualityMetrics(TypedDict, total=False):
    k_selected: int
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    k_candidates: dict[int, float]   # K -> silhouette


class SectionDraft(TypedDict):
    cluster_id: int | None
    kind: str                 # "intro" | "cluster" | "cross_cutting" | "future" | "conclusion"
    title: str
    content: str


class DimensionScore(TypedDict, total=False):
    score: float
    weight: float
    category: str
    justification: str
    metrics: str
    specific_examples: list[str]


class Evaluation(TypedDict, total=False):
    dimensional_scores: dict[str, DimensionScore]
    overall_assessment: dict[str, Any]      # weighted_total_score, score_breakdown, quality_level, publication_readiness
    comparison_to_standards: dict[str, str]
    strengths: list[str]
    weaknesses: list[str]
    prioritized_recommendations: list[dict[str, str]]
    executive_summary: str


class SurveyState(TypedDict, total=False):
    # Inputs
    topic: str
    config: dict[str, Any]

    # Stage 1 — Search
    search_queries: list[str]
    raw_papers: list[Paper]
    papers: list[Paper]
    search_statistics: dict[str, int]

    # Stage 2 — Cluster
    clusters: list[Cluster]
    cluster_relationships: list[ClusterRelationship]
    cluster_quality_metrics: ClusterQualityMetrics
    outliers: list[str]                      # paper_ids of low-confidence papers

    # Stage 3 — Writer
    abstract_text: str
    sections: list[SectionDraft]             # ordered: intro -> cluster*N -> cross_cutting -> future -> conclusion
    references: list[Paper]                  # cited papers, in appearance order

    # Stage 4 — Evaluator
    evaluation: Evaluation

    # Logs
    logs: list[str]
