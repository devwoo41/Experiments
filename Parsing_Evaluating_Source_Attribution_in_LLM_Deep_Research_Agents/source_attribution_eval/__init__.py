"""Source Attribution Evaluation Framework.

Reimplementation of "Cited but Not Verified: Parsing and Evaluating Source
Attribution in LLM Deep Research Agents" (arXiv:2605.06635, Onweller et al.
2026). Implements the 3-phase pipeline from Algorithm 1:
  Phase 0  Report generation by a deep research agent.
  Phase 1  Markdown AST parsing -> AttributionDocument(citations, attributions).
  Phase 2  Three evaluators (Link Works, Relevant Content, Fact Check) score
           each attribution-citation pair in parallel.
"""

from .graph import build_graph, run_evaluation

__all__ = ["build_graph", "run_evaluation"]
