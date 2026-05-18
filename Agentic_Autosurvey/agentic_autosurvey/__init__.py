"""Agentic AutoSurvey — multi-agent LLM survey generation.

Reimplementation of "Agentic AutoSurvey: Let LLMs Survey LLMs"
(arXiv:2509.18661). Four specialized agents — Search, Cluster, Writer,
Evaluator — collaborate via a LangGraph workflow backed by Gemini.
"""

from .graph import build_graph, run_survey

__all__ = ["build_graph", "run_survey"]
