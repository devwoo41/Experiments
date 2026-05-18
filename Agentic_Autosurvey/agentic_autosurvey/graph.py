"""LangGraph wiring for Agentic AutoSurvey.

The graph is a linear pipeline:

    START -> search -> cluster -> writer -> evaluator -> END

LangGraph still buys us: (a) typed shared state, (b) per-node logging /
checkpointing, (c) trivial extension to feedback loops (e.g. re-running
the writer if the evaluator score is below threshold) later on.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .agents import cluster_node, evaluator_node, search_node, writer_node
from .llm import GeminiLLM, LLMConfig
from .state import SurveyState


def build_graph(llm: GeminiLLM):
    g = StateGraph(SurveyState)

    g.add_node("search", lambda s: search_node(s, llm))
    g.add_node("cluster", lambda s: cluster_node(s, llm))
    g.add_node("writer", lambda s: writer_node(s, llm))
    g.add_node("evaluator", lambda s: evaluator_node(s, llm))

    g.add_edge(START, "search")
    g.add_edge("search", "cluster")
    g.add_edge("cluster", "writer")
    g.add_edge("writer", "evaluator")
    g.add_edge("evaluator", END)

    return g.compile()


def run_survey(topic: str, config: dict[str, Any]) -> SurveyState:
    llm_cfg = LLMConfig.from_yaml(config["models"])
    llm = GeminiLLM(llm_cfg)
    graph = build_graph(llm)

    initial: SurveyState = {"topic": topic, "config": config, "logs": []}
    final = graph.invoke(initial, config={"recursion_limit": 25})
    return final
