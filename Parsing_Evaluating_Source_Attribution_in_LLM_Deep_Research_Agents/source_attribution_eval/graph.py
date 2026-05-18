"""LangGraph orchestration matching Algorithm 1 of the paper.

Linear pipeline per (query, model, depth) cell:

    START -> researcher -> parser -> fetcher -> evaluator_runner -> END

`evaluator_runner` parallelises the per-pair Relevant Content and Fact Check
LLM judges with a bounded thread pool (paper §3.4: "15 concurrent
evaluators"). `fetcher` parallelises HTTP retrieval across unique citations.
"""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from .agent import researcher_node
from .evaluators import fact_check, link_works, relevant_content
from .llm import GeminiClient, LLMConfig
from .parser import parse_markdown_report
from .state import AttributionDocument, Citation, PairEval, PipelineState


# ------------------------------------------------------------------ parser node

def parser_node(state: PipelineState) -> dict[str, Any]:
    doc_in: AttributionDocument = state["document"]
    pcfg = state["config"]["parser"]
    doc = parse_markdown_report(
        doc_in["raw_markdown"],
        query_id=state["query_id"],
        query=state["query"],
        model=state["model"],
        depth=state["depth"],
        backward_attribution=bool(pcfg.get("backward_attribution", True)),
        segmenter=str(pcfg.get("segmenter", "regex")),
    )
    log = (
        f"[Phase 1] parsed {len(doc['citations'])} unique citations, "
        f"{len(doc['attributions'])} attributions"
    )
    return {"document": doc, "logs": state.get("logs", []) + [log]}


# ----------------------------------------------------------------- fetcher node

def fetcher_node(state: PipelineState) -> dict[str, Any]:
    """Algorithm 1 lines 8-10: fetch each unique citation in parallel and run
    the Link Works probe at the same time.

    Each fetch is given a hard per-task timeout (timeout + 10s buffer). If a
    worker hangs beyond that, we abandon the future as a failed link, so a
    single stuck URL cannot deadlock the whole run.
    """
    ecfg = state["config"]["evaluators"]
    lw_cfg = ecfg["link_works"]
    timeout = int(lw_cfg["timeout_seconds"])
    ua = str(lw_cfg["user_agent"])
    max_workers = int(ecfg.get("concurrent_evaluators", 15))
    hard_per_task = timeout + 10

    doc: AttributionDocument = state["document"]
    citations: list[Citation] = doc["citations"]
    cit_by_id: dict[str, Citation] = {c["citation_id"]: c for c in citations}

    cid_to_lw: dict[str, tuple[int, str]] = {}

    def _one(c: Citation) -> tuple[str, int, str, str]:
        try:
            score, reason, text = link_works(c, timeout=timeout, user_agent=ua)
        except Exception as e:                                  # pragma: no cover
            return c["citation_id"], 0, f"exception:{type(e).__name__}", ""
        return c["citation_id"], score, reason, text

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_one, c): c["citation_id"] for c in citations}
        for fut in concurrent.futures.as_completed(futures, timeout=None):
            cid = futures[fut]
            try:
                _cid, score, reason, text = fut.result(timeout=hard_per_task)
            except concurrent.futures.TimeoutError:
                score, reason, text = 0, f"task_timeout_{hard_per_task}s", ""
            except Exception as e:                              # pragma: no cover
                score, reason, text = 0, f"exception:{type(e).__name__}", ""
            cid_to_lw[cid] = (score, reason)
            if cid in cit_by_id:
                cit_by_id[cid]["url_content"] = text

    doc["citations"] = citations
    n_ok = sum(1 for s, _ in cid_to_lw.values() if s == 1)
    log = f"[Fetch+LinkWorks] {n_ok}/{len(citations)} URLs accessible"
    # Carry per-citation LW results into the pair evals (one pair per attribution-citation)
    pair_evals: list[PairEval] = []
    for a in doc["attributions"]:
        for cid in a["citation_ids"]:
            s, r = cid_to_lw.get(cid, (0, "unknown"))
            pair_evals.append(PairEval(
                attribution_id=a["attribution_id"],
                citation_id=cid,
                link_works=s,
                link_works_reason=r,
                relevant_content=-1,
                relevant_content_explanation="",
                fact_check=-1,
                fact_check_explanation="",
                error=None,
            ))
    doc["evals"] = pair_evals
    return {"document": doc, "logs": state.get("logs", []) + [log]}


# ----------------------------------------------------------- evaluator runner

def evaluator_runner_node(state: PipelineState, llm: GeminiClient) -> dict[str, Any]:
    """Algorithm 1 lines 11-16: parallel Relevant Content + Fact Check per pair.

    We only run the LLM judges on pairs where Link Works == 1 — there is no
    point in asking the judge about a 404 page.
    """
    ecfg = state["config"]["evaluators"]
    trunc = int(ecfg["source_truncation_chars"])
    max_workers = int(ecfg.get("concurrent_evaluators", 15))

    doc: AttributionDocument = state["document"]
    attributions_by_id = {a["attribution_id"]: a for a in doc["attributions"]}
    citations_by_id = {c["citation_id"]: c for c in doc["citations"]}

    targets: list[tuple[int, PairEval]] = [
        (i, pe) for i, pe in enumerate(doc["evals"]) if pe["link_works"] == 1
    ]

    def _judge(i_pe: tuple[int, PairEval]) -> tuple[int, int, str, int, str, str | None]:
        i, pe = i_pe
        a = attributions_by_id[pe["attribution_id"]]
        c = citations_by_id[pe["citation_id"]]
        claim = a.get("text_nocite") or a.get("text") or ""
        source = c.get("url_content") or ""
        url = c.get("url") or ""
        err: str | None = None
        try:
            rc, rc_ex = relevant_content(
                claim, source, url, truncation_chars=trunc, llm=llm
            )
        except Exception as e:                           # pragma: no cover
            rc, rc_ex = 0, ""
            err = f"relevant_content:{type(e).__name__}"
        try:
            fc, fc_ex = fact_check(
                claim, source, url, truncation_chars=trunc, llm=llm
            )
        except Exception as e:                           # pragma: no cover
            fc, fc_ex = 0, ""
            err = (err + " | " if err else "") + f"fact_check:{type(e).__name__}"
        return i, rc, rc_ex, fc, fc_ex, err

    judge_hard_timeout = 120  # per-pair LLM judge budget (seconds)
    if targets:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_i = {ex.submit(_judge, t): t[0] for t in targets}
            for fut in concurrent.futures.as_completed(future_to_i):
                i = future_to_i[fut]
                try:
                    _, rc, rc_ex, fc, fc_ex, err = fut.result(timeout=judge_hard_timeout)
                except concurrent.futures.TimeoutError:
                    rc, rc_ex, fc, fc_ex = 0, "", 0, ""
                    err = f"judge_timeout_{judge_hard_timeout}s"
                except Exception as e:                          # pragma: no cover
                    rc, rc_ex, fc, fc_ex = 0, "", 0, ""
                    err = f"judge_exception:{type(e).__name__}"
                pe = doc["evals"][i]
                pe["relevant_content"] = rc
                pe["relevant_content_explanation"] = rc_ex
                pe["fact_check"] = fc
                pe["fact_check_explanation"] = fc_ex
                if err:
                    pe["error"] = err

    # Stats
    n = len(doc["evals"])
    n_lw = sum(1 for pe in doc["evals"] if pe["link_works"] == 1)
    n_rc = sum(1 for pe in doc["evals"] if pe["relevant_content"] == 1)
    n_fc = sum(1 for pe in doc["evals"] if pe["fact_check"] == 1)

    def pct(a, b):
        return f"{(100*a/b):.0f}%" if b else "n/a"

    log = (
        f"[Phase 2] {n} pairs | "
        f"LinkWorks {n_lw}/{n} ({pct(n_lw, n)}), "
        f"Relevant {n_rc}/{n} ({pct(n_rc, n)}), "
        f"FactCheck {n_fc}/{n} ({pct(n_fc, n)})"
    )
    return {"document": doc, "logs": state.get("logs", []) + [log]}


# ---------------------------------------------------------------- graph wiring

def build_graph(llm: GeminiClient):
    g = StateGraph(PipelineState)
    g.add_node("researcher", lambda s: researcher_node(s, llm))
    g.add_node("parser", parser_node)
    g.add_node("fetcher", fetcher_node)
    g.add_node("evaluator", lambda s: evaluator_runner_node(s, llm))

    g.add_edge(START, "researcher")
    g.add_edge("researcher", "parser")
    g.add_edge("parser", "fetcher")
    g.add_edge("fetcher", "evaluator")
    g.add_edge("evaluator", END)
    return g.compile()


def run_evaluation(query_id: str, query: str, model: str, depth: str,
                   config: dict[str, Any]) -> PipelineState:
    rcfg = config["research"]
    llm_cfg = LLMConfig(
        research_temperature=float(rcfg.get("temperature", 0.4)),
        research_max_output_tokens=int(rcfg.get("max_output_tokens", 8192)),
        judge_model=str(config["evaluators"].get("judge_model", "gemini-2.5-pro")),
        judge_temperature=float(config["evaluators"].get("judge_temperature", 0.0)),
        request_timeout_seconds=int(rcfg.get("request_timeout_seconds", 180)),
    )
    llm = GeminiClient(llm_cfg)
    graph = build_graph(llm)
    initial: PipelineState = {
        "query_id": query_id, "query": query, "model": model, "depth": depth,
        "config": config, "logs": [], "document": {},
    }
    return graph.invoke(initial, config={"recursion_limit": 25})
