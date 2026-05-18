"""Phase 0 — Deep Research Agent.

Calls Gemini with the built-in `google_search` grounding tool to produce a
markdown research report with inline `[text](url)` citations. The system
prompt enforces the paper's two constraints (§3.4):
  (a) citation format requirements
  (b) a minimum search depth — here approximated by a depth instruction
      because Gemini grounding does not expose a tool-call cap.
"""

from __future__ import annotations

from typing import Any

from ..llm import GeminiClient
from ..state import AttributionDocument, PipelineState


SYSTEM_TEMPLATE = """You are a deep research agent. The user gives you a research
question. You must:

1. Use the `google_search` tool to find relevant, authoritative web sources.
2. {depth_instruction}
3. Write a comprehensive Markdown report (3-8 paragraphs) that answers the question.
4. Cite every factual claim, statistic, name, date, or quote with an inline
   Markdown link of the form `[short label](https://...)`. The URL MUST be a
   real URL you retrieved through search, not a placeholder.
5. Citations must appear inline next to the specific claim they support, not
   only at the end of the report.
6. If you cannot find sufficient sources for a claim, do not invent one —
   leave the claim uncited or omit it.

Output ONLY the Markdown report. Do not include preambles like "Here is the
report"."""


USER_TEMPLATE = "Research question:\n{query}\n"


def researcher_node(state: PipelineState, llm: GeminiClient) -> dict[str, Any]:
    cfg = state["config"]["research"]
    depth = state["depth"]
    depth_instructions: dict[str, str] = cfg["depth_levels"]
    if depth not in depth_instructions:
        raise KeyError(f"Unknown depth level {depth!r}; expected one of {list(depth_instructions)}")

    system = SYSTEM_TEMPLATE.format(depth_instruction=depth_instructions[depth])
    prompt = USER_TEMPLATE.format(query=state["query"])

    text, grounded_urls = llm.generate_grounded_markdown(
        model_name=state["model"], prompt=prompt, system=system
    )

    document: AttributionDocument = {
        "query_id": state["query_id"],
        "query": state["query"],
        "model": state["model"],
        "depth": depth,
        "raw_markdown": text,
        "citations": [],
        "attributions": [],
        "evals": [],
    }
    log = (
        f"[Phase 0] {state['model']} ({depth}): "
        f"{len(text)} chars markdown, {len(grounded_urls)} grounded sources"
    )
    return {"document": document, "logs": state.get("logs", []) + [log]}
