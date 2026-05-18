"""Evaluator 3 — Fact Check (paper §3.3.3).

  "Fact Check verifies whether specific factual claims are accurately
   supported by the source content. Using an LLM-as-a-judge approach, the
   evaluator examines facts, numbers, dates, and assertions in the
   attribution text against the retrieved source. The evaluator produces a
   binary score of 1 if the facts are supported or consistent and 0 if they
   are contradicted, absent, or uncertain. ... the Fact Check evaluator was
   calibrated through manual review of 50-100 LLM judgments."

The paper does NOT publish the prompt verbatim. The rubric in the prompt
below tracks §3.3.3 closely. The "calibration" step (50-100 manual labels)
is not implemented in v1 — see README deviation audit.
"""

from __future__ import annotations

from ..llm import GeminiClient


SYSTEM = (
    "You are a strict LLM-as-a-judge evaluator of source attribution. "
    "Your task is to decide whether the factual content of a claim is "
    "directly supported by a cited source. Return STRICT JSON only."
)

PROMPT = """A research report contained the following sentence with an inline citation:

CLAIM (citation markers removed):
\"\"\"{claim}\"\"\"

The CITED SOURCE was retrieved from {url}. The first {n_chars} characters of
the extracted main text are below:

SOURCE CONTENT:
\"\"\"{source}\"\"\"

Verify the SPECIFIC factual content of the claim against the source:
- Check named entities (people, organisations, products).
- Check numbers, percentages, dates, durations, units.
- Check directional assertions (X is larger than Y, X causes Y, etc.).
- Check direct quotes against verbatim text in the source.

Scoring rubric (binary, as in the paper):
- score=1 ONLY if every factual element in the claim is SUPPORTED or
  consistent with the source content. Implicit support is acceptable when
  the inference is direct (within one logical step).
- score=0 if any factual element is contradicted by the source, missing
  from the source, or uncertain (cannot be verified from the truncated
  content). Topical relevance is NOT enough.

Return STRICT JSON:
{{
  "score": 0 or 1,
  "explanation": "1-3 sentence justification naming the specific facts checked"
}}
"""


def fact_check(claim: str, source_text: str, url: str, *,
               truncation_chars: int,
               llm: GeminiClient) -> tuple[int, str]:
    truncated = (source_text or "")[:truncation_chars]
    parsed = llm.judge_json(
        PROMPT.format(
            claim=claim or "",
            url=url or "",
            n_chars=truncation_chars,
            source=truncated,
        ),
        system=SYSTEM,
    )
    score = _coerce_binary(parsed.get("score") if isinstance(parsed, dict) else None)
    explanation = str(parsed.get("explanation", "")) if isinstance(parsed, dict) else ""
    return score, explanation


def _coerce_binary(v) -> int:
    try:
        x = int(v)
        return 1 if x >= 1 else 0
    except (TypeError, ValueError):
        return 0
