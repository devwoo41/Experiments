"""Evaluator 2 — Relevant Content (paper §3.3.2).

  "Relevant Content measures topical alignment between the claim and the
   cited source using an LLM-as-a-judge approach. Given the attribution text
   and retrieved source content (truncated to 5,000 characters), the
   evaluator determines whether the source addresses the same topic as the
   claim. The evaluator produces a binary score with a natural language
   explanation."

The paper does NOT publish the prompt verbatim. The wording below is a
rubric-based prompt derived from §3.3.2 and labelled as such in the README's
deviation audit.
"""

from __future__ import annotations

from ..llm import GeminiClient


SYSTEM = (
    "You are a rigorous LLM-as-a-judge evaluator of source attribution. "
    "Your task is to decide whether a cited source is TOPICALLY RELEVANT to "
    "a claim made in an LLM-generated research report. "
    "You return STRICT JSON only."
)

PROMPT = """A research report contained the following sentence with a citation:

CLAIM (citation markers removed):
\"\"\"{claim}\"\"\"

The CITED SOURCE was retrieved from {url}. The first {n_chars} characters of
the extracted main text are below:

SOURCE CONTENT:
\"\"\"{source}\"\"\"

Decide topical alignment ONLY. Do NOT check factual accuracy here.
- score=1 if the source clearly addresses the same topic or sub-topic as the claim.
- score=0 if the source is off-topic, parked, generic landing page, error page,
  or covers a different subject entirely.

Return STRICT JSON:
{{
  "score": 0 or 1,
  "explanation": "1-3 sentence natural-language justification"
}}
"""


def relevant_content(claim: str, source_text: str, url: str, *,
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
