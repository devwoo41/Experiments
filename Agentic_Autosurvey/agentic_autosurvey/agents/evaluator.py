"""Agent 4 — Quality Evaluator (paper §2.2.4, Table 4, K.4).

The prompt below is reproduced as closely as possible to Appendix K.4 of the
paper, with only the bracketed placeholders ([N], [TOPIC], [K]) substituted.
The 12 dimensions and their 60/20/20 weighted aggregation are taken from
the paper's Table 4. Output schema also matches K.4 verbatim.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from ..llm import GeminiLLM
from ..state import Evaluation, SurveyState


# Deviation #2 from paper: the K.4 prompt does not specify the current date.
# That is fine when the judge model's training cutoff is recent (paper used
# Claude Opus 4.1). With an older-cutoff judge (e.g. Gemini Flash) the model
# can hallucinate that recent papers are "future-dated" and tank the Accuracy
# and Citation Coverage scores. We inject the current date in the system
# prompt only; the user-facing K.4 prompt body remains verbatim.
EVAL_SYSTEM = (
    "You are the Quality Evaluator subagent in the Agentic AutoSurvey "
    "pipeline (arXiv:2509.18661). You assess generated surveys using a "
    "12-dimensional rubric, applying contextual academic-reviewer judgment. "
    "Return STRICT JSON only — no commentary, no fences.\n\n"
    f"IMPORTANT — Today's date is {date.today().isoformat()}. "
    "Any paper dated on or before today is a real published or preprint work, "
    "NOT a future-dated or fabricated reference. Do not penalize a paper for "
    "having a publication year that postdates your own training cutoff."
)


# Verbatim prompt body from paper Appendix K.4. Placeholders [N], [TOPIC], [K]
# are substituted at call time. Output schema is preserved.
EVAL_PROMPT = """Evaluate the generated survey using a comprehensive 12-dimensional framework.

The survey was generated from {N} papers on {TOPIC}, organized into {K} clusters.

EVALUATION FRAMEWORK:

CORE QUALITY (60% weight):
1. Citation Coverage - % of papers cited effectively
   - Calculate exact percentage of corpus cited
   - Assess distribution across clusters
   - Check for key papers inclusion

2. Accuracy - Factual correctness and attribution
   - Verify claims are properly supported
   - Check author/year attribution accuracy
   - Identify any unsupported generalizations

3. Synthesis Quality - Integration vs mere listing
   - Measure synthesis ratio (integrated vs sequential)
   - Identify cross-paper connections
   - Evaluate comparative analysis depth

4. Organization - Logical flow and structure
   - Assess section/subsection hierarchy
   - Evaluate transition quality
   - Check information progression logic

WRITING QUALITY (20% weight):
5. Readability - Clarity for target audience
   - Sentence complexity and variety
   - Technical term introduction/explanation
   - Paragraph coherence

6. Academic Rigor - Adherence to scholarly standards
   - Citation format consistency
   - Methodological transparency
   - Limitation acknowledgment

7. Clarity - Precision in technical descriptions
   - Concept explanation quality
   - Ambiguity identification
   - Example usage effectiveness

8. Coherence - Internal consistency
   - Thematic consistency
   - Cross-reference accuracy
   - Narrative flow maintenance

CONTENT DEPTH (20% weight):
9. Comprehensiveness - Breadth of topic coverage
   - Cluster representation completeness
   - Temporal coverage (publication years)
   - Geographic/institutional diversity

10. Critical Analysis - Depth of evaluation
    - Limitation discussion depth
    - Conflicting findings acknowledgment
    - Methodological critique presence

11. Novelty & Insights - Original contributions
    - Novel connections identified
    - Pattern recognition quality
    - Taxonomy/framework contributions

12. Future Directions - Research trajectory identification
    - Specificity of proposed directions
    - Feasibility assessment
    - Gap identification quality

OUTPUT REQUIREMENTS:

Provide detailed JSON with:
- dimensional_scores: {{
    dimension_name: {{
      score: 0-10,
      weight: percentage,
      justification: detailed explanation,
      metrics: quantitative measures,
      specific_examples: 3+ concrete examples
    }}
  }}

- overall_assessment: {{
    weighted_total_score: calculated score,
    score_breakdown: by category,
    quality_level: grade and description,
    publication_readiness: specific assessment
  }}

- comparison_to_standards: {{
    vs_acm_computing_surveys: assessment,
    vs_conference_surveys: assessment,
    vs_workshop_papers: assessment
  }}

- strengths: [7+ specific strengths with evidence]

- weaknesses: [7+ specific weaknesses with evidence]

- prioritized_recommendations: [
    {{priority: HIGH/MEDIUM/LOW,
     recommendation: specific action,
     impact: expected improvement,
     effort: implementation difficulty}}
  ]

- executive_summary: 200-word synthesis

Use nuanced, context-aware evaluation rather than rigid rules.
Consider the specific domain, corpus size, and clustering quality.
Be critical but fair, acknowledging both achievements and gaps.

THE SURVEY UNDER REVIEW:

ABSTRACT:
{abstract}

{sections}
"""


# Map between display names used in paper prompt and snake_case dimension keys
# from config.yaml. The Evaluator JSON returns keys as they appeared in the
# rubric; we normalise both.
_NAME_ALIASES = {
    "citation coverage": "citation_coverage",
    "accuracy": "accuracy",
    "synthesis quality": "synthesis_quality",
    "organization": "organization",
    "readability": "readability",
    "academic rigor": "academic_rigor",
    "clarity": "clarity",
    "coherence": "coherence",
    "comprehensiveness": "comprehensiveness",
    "critical analysis": "critical_analysis",
    "novelty & insights": "novelty_insights",
    "novelty and insights": "novelty_insights",
    "novelty insights": "novelty_insights",
    "future directions": "future_directions",
}


def _normalise_dim_key(name: str) -> str:
    n = name.strip().lower().replace("_", " ")
    return _NAME_ALIASES.get(n, n.replace(" ", "_"))


def _format_sections(sections: list[dict]) -> str:
    out = []
    for s in sections:
        out.append(f"{s['title'].upper()}:\n{s['content']}\n")
    return "\n".join(out)


def evaluator_node(state: SurveyState, llm: GeminiLLM) -> dict[str, Any]:
    cfg = state["config"]["evaluator"]
    expected_dims = {d["name"]: d for d in cfg["dimensions"]}
    cat_weights: dict[str, float] = cfg["category_weights"]
    max_score = float(cfg["max_score"])

    n_papers = len(state.get("papers", []))
    k_clusters = len(state.get("clusters", []))

    prompt = EVAL_PROMPT.format(
        N=n_papers,
        TOPIC=state["topic"],
        K=k_clusters,
        abstract=state.get("abstract_text", ""),
        sections=_format_sections(state.get("sections", [])),
    )
    parsed = llm.generate_json(prompt, tier="light", system=EVAL_SYSTEM)
    if not isinstance(parsed, dict):
        parsed = {}

    # ---- dimensional_scores ----
    raw_dim = parsed.get("dimensional_scores", {}) or {}
    dim_out: dict[str, dict[str, Any]] = {}
    for raw_name, raw_val in raw_dim.items():
        key = _normalise_dim_key(raw_name)
        if key not in expected_dims:
            continue
        spec = expected_dims[key]
        if isinstance(raw_val, dict):
            try:
                score = max(0.0, min(max_score, float(raw_val.get("score", 0))))
            except (TypeError, ValueError):
                score = 0.0
            dim_out[key] = {
                "score": score,
                "weight": spec["weight"],
                "category": spec["category"],
                "justification": str(raw_val.get("justification", "")),
                "metrics": str(raw_val.get("metrics", "")),
                "specific_examples": [str(x) for x in (raw_val.get("specific_examples") or [])],
            }
    # Fill missing dims with zero so downstream aggregation is stable
    for key, spec in expected_dims.items():
        if key not in dim_out:
            dim_out[key] = {
                "score": 0.0, "weight": spec["weight"], "category": spec["category"],
                "justification": "(missing from judge output)",
                "metrics": "", "specific_examples": [],
            }

    # ---- weighted aggregation per paper Table 4: 60/20/20 ----
    by_cat: dict[str, list[tuple[float, float]]] = {}  # category -> [(score, intra_weight)]
    for key, item in dim_out.items():
        by_cat.setdefault(item["category"], []).append(
            (item["score"], item["weight"])
        )

    # Intra-category aggregation: simple weighted mean (all four dims in a
    # category share equal in-category weight, so this matches the paper)
    cat_scores: dict[str, float] = {}
    for cat, items in by_cat.items():
        total_w = sum(w for _, w in items) or 1.0
        cat_scores[cat] = sum(s * w for s, w in items) / total_w

    weighted_total = sum(cat_scores.get(cat, 0.0) * w for cat, w in cat_weights.items())

    quality_level = _quality_level(weighted_total)
    overall_assessment_from_llm = parsed.get("overall_assessment", {}) or {}

    overall_assessment = {
        "weighted_total_score": round(weighted_total, 3),
        "score_breakdown": {
            f"{cat} ({cat_weights.get(cat, 0)*100:.0f}%)": round(cat_scores.get(cat, 0.0), 3)
            for cat in cat_weights
        },
        "quality_level": overall_assessment_from_llm.get("quality_level") or quality_level,
        "publication_readiness": overall_assessment_from_llm.get("publication_readiness", ""),
    }

    evaluation: Evaluation = {
        "dimensional_scores": dim_out,
        "overall_assessment": overall_assessment,
        "comparison_to_standards": parsed.get("comparison_to_standards", {}) or {},
        "strengths": [str(x) for x in (parsed.get("strengths") or [])],
        "weaknesses": [str(x) for x in (parsed.get("weaknesses") or [])],
        "prioritized_recommendations": [
            r for r in (parsed.get("prioritized_recommendations") or []) if isinstance(r, dict)
        ],
        "executive_summary": str(parsed.get("executive_summary", "")),
    }

    log = (
        f"[Evaluator] weighted_total={weighted_total:.2f}/{max_score:.0f} "
        f"(core={cat_scores.get('core', 0):.2f}, "
        f"writing={cat_scores.get('writing', 0):.2f}, "
        f"depth={cat_scores.get('depth', 0):.2f})"
    )
    return {"evaluation": evaluation, "logs": state.get("logs", []) + [log]}


def _quality_level(score: float) -> str:
    if score >= 9.0:
        return "A — publication-ready (top venue)"
    if score >= 8.0:
        return "A- — strong submission, minor revisions"
    if score >= 7.0:
        return "B+ — solid, moderate revisions needed"
    if score >= 6.0:
        return "B — competent but uneven"
    if score >= 5.0:
        return "C — major revisions required"
    return "D — significant rework needed"
