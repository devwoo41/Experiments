"""Agent 3 — Academic Survey Writer (paper §2.2.3, K.3).

Faithful to the paper:
  - Structure: Abstract -> Introduction -> one section per cluster ->
                Cross-cutting analysis -> Future directions -> Conclusion.
  - Target 8,000-12,000 words.
  - Cite at least 80% of papers.
  - [Author, Year] citation format (verbatim in prose).
  - Synthesis-first, comparative analysis, pattern/trend identification,
    methodology comparison, research-gap highlighting.
"""

from __future__ import annotations

import re
from typing import Any

from ..llm import GeminiLLM
from ..state import Cluster, Paper, SectionDraft, SurveyState


# Writer system prompt mirrors the K.3 invocation exactly in spirit.
WRITER_SYSTEM = (
    "You are the Academic Survey Writer subagent in the Agentic AutoSurvey "
    "pipeline (arXiv:2509.18661). You generate publication-quality academic "
    "survey content. You always cite using the inline [Author, Year] format "
    "(e.g. '[Lewis et al., 2020]') drawn from the paper list you are given. "
    "You synthesize across papers (no paper-by-paper enumeration); you "
    "compare methodologies, identify patterns and trends, and highlight "
    "research gaps. Use academic prose; do not emit markdown bullet outlines."
)


# Verbatim K.3 directives baked into every Writer call:
WRITER_DIRECTIVES = """Requirements:
1. Structure (this single section only — produced as part of the larger survey):
   {structural_note}

2. Writing guidelines:
   - Target length for THIS section: {section_target_words} words.
   - Cite at least 80% of relevant papers from the list below.
   - Use [Author, Year] citation format. Authors are given verbatim — use them.
   - Focus on synthesis over enumeration.
   - Identify patterns and trends.
   - Compare methodologies.
   - Highlight research gaps.

3. Quality criteria:
   - Academic rigor and clarity.
   - Comprehensive coverage across clusters.
   - Critical analysis not just summary.
   - Smooth transitions between sections.
   - Technical accuracy.

Generate publication-quality prose that provides genuine insights and value
to researchers in the field."""


SECTION_PROMPT = """Topic: "{topic}"

You are writing the **{section_label}** section.
{specific_directive}

Papers available for citation (use the EXACT [Author, Year] keys listed):
{papers}

{directives}

Return ONLY the section body in plain text (LaTeX-compatible). Do not include
the \\section heading; the orchestrator inserts it. Use \\subsection*{{...}}
for sub-headings only if genuinely needed."""


ABSTRACT_PROMPT = """Topic: "{topic}"

Write the Abstract of a comprehensive academic survey on this topic.

Constraints (paper-specified):
- Length: 200-300 words.
- Mention scope, the {n_clusters} sub-topic clusters covered ({cluster_titles}),
  and the survey's distinguishing perspective.
- Do NOT include a heading. Plain prose only."""


def _format_authors_short(authors: list[str]) -> str:
    if not authors:
        return "Anonymous"
    first = authors[0]
    last_name = first.split()[-1] if first else "Anonymous"
    if len(authors) == 1:
        return last_name
    if len(authors) == 2:
        return f"{last_name} and {authors[1].split()[-1]}"
    return f"{last_name} et al."


def _citation_key(p: Paper) -> str:
    authors_short = _format_authors_short(p.get("authors", []))
    year = p.get("year") or "n.d."
    return f"[{authors_short}, {year}]"


def _assign_citation_keys(papers: list[Paper]) -> None:
    """Stamp each paper with its [Author, Year] citation key, disambiguating ties with a/b/c."""
    used: dict[str, int] = {}
    for p in papers:
        key = _citation_key(p)
        used[key] = used.get(key, 0) + 1
    seen: dict[str, int] = {}
    for p in papers:
        key = _citation_key(p)
        if used[key] > 1:
            n = seen.get(key, 0)
            suffix = chr(ord("a") + n)
            seen[key] = n + 1
            base = key.rstrip("]")
            key = f"{base}{suffix}]"
        p["citation_key"] = key


def _format_papers_for_prompt(papers: list[Paper], max_chars: int = 18000) -> str:
    out: list[str] = []
    total = 0
    for p in papers:
        key = p.get("citation_key") or _citation_key(p)
        title = p.get("title", "")
        abstract = (p.get("abstract") or "")[:700]
        card = f"- key={key} | {title}\n    abstract: {abstract}"
        if total + len(card) > max_chars:
            break
        out.append(card)
        total += len(card)
    return "\n".join(out)


def _papers_by_id(state: SurveyState) -> dict[str, Paper]:
    return {p["paper_id"]: p for p in state["papers"]}


# Citation-key extraction (e.g. [Smith et al., 2023], [Lewis et al., 2020a])
_CITE_RE = re.compile(r"\[([^\[\]]+,\s*(?:\d{4}[a-z]?|n\.d\.))\]")


def _extract_cited_keys(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in _CITE_RE.finditer(text or ""):
        key = f"[{m.group(1)}]"
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def writer_node(state: SurveyState, llm: GeminiLLM) -> dict[str, Any]:
    cfg = state["config"]["writer"]
    topic = state["topic"]
    clusters: list[Cluster] = state["clusters"]
    papers: list[Paper] = state["papers"]
    papers_by_id = _papers_by_id(state)

    _assign_citation_keys(papers)
    key_to_paper: dict[str, Paper] = {p["citation_key"]: p for p in papers if p.get("citation_key")}

    # Word budgets per section to land on 8,000-12,000 words total
    n_clusters = len(clusters)
    has_cross = cfg.get("include_cross_cutting", True)
    has_future = cfg.get("include_future_directions", True)
    total_target = (cfg["target_word_count_min"] + cfg["target_word_count_max"]) // 2  # ~10k
    cluster_share = int(total_target * 0.65 / max(1, n_clusters))      # ~65% across clusters
    intro_words = int(total_target * 0.10)
    cross_words = int(total_target * 0.10) if has_cross else 0
    future_words = int(total_target * 0.07) if has_future else 0
    conclusion_words = int(total_target * 0.08)

    def directives(section_target_words: int, structural_note: str) -> str:
        return WRITER_DIRECTIVES.format(
            section_target_words=section_target_words,
            structural_note=structural_note,
        )

    sections: list[SectionDraft] = []

    # --- Introduction ---
    intro_text = llm.generate(
        SECTION_PROMPT.format(
            topic=topic,
            section_label="Introduction",
            specific_directive=(
                "Cover (a) motivation, (b) scope and contributions of this survey, "
                "(c) a one-paragraph roadmap that names each of the following clusters in order: "
                + "; ".join(f"\"{c['name']}\"" for c in clusters)
            ),
            papers=_format_papers_for_prompt(papers),
            directives=directives(intro_words,
                                  "You are producing the Introduction (motivation + contributions)."),
        ),
        tier="writer",
        system=WRITER_SYSTEM,
    ).strip()
    sections.append(SectionDraft(cluster_id=None, kind="intro",
                                 title="Introduction", content=intro_text))

    # --- One section per cluster ---
    for c in clusters:
        member_papers = [papers_by_id[pid] for pid in c["paper_ids"] if pid in papers_by_id]
        body = llm.generate(
            SECTION_PROMPT.format(
                topic=topic,
                section_label=c["name"],
                specific_directive=(
                    f"This section covers the cluster '{c['name']}' "
                    f"(key terms: {', '.join(c.get('key_terms', []))}). "
                    "Synthesize across the papers below; do not enumerate them."
                ),
                papers=_format_papers_for_prompt(member_papers),
                directives=directives(cluster_share,
                                      "One section of the survey, focused on a single discovered cluster."),
            ),
            tier="writer",
            system=WRITER_SYSTEM,
        ).strip()
        sections.append(SectionDraft(cluster_id=c["cluster_id"], kind="cluster",
                                     title=c["name"], content=body))

    # --- Cross-cutting analysis (paper-required) ---
    if has_cross:
        cluster_outline = "\n".join(
            f"  - {c['name']} (size {c['size']}): {', '.join(c.get('key_terms', [])[:5])}"
            for c in clusters
        )
        cross_body = llm.generate(
            SECTION_PROMPT.format(
                topic=topic,
                section_label="Cross-Cutting Analysis",
                specific_directive=(
                    "Identify connections, contradictions, and shared assumptions across "
                    "the clusters listed below. Compare methodologies across clusters, "
                    "highlight overarching trends, and surface tensions between approaches.\n"
                    f"Clusters:\n{cluster_outline}"
                ),
                papers=_format_papers_for_prompt(papers),
                directives=directives(cross_words,
                                      "A dedicated cross-cutting analysis section synthesizing across clusters."),
            ),
            tier="writer",
            system=WRITER_SYSTEM,
        ).strip()
        sections.append(SectionDraft(cluster_id=None, kind="cross_cutting",
                                     title="Cross-Cutting Analysis", content=cross_body))

    # --- Future directions (paper-required) ---
    if has_future:
        future_body = llm.generate(
            SECTION_PROMPT.format(
                topic=topic,
                section_label="Future Directions",
                specific_directive=(
                    "Propose 5-8 concrete future research directions for the field, "
                    "each grounded in gaps surfaced by the surveyed papers. "
                    "Be specific about feasibility and what kind of work would be needed."
                ),
                papers=_format_papers_for_prompt(papers),
                directives=directives(future_words,
                                      "A standalone Future Directions section."),
            ),
            tier="writer",
            system=WRITER_SYSTEM,
        ).strip()
        sections.append(SectionDraft(cluster_id=None, kind="future",
                                     title="Future Directions", content=future_body))

    # --- Conclusion ---
    conclusion_body = llm.generate(
        SECTION_PROMPT.format(
            topic=topic,
            section_label="Conclusion",
            specific_directive=(
                "Synthesize the key findings of the survey, restate the main "
                "contributions, and close with a forward-looking statement."
            ),
            papers=_format_papers_for_prompt(papers),
            directives=directives(conclusion_words,
                                  "The closing Conclusion section of the survey."),
        ),
        tier="writer",
        system=WRITER_SYSTEM,
    ).strip()
    sections.append(SectionDraft(cluster_id=None, kind="conclusion",
                                 title="Conclusion", content=conclusion_body))

    # --- Abstract (200-300 words, paper-required) ---
    abstract = llm.generate(
        ABSTRACT_PROMPT.format(
            topic=topic,
            n_clusters=n_clusters,
            cluster_titles="; ".join(c["name"] for c in clusters),
        ),
        tier="writer",
        system=WRITER_SYSTEM,
    ).strip()

    # --- Collect cited references in appearance order ---
    cited_keys: list[str] = []
    seen: set[str] = set()
    for sec in sections:
        for k in _extract_cited_keys(sec["content"]):
            if k not in seen:
                seen.add(k)
                cited_keys.append(k)
    ordered_refs: list[Paper] = [key_to_paper[k] for k in cited_keys if k in key_to_paper]

    # 80% citation-rate diagnostic (paper §2.2.3)
    cite_rate = (len(ordered_refs) / max(1, len(papers)))
    total_words = sum(len((s["content"] or "").split()) for s in sections) + len(abstract.split())
    log = (
        f"[Writer] {len(sections)} sections, ~{total_words} words, "
        f"{len(ordered_refs)}/{len(papers)} cited ({cite_rate:.0%})"
    )

    return {
        "abstract_text": abstract,
        "sections": sections,
        "references": ordered_refs,
        "logs": state.get("logs", []) + [log],
    }
