"""Aggregate per-pair scores into the paper's headline metrics.

Tables produced (paper §4):
  - per (model, depth) breakdown: Link Works pass rate, Relevant Content pass
    rate, Fact Check pass rate, # citations, # attributions, # pairs.
  - per model summary across depths.
  - simple ablation view: depth x fact_check.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..state import AttributionDocument


def _pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 1) if den else 0.0


def _empty_row() -> dict[str, int]:
    return {"n_pairs": 0, "lw_pass": 0, "rc_pass": 0, "fc_pass": 0,
            "n_citations": 0, "n_attributions": 0, "n_queries": 0}


def aggregate(docs: list[AttributionDocument]) -> dict[str, Any]:
    by_cell: dict[tuple[str, str], dict[str, int]] = defaultdict(_empty_row)
    by_model: dict[str, dict[str, int]] = defaultdict(_empty_row)

    for d in docs:
        if not d:
            continue
        model = d.get("model", "?")
        depth = d.get("depth", "?")
        cell = by_cell[(model, depth)]
        mrow = by_model[model]
        for row in (cell, mrow):
            row["n_queries"] += 1
            row["n_citations"] += len(d.get("citations", []))
            row["n_attributions"] += len(d.get("attributions", []))
        for pe in d.get("evals", []):
            for row in (cell, mrow):
                row["n_pairs"] += 1
                row["lw_pass"] += 1 if pe.get("link_works") == 1 else 0
                if pe.get("link_works") == 1:
                    row["rc_pass"] += 1 if pe.get("relevant_content") == 1 else 0
                    row["fc_pass"] += 1 if pe.get("fact_check") == 1 else 0

    def _ratify(row: dict[str, int]) -> dict[str, Any]:
        return {
            **row,
            "link_works_pct":      _pct(row["lw_pass"], row["n_pairs"]),
            "relevant_content_pct": _pct(row["rc_pass"], row["lw_pass"] or 1),
            "fact_check_pct":      _pct(row["fc_pass"], row["lw_pass"] or 1),
        }

    by_cell_out = {
        f"{m}|{d}": _ratify(row) for (m, d), row in sorted(by_cell.items())
    }
    by_model_out = {m: _ratify(row) for m, row in sorted(by_model.items())}

    # Simple ablation: model -> depth -> fact_check_pct
    ablation: dict[str, dict[str, float]] = defaultdict(dict)
    for (m, d), row in by_cell.items():
        ablation[m][d] = _pct(row["fc_pass"], row["lw_pass"] or 1)

    return {
        "by_cell": by_cell_out,
        "by_model": by_model_out,
        "ablation_fact_check_by_depth": dict(ablation),
    }


def render_text_table(agg: dict[str, Any]) -> str:
    """Human-readable summary lines printed at the end of a run."""
    lines: list[str] = []
    lines.append("\n=== Per-(model, depth) cell ===")
    lines.append(f"{'cell':35s} {'pairs':>6} {'LW%':>7} {'RC%':>7} {'FC%':>7}")
    for cell, row in agg["by_cell"].items():
        lines.append(
            f"{cell:35s} {row['n_pairs']:>6} {row['link_works_pct']:>6.1f}%"
            f" {row['relevant_content_pct']:>6.1f}% {row['fact_check_pct']:>6.1f}%"
        )
    lines.append("\n=== Per model (across depths) ===")
    lines.append(f"{'model':30s} {'pairs':>6} {'LW%':>7} {'RC%':>7} {'FC%':>7}")
    for m, row in agg["by_model"].items():
        lines.append(
            f"{m:30s} {row['n_pairs']:>6} {row['link_works_pct']:>6.1f}%"
            f" {row['relevant_content_pct']:>6.1f}% {row['fact_check_pct']:>6.1f}%"
        )
    lines.append("\n=== Ablation: Fact Check % by depth ===")
    for m, by_d in agg["ablation_fact_check_by_depth"].items():
        ordered = " | ".join(f"{d}={v:.1f}%" for d, v in by_d.items())
        lines.append(f"  {m}: {ordered}")
    return "\n".join(lines)
