"""CLI entry point for Agentic AutoSurvey.

Examples
--------
    # Quick test (small, ~20 papers)
    python main.py --topic "retrieval augmented generation" --target-papers 20

    # Paper-faithful run (100-150 papers, K \in [5,15], full 12-dim eval)
    python main.py --topic "LLM alignment with human feedback"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from agentic_autosurvey.exporters import export_latex
from agentic_autosurvey.graph import run_survey


HERE = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Agentic AutoSurvey — faithful reimplementation of arXiv:2509.18661",
    )
    p.add_argument("--topic", required=True, help="Survey topic")
    p.add_argument("--config", default=str(HERE / "config.yaml"))
    p.add_argument("--target-papers", type=int, default=None,
                   help="Override search.target_papers (paper default 100-150)")
    p.add_argument("--max-papers", type=int, default=None,
                   help="Override search.max_papers")
    p.add_argument("--num-queries", type=int, default=None,
                   help="Override search.num_query_expansions (paper: 20-30)")
    p.add_argument("--k-min", type=int, default=None,
                   help="Override clustering.k_min (paper: 5)")
    p.add_argument("--k-max", type=int, default=None,
                   help="Override clustering.k_max (paper: 15)")
    p.add_argument("--year-min", type=int, default=None)
    p.add_argument("--year-max", type=int, default=None)
    p.add_argument("--sources", default=None,
                   help="Comma-separated subset of [arxiv,semantic_scholar]")
    p.add_argument("--output-dir", default=None)
    return p.parse_args(argv)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.target_papers is not None:
        cfg["search"]["target_papers"] = args.target_papers
        cfg["search"]["max_papers"] = max(cfg["search"]["max_papers"], args.target_papers)
    if args.max_papers is not None:
        cfg["search"]["max_papers"] = args.max_papers
    if args.num_queries is not None:
        cfg["search"]["num_query_expansions"] = args.num_queries
    if args.k_min is not None:
        cfg["clustering"]["k_min"] = args.k_min
    if args.k_max is not None:
        cfg["clustering"]["k_max"] = args.k_max
    if args.year_min is not None:
        cfg["search"]["year_min"] = args.year_min
    if args.year_max is not None:
        cfg["search"]["year_max"] = args.year_max
    if args.sources is not None:
        cfg["search"]["sources"] = [s.strip() for s in args.sources.split(",") if s.strip()]
    if args.output_dir is not None:
        cfg["output"]["outputs_dir"] = args.output_dir
    return cfg


def main(argv: list[str] | None = None) -> int:
    load_dotenv(HERE / ".env")
    args = parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args)

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set. Copy .env.example to .env and fill it in.",
              file=sys.stderr)
        return 2

    print("=== Agentic AutoSurvey (arXiv:2509.18661 reimplementation) ===")
    print(f"Topic           : {args.topic}")
    print(f"Sources         : {cfg['search']['sources']}")
    print(f"Query expansions: {cfg['search']['num_query_expansions']}")
    print(f"Target papers   : {cfg['search']['target_papers']} (max {cfg['search']['max_papers']})")
    print(f"K range         : [{cfg['clustering']['k_min']}, {cfg['clustering']['k_max']}] (silhouette-selected)")
    print(f"Writer model    : {cfg['models']['writer']}")
    print(f"Light model     : {cfg['models']['light']}")
    print(f"Embedding model : {cfg['clustering']['embedding_model']}")
    print()

    state = run_survey(args.topic, cfg)

    for line in state.get("logs", []):
        print(line)

    out_dir = Path(cfg["output"]["outputs_dir"])
    if not out_dir.is_absolute():
        out_dir = HERE / out_dir
    tex_path = export_latex(state, out_dir)

    ev = state.get("evaluation") or {}
    overall = (ev.get("overall_assessment") or {}).get("weighted_total_score")
    print()
    print(f"LaTeX written to: {tex_path}")
    print(f"Eval report:      {tex_path.with_suffix('').as_posix()}.eval.json")
    if overall is not None:
        print(f"Weighted overall: {overall}/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
