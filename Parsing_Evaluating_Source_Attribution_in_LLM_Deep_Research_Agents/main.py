"""CLI entry point for the Source Attribution Evaluation Framework.

Examples
--------
    # Run all queries × all models × all depths (default)
    python main.py

    # Quick smoke test: 1 query, Pro only, moderate depth
    python main.py --queries q1_quantum --models gemini-2.5-pro --depths moderate

    # Custom queries file
    python main.py --queries-file my_queries.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from source_attribution_eval.graph import run_evaluation
from source_attribution_eval.reports import aggregate, render_text_table


HERE = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Source Attribution Evaluation — reimpl of arXiv:2605.06635",
    )
    p.add_argument("--config", default=str(HERE / "config.yaml"))
    p.add_argument("--queries-file", default=str(HERE / "queries.yaml"))
    p.add_argument("--queries", default=None,
                   help="Comma-separated list of query IDs (default: all)")
    p.add_argument("--models", default=None,
                   help="Comma-separated model overrides (default: config research.models)")
    p.add_argument("--depths", default=None,
                   help="Comma-separated subset of [brief,moderate,extensive] "
                        "(default: all three)")
    p.add_argument("--output-dir", default=None)
    return p.parse_args(argv)


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> int:
    load_dotenv(HERE / ".env")
    args = parse_args(argv)

    cfg = _load_yaml(args.config)
    queries_data = _load_yaml(args.queries_file)
    if args.output_dir:
        cfg["output"]["outputs_dir"] = args.output_dir

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set. Copy .env.example to .env.",
              file=sys.stderr)
        return 2

    # Determine the (query, model, depth) matrix
    all_queries = {q["id"]: q for q in queries_data["queries"]}
    if args.queries:
        qids = [q.strip() for q in args.queries.split(",") if q.strip()]
        missing = [q for q in qids if q not in all_queries]
        if missing:
            print(f"ERROR: unknown query ids: {missing}", file=sys.stderr); return 2
    else:
        qids = list(all_queries.keys())

    models = ([m.strip() for m in args.models.split(",") if m.strip()]
              if args.models else list(cfg["research"]["models"]))
    depths = ([d.strip() for d in args.depths.split(",") if d.strip()]
              if args.depths else list(cfg["research"]["depth_levels"].keys()))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(cfg["output"]["outputs_dir"])
    if not out_dir.is_absolute():
        out_dir = HERE / out_dir
    run_dir = out_dir / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=== Source Attribution Evaluation (arXiv:2605.06635 reimpl) ===")
    print(f"Queries : {len(qids)}  -> {qids}")
    print(f"Models  : {models}")
    print(f"Depths  : {depths}")
    print(f"Output  : {run_dir}")
    print()

    docs = []
    cells = [(q, m, d) for q in qids for m in models for d in depths]

    for qid, model, depth in tqdm(cells, desc="cells"):
        q = all_queries[qid]
        try:
            state = run_evaluation(q["id"], q["text"], model, depth, cfg)
        except Exception as e:
            print(f"  [error] {qid} | {model} | {depth}: {type(e).__name__}: {e}")
            continue
        doc = state.get("document", {})
        docs.append(doc)
        # Per-cell JSON
        cell_path = run_dir / f"{qid}__{model.replace('/', '_')}__{depth}.json"
        cell_path.write_text(
            json.dumps({"logs": state.get("logs", []), "document": doc},
                       indent=2 if cfg["output"]["pretty_print_json"] else None,
                       ensure_ascii=False),
            encoding="utf-8",
        )
        for line in state.get("logs", []):
            print("  " + line)

    # Aggregate
    agg = aggregate(docs)
    (run_dir / "_summary.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(render_text_table(agg))
    print(f"\nResults saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
