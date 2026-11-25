#!/usr/bin/env python3
"""Paired bootstrap comparison of two HierGeo-Bench runs (Tab. 11).

    python scripts/compare_runs.py \
        --candidate results/clip_ft/country_city/per_query.jsonl \
        --baseline  results/clip_pt/country_city/per_query.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hiergeo.benchmark import STAGE_NAMES, STAGES, read_jsonl  # noqa: E402
from hiergeo.stats import paired_bootstrap  # noqa: E402


def per_query(path: str, metric: str, stage: str | None):
    return {
        r["query_id"]: r["metrics"][metric]
        for r in read_jsonl(path)
        if r.get("has_positive") and (stage is None or r["stage"] == stage)
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--metric", default="nDCG@10")
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = {}
    print(f"| Split | Queries | {args.metric} candidate | baseline | delta | 95% CI |")
    print("|---|---:|---:|---:|---:|---|")
    for split in (None, *STAGES):
        r = paired_bootstrap(per_query(args.candidate, args.metric, split), per_query(args.baseline, args.metric, split),
                             samples=args.samples, seed=args.seed)
        name = "overall" if split is None else split
        result[name] = r
        label = "Overall" if split is None else STAGE_NAMES[split]
        print(f"| {label} | {r['queries']:,} | {r['candidate_mean']:.4f} | {r['baseline_mean']:.4f} | "
              f"{r['delta']:+.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] |")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
