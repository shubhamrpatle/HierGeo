#!/usr/bin/env python3
"""Metadata-only shortcut baselines on HierGeo-Bench (no pixels; Tab. 6, top block).

    python scripts/run_metadata_baselines.py --out results/metadata_baselines.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hiergeo.baselines import STRATEGIES, MetadataCorpus, run_metadata_baselines  # noqa: E402
from hiergeo.benchmark import STAGE_NAMES, STAGES, load_benchmark  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", default="data/hiergeo_metadata.csv")
    ap.add_argument("--queries", default="data/bench/queries.jsonl")
    ap.add_argument("--qrels", default="data/bench/qrels.jsonl")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", default="results/metadata_baselines.json")
    args = ap.parse_args()

    queries, qrels = load_benchmark(args.queries, args.qrels)
    result = run_metadata_baselines(MetadataCorpus(args.metadata), queries, qrels, seeds=args.seeds)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    keys = ("nDCG@10", "mAP@10", "Recall@10", "Success@10", "P@10")
    for split in ("overall", *STAGES):
        print(f"\n### {'Overall' if split == 'overall' else STAGE_NAMES[split]} (mean ± std over {len(args.seeds)} seeds)\n")
        print("| Baseline | " + " | ".join(keys) + " |")
        print("|---|" + "---:|" * len(keys))
        for strategy in STRATEGIES:
            cells = [f"{result[split][strategy][k]['mean']:.3f} ± {result[split][strategy][k]['std']:.3f}" for k in keys]
            print(f"| {strategy} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
