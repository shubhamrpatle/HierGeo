#!/usr/bin/env python3
"""Evaluate a dual encoder on HierGeo-Bench.

Example (fine-tuned SigLIP2, the three routing policies of Tab. 7):

    python scripts/evaluate_bench.py \
        --encoder hf_siglip --model checkpoints/hiergeo-siglip2-l16-512 \
        --embeddings indices/siglip2_ft/image_embeddings.npy \
        --modes global country country_city --out results/siglip2_ft

Writes ``<out>/<mode>/per_query.jsonl`` and ``<out>/<mode>/summary.json`` and
prints a Markdown table per mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hiergeo.benchmark import format_summary, load_benchmark, per_country, run_visual, score_rows, summarize, write_jsonl  # noqa: E402
from hiergeo.models import ENCODER_KINDS, load_encoder  # noqa: E402
from hiergeo.retrieval import ROUTING_MODES, ShardedImageIndex  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", choices=ENCODER_KINDS, required=True)
    ap.add_argument("--model", required=True, help="HF id / local dir, or OpenCLIP architecture")
    ap.add_argument("--checkpoint", default=None, help="OpenCLIP .pt state dict (RemoteCLIP / SkyCLIP)")
    ap.add_argument("--pretrained", default=None, help="OpenCLIP pretrained tag if no --checkpoint")
    ap.add_argument("--embeddings", required=True, help="image_embeddings.npy built with scripts/build_index.py")
    ap.add_argument("--metadata", default="data/hiergeo_metadata.csv")
    ap.add_argument("--queries", default="data/bench/queries.jsonl")
    ap.add_argument("--qrels", default="data/bench/qrels.jsonl")
    ap.add_argument("--modes", nargs="+", choices=ROUTING_MODES, default=["country_city"])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument(
        "--text-padding",
        choices=["longest", "max_length"],
        default="longest",
        help="'longest' reproduces the paper's tables; 'max_length' matches SigLIP training",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    queries, qrels = load_benchmark(args.queries, args.qrels)
    encoder = load_encoder(args.encoder, args.model, device=args.device, checkpoint=args.checkpoint, pretrained=args.pretrained)
    index = ShardedImageIndex.load(args.embeddings, args.metadata, device=encoder.device)
    text_autocast = torch.float16 if encoder.device.startswith("cuda") else None

    for mode in args.modes:
        rows = run_visual(
            index, encoder, queries, mode=mode, top_k=args.top_k,
            text_padding=args.text_padding, text_autocast=text_autocast,
        )
        score_rows(rows, qrels, k=args.top_k)
        summary = summarize(rows)
        summary["per_country"] = per_country(rows)
        out = Path(args.out) / mode
        write_jsonl(out / "per_query.jsonl", rows)
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(format_summary(summary, title=f"{args.model} | routing={mode}"))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
