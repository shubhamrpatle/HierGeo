#!/usr/bin/env python3
"""Place-constrained search from the command line.

    python scripts/search.py "Residential in Luanda, Angola" \
        --encoder hf_clip --model checkpoints/hiergeo-clip-l14 \
        --embeddings indices/clip_ft/image_embeddings.npy --top-k 5

Prints the parsed route (city / country / global), the number of candidates in
the shard, and the top-k images with their captions.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hiergeo.models import ENCODER_KINDS, load_encoder  # noqa: E402
from hiergeo.retrieval import ROUTING_MODES, HierGeoRetriever, ShardedImageIndex  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="+", help="one or more queries")
    ap.add_argument("--encoder", choices=ENCODER_KINDS, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--pretrained", default=None)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--metadata", default="data/hiergeo_metadata.csv")
    ap.add_argument("--mode", choices=ROUTING_MODES, default="country_city")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    encoder = load_encoder(args.encoder, args.model, device=args.device, checkpoint=args.checkpoint, pretrained=args.pretrained)
    index = ShardedImageIndex.load(args.embeddings, args.metadata, device=encoder.device)
    with open(args.metadata, newline="", encoding="utf-8") as handle:
        rows = [(r["image_path"], r.get("caption", "")) for r in csv.DictReader(handle)]
    retriever = HierGeoRetriever(index, encoder)

    for query in args.query:
        result = retriever.search(query, top_k=args.top_k, mode=args.mode)
        r = result.route
        where = {"city": f"{r.city}, {r.country}", "country": r.country, "global": "whole corpus"}[r.scope]
        note = f" (named '{r.mentioned_country}', not in the corpus)" if r.scope == "global" and r.mentioned_country else ""
        print(f"\nquery : {query}\nroute : {r.scope} -> {where}{note}; {result.num_candidates:,} candidates")
        for rank, (image_id, score) in enumerate(zip(result.image_ids, result.scores), start=1):
            path, caption = rows[image_id]
            print(f"  {rank:2d}. {score:.3f}  {path}\n      {caption}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
