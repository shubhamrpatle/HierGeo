#!/usr/bin/env python3
"""Embed every corpus image with a dual encoder (one row per metadata row).

    python scripts/build_index.py --encoder hf_siglip \
        --model checkpoints/hiergeo-siglip2-l16-512 \
        --metadata data/hiergeo_metadata.csv --image-root data/images \
        --out indices/siglip2_ft/image_embeddings.npy

Several GPUs: run one process per GPU with ``--shard i --num-shards n`` and then
``--merge`` (see scripts/build_index.sh). Unreadable images get a zero vector and
are listed in ``<out>.missing.txt``; they can never be retrieved.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hiergeo.models import ENCODER_KINDS, load_encoder  # noqa: E402


class _Corpus(Dataset):
    def __init__(self, paths, ids, encoder):
        self.paths, self.ids, self.encoder = paths, ids, encoder

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        gid = self.ids[i]
        try:
            with Image.open(self.paths[gid]) as im:
                return self.encoder.preprocess([im.convert("RGB")])[0], gid, 1
        except Exception:
            s = self.encoder.image_size
            return torch.zeros(3, s, s), gid, 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", choices=ENCODER_KINDS)
    ap.add_argument("--model")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--pretrained", default=None)
    ap.add_argument("--metadata", default="data/hiergeo_metadata.csv")
    ap.add_argument("--image-root", default="", help="prefix for relative image_path values")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true", help="merge <out>.shard*.npz into <out>")
    args = ap.parse_args()

    with open(args.metadata, newline="", encoding="utf-8") as handle:
        rel = [row["image_path"] for row in csv.DictReader(handle)]
    paths = [p if os.path.isabs(p) or not args.image_root else os.path.join(args.image_root, p) for p in rel]

    if args.merge:
        parts = sorted(glob.glob(f"{args.out}.shard*.npz"))
        if not parts:
            raise SystemExit(f"no shards found for {args.out}")
        dim = int(np.load(parts[0])["emb"].shape[1])
        emb = np.zeros((len(paths), dim), dtype=np.float32)
        done = np.zeros(len(paths), dtype=bool)
        for part in parts:
            data = np.load(part)
            emb[data["ids"]] = data["emb"]
            done[data["ids"]] = data["ok"].astype(bool)
        np.save(args.out, emb)
        missing = np.flatnonzero(~done)
        Path(f"{args.out}.missing.txt").write_text("".join(f"{i}\t{rel[i]}\n" for i in missing))
        print(f"saved {args.out} {emb.shape}; {len(missing)} unreadable images")
        return 0

    encoder = load_encoder(args.encoder, args.model, checkpoint=args.checkpoint, pretrained=args.pretrained, dtype="float32")
    ids = list(range(args.shard, len(paths), args.num_shards))
    loader = DataLoader(_Corpus(paths, ids, encoder), batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=encoder.device.startswith("cuda"))
    out_ids, out_emb, out_ok = [], [], []
    for n, (pixels, gid, ok) in enumerate(loader):
        feats = encoder.image_features(pixels, autocast_dtype=torch.bfloat16).float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        feats[~ok.bool().to(feats.device)] = 0.0
        out_ids.append(gid.numpy())
        out_emb.append(feats.cpu().numpy())
        out_ok.append(ok.numpy())
        if n % 50 == 0:
            print(f"[shard {args.shard}] {min((n + 1) * args.batch_size, len(ids))}/{len(ids)}", flush=True)
    target = f"{args.out}.shard{args.shard:02d}.npz" if args.num_shards > 1 else None
    ids_arr, emb_arr, ok_arr = np.concatenate(out_ids), np.concatenate(out_emb), np.concatenate(out_ok)
    if target:
        np.savez(target, ids=ids_arr, emb=emb_arr, ok=ok_arr)
        print(f"saved {target}")
    else:
        emb = np.zeros((len(paths), emb_arr.shape[1]), dtype=np.float32)
        emb[ids_arr] = emb_arr
        np.save(args.out, emb)
        missing = ids_arr[ok_arr == 0]
        Path(f"{args.out}.missing.txt").write_text("".join(f"{i}\t{rel[i]}\n" for i in missing))
        print(f"saved {args.out} {emb.shape}; {len(missing)} unreadable images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
