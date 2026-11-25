#!/usr/bin/env python3
"""Zero-shot classification (7 datasets) and RSICD/RSITMD retrieval for one model.

The benchmark files are the SkyScript releases (see docs/EVALUATION.md):

    python scripts/evaluate_external.py --encoder hf_siglip \
        --model checkpoints/hiergeo-siglip2-l16-512 \
        --cls-root benchmarks/classification --ret-root benchmarks/retrieval \
        --out results/external/siglip2_ft

Writes one JSON per dataset plus ``summary.json`` and prints a Markdown row.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hiergeo.eval.classification import SKYSCRIPT_CLS, zero_shot_classify  # noqa: E402
from hiergeo.eval.rs_retrieval import SKYSCRIPT_RET, evaluate_retrieval  # noqa: E402
from hiergeo.models import ENCODER_KINDS, load_encoder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", choices=ENCODER_KINDS, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--pretrained", default=None)
    ap.add_argument("--cls-root", default=None, help="directory holding aid/, eurosat/, fmow/, ...")
    ap.add_argument("--ret-root", default=None, help="directory holding RSICD/ and RSITMD/")
    ap.add_argument("--datasets", nargs="+", default=list(SKYSCRIPT_CLS) + list(SKYSCRIPT_RET))
    ap.add_argument("--prompt", default="a satellite image of {}")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--image-size", type=int, default=None, help="default: native resolution")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    encoder = load_encoder(args.encoder, args.model, device=args.device, image_size=args.image_size,
                           checkpoint=args.checkpoint, pretrained=args.pretrained, dtype=args.dtype)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in args.datasets:
        if name in SKYSCRIPT_CLS:
            if not args.cls_root:
                continue
            csv_rel = SKYSCRIPT_CLS[name]
            csv_path = os.path.join(args.cls_root, csv_rel)
            result = zero_shot_classify(
                encoder, csv_path, os.path.join(os.path.dirname(csv_path), "classnames.txt"),
                data_root=os.path.join(args.cls_root, name), prompt=args.prompt, batch_size=args.batch_size,
            )
            summary[name] = result["top1"]
            print(f"{name:11s} top-1 {result['top1']:.2f}% ({result['num_images']} images)", flush=True)
        elif name in SKYSCRIPT_RET:
            if not args.ret_root:
                continue
            csv_rel, image_dir, split_col = SKYSCRIPT_RET[name]
            result = evaluate_retrieval(encoder, os.path.join(args.ret_root, csv_rel), os.path.join(args.ret_root, image_dir),
                                        split_col=split_col, batch_size=args.batch_size)
            summary[name] = result
            print(f"{name:11s} I->T {result['image_to_text']}  T->I {result['text_to_image']}  mean {result['mean_recall']:.2f}", flush=True)
        else:
            raise ValueError(f"unknown dataset {name}")
        (out / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    cls = [summary[n] for n in SKYSCRIPT_CLS if n in summary]
    if len(cls) == len(SKYSCRIPT_CLS):
        summary["classification_mean"] = round(sum(cls) / len(cls), 2)
    ret = [summary[n]["mean_recall"] for n in SKYSCRIPT_RET if n in summary]
    if len(ret) == len(SKYSCRIPT_RET):
        summary["retrieval_mean"] = round(sum(ret) / len(ret), 2)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if "classification_mean" in summary:
        print(f"classification mean: {summary['classification_mean']:.2f}")
    if "retrieval_mean" in summary:
        print(f"retrieval mean recall: {summary['retrieval_mean']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
