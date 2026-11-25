"""Zero-shot cross-modal retrieval on RSICD and RSITMD (Tab. 5).

Follows the SkyScript protocol: test split of the SkyScript CSV release, every
caption prefixed with ``"a satellite image. "``, duplicate captions merged (a
caption shared by several images is correct for all of them), and Recall@1/5/10
in both directions. ``mean_recall`` is the mean of the six recalls.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Dict, List, Sequence, Set

import torch
import torch.nn.functional as F
from PIL import Image

# file layout of the SkyScript retrieval release; RSITMD has a "split" column
SKYSCRIPT_RET = {
    "rsicd": ("RSICD/RSICD/RSICD_img_txt_pairs_test.csv", "RSICD", None),
    "rsitmd": ("RSITMD/RSITMD/RSITMD_img_txt_pairs_test.csv", "RSITMD", "split"),
}
CAPTION_PREFIX = "a satellite image. "


def load_pairs(csv_path: str, split_col=None, prefix: str = CAPTION_PREFIX):
    images: List[str] = []
    captions: List[str] = []
    cap_index: Dict[str, int] = {}
    img_caps: Dict[str, List[str]] = defaultdict(list)
    seen = set()
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if split_col and row.get(split_col, "test") != "test":
                continue
            path, cap = row["filepath"], prefix + row["title"].strip()
            if (path, cap) in seen:
                continue
            seen.add((path, cap))
            if path not in img_caps:
                images.append(path)
            img_caps[path].append(cap)
    for path in images:
        for cap in img_caps[path]:
            if cap not in cap_index:
                cap_index[cap] = len(captions)
                captions.append(cap)
    img2txt = [{cap_index[c] for c in img_caps[p]} for p in images]
    txt2img: List[Set[int]] = [set() for _ in captions]
    for i, p in enumerate(images):
        for c in img_caps[p]:
            txt2img[cap_index[c]].add(i)
    return images, captions, img2txt, txt2img


def recall_at(scores: torch.Tensor, truth: Sequence[Set[int]], ks=(1, 5, 10)) -> Dict[str, float]:
    order = scores.argsort(dim=1, descending=True)[:, : max(ks)].cpu().tolist()
    return {f"R@{k}": round(100.0 * sum(any(j in t for j in row[:k]) for row, t in zip(order, truth)) / len(truth), 2) for k in ks}


@torch.no_grad()
def evaluate_retrieval(encoder, csv_path: str, image_dir: str, split_col=None, batch_size: int = 64) -> Dict[str, object]:
    images, captions, img2txt, txt2img = load_pairs(csv_path, split_col)
    feats = []
    for start in range(0, len(images), batch_size):
        batch = [Image.open(os.path.join(image_dir, p)).convert("RGB") for p in images[start : start + batch_size]]
        feats.append(encoder.image_features(encoder.preprocess(batch)).to(encoder.dtype))
    img = F.normalize(torch.cat(feats), dim=-1)
    txt = F.normalize(
        torch.cat([encoder.text_features(captions[i : i + 256], padding="max_length").to(encoder.dtype) for i in range(0, len(captions), 256)]),
        dim=-1,
    )
    i2t = recall_at(img @ txt.t(), img2txt)
    t2i = recall_at(txt @ img.t(), txt2img)
    mean = round((sum(i2t.values()) + sum(t2i.values())) / 6, 2)
    return {"images": len(images), "captions": len(captions), "image_to_text": i2t, "text_to_image": t2i, "mean_recall": mean}
