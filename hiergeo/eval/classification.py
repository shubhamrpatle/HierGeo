"""Zero-shot scene classification on the SkyScript benchmark files (Tab. 4).

Each dataset directory follows the SkyScript release format: a CSV with columns
``filepath,label`` (``filepath`` relative to ``--data-root``, ``label`` an integer
class index) and a ``classnames.txt`` with one class name per line.

Protocol used in the paper: one prompt, ``"a satellite image of {}"``; weights in
``bfloat16``; images resized to the model's native resolution (224 for CLIP and
the OpenCLIP baselines, 384 for SigLIP-L/16, 512 for SigLIP2-L/16); top-1 accuracy.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

Image.MAX_IMAGE_PIXELS = None  # some FMoW images are very large

# file layout of the SkyScript benchmark release
SKYSCRIPT_CLS = {
    "aid": "aid/aid/aid_img_txt_pairs_test.csv",
    "eurosat": "eurosat/eurosat/eurosat_img_txt_pairs_test.csv",
    "fmow": "fmow/fmow/fmow_img_txt_pairs_val_rgb.csv",
    "millionaid": "millionaid/millionaid/img_txt_pairs_train.csv",
    "nwpu": "nwpu/nwpu/img_txt_pairs_train.csv",
    "patternnet": "patternnet/patternnet/img_txt_pairs_train.csv",
    "rsicb256": "rsicb256/rsicb256/img_txt_pairs_train.csv",
}


class _Images(Dataset):
    def __init__(self, paths: Sequence[str], encoder) -> None:
        self.paths, self.encoder = list(paths), encoder

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            image = Image.open(self.paths[i]).convert("RGB")
            return self.encoder.preprocess([image])[0], i, 1
        except Exception:
            size = self.encoder.image_size
            return torch.zeros(3, size, size), i, 0


@torch.no_grad()
def class_text_embeddings(encoder, class_names: Sequence[str], prompt: str, batch_size: int = 256) -> torch.Tensor:
    prompts = [prompt.format(c) for c in class_names]
    chunks = [encoder.text_features(prompts[i : i + batch_size], padding="max_length") for i in range(0, len(prompts), batch_size)]
    return F.normalize(torch.cat(chunks).to(encoder.dtype), dim=-1)


@torch.no_grad()
def zero_shot_classify(
    encoder,
    csv_path: str,
    classnames_path: str,
    data_root: str,
    prompt: str = "a satellite image of {}",
    batch_size: int = 64,
    num_workers: int = 8,
) -> Dict[str, object]:
    with open(classnames_path, encoding="utf-8") as handle:
        class_names = [line.strip() for line in handle if line.strip()]
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    paths = [os.path.join(data_root, row["filepath"]) for row in rows]
    labels = [int(row["label"]) for row in rows]

    text = class_text_embeddings(encoder, class_names, prompt)
    loader = DataLoader(_Images(paths, encoder), batch_size=batch_size, num_workers=num_workers, shuffle=False,
                        pin_memory=encoder.device.startswith("cuda"))
    feats: List[torch.Tensor] = []
    kept: List[int] = []
    for pixels, idx, ok in loader:
        ok = ok.bool()
        if ok.any():
            feats.append(encoder.image_features(pixels[ok]).to(encoder.dtype).cpu())
            kept.extend(idx[ok].tolist())
    images = F.normalize(torch.cat(feats), dim=-1).to(encoder.device)
    y = torch.tensor([labels[i] for i in kept], device=encoder.device)
    pred = (images @ text.t()).argmax(-1)
    correct = (pred == y).float()
    per_class = {
        name: round(100.0 * correct[y == c].mean().item(), 4) for c, name in enumerate(class_names) if (y == c).any()
    }
    return {
        "top1": round(100.0 * correct.mean().item(), 4),
        "num_images": len(kept),
        "num_unreadable": len(paths) - len(kept),
        "num_classes": len(class_names),
        "prompt": prompt,
        "per_class_top1": per_class,
    }
