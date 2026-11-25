"""Contrastive fine-tuning of CLIP / SigLIP / SigLIP2 on HierGeo-style captions.

Input is a TSV file with one ``<image_path>\\t<caption>`` pair per line. Lines that
share an image path are grouped, and every caption of a group is a positive for
every image of the group (``--grouping image``). With ``--grouping scene`` all
patches cut from the same Core-Five scene (file names ``<scene>_y<row>_x<col>.png``)
form one group.

Objective (Sec. 5.1 / App. A of the paper): a multi-positive sigmoid loss,

    logits[i, j] = exp(t) * <v_i, u_j> + b,   target[i, j] = [group_i == group_j],
    loss = BCEWithLogits(logits, target)       (mean over the local batch),

with the model's own learnable temperature ``t`` and bias ``b`` (CLIP has no bias).
Negatives come from the local per-GPU batch.

Run with ``torchrun`` for multiple GPUs. The forward pass goes through the
``DistributedDataParallel`` wrapper, so gradients are averaged across GPUs.
(The checkpoints reported in the paper were trained with an earlier script that
bypassed this wrapper; see the Limitations section of the paper.)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True
_SCENE_RE = re.compile(r"^(.+?)_y\d+_x\d+\.(?:png|jpe?g|tif{1,2})$", re.IGNORECASE)
NO_DECAY = ("bias", "layernorm.weight", "layer_norm.weight", "norm.weight", "logit_scale", "logit_bias")


# --------------------------------------------------------------------------- utils
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def world() -> int:
    return dist.get_world_size() if is_dist() else 1


def log(msg: str) -> None:
    if rank() == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scene_id(path: str) -> str:
    name = os.path.basename(path)
    match = _SCENE_RE.match(name)
    return match.group(1) if match else os.path.splitext(name)[0].split("_")[0]


# --------------------------------------------------------------------------- data
class CaptionDataset(Dataset):
    """Images grouped with all of their captions."""

    def __init__(self, tsv: str, image_root: str = "", grouping: str = "image", lowercase: bool = True,
                 max_captions: int = 0, check_files: bool = True) -> None:
        captions: Dict[str, List[str]] = {}
        skipped = 0
        with open(tsv, encoding="utf-8") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if "\t" not in line:
                    skipped += 1
                    continue
                path, text = line.split("\t", 1)
                path, text = path.strip(), text.strip()
                if lowercase:
                    text = text.lower()
                if not path or not text:
                    skipped += 1
                    continue
                full = os.path.join(image_root, path) if image_root and not os.path.isabs(path) else path
                if check_files and not os.path.exists(full):
                    skipped += 1
                    continue
                captions.setdefault(full, []).append(text)
        if max_captions > 0:
            captions = {k: v[:max_captions] for k, v in captions.items()}
        self.paths = list(captions)
        self.captions = captions
        if grouping == "scene":
            scenes: Dict[str, int] = {}
            self.group = [scenes.setdefault(scene_id(p), len(scenes)) for p in self.paths]
        elif grouping == "image":
            self.group = list(range(len(self.paths)))
        else:
            raise ValueError("grouping must be 'image' or 'scene'")
        self.skipped = skipped

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Optional[Tuple[Image.Image, List[str], int]]:
        try:
            image = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            return None
        return image, self.captions[self.paths[i]], self.group[i]


@dataclass
class Collator:
    processor: Any
    max_length: int
    image_size: int
    is_clip: bool

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        images, texts, groups = [], [], []
        for image, caps, group in batch:
            for text in caps:  # one (image, caption) pair per caption
                images.append(image)
                texts.append(text)
                groups.append(group)
        if not images:
            return None
        size = {"height": self.image_size, "width": self.image_size}
        image_kwargs = {"size": {"shortest_edge": self.image_size}, "crop_size": size} if self.is_clip else {"size": size}
        pixels = self.processor.image_processor(images=images, return_tensors="pt", **image_kwargs)["pixel_values"]
        tok = self.processor.tokenizer(texts, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        out = {"pixel_values": pixels, "input_ids": tok["input_ids"], "group_ids": torch.tensor(groups)}
        if self.is_clip and "attention_mask" in tok:
            out["attention_mask"] = tok["attention_mask"]
        return out


# --------------------------------------------------------------------------- model
def contrastive_logits(model: torch.nn.Module, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    base = model.module if isinstance(model, DDP) else model
    sim = F.normalize(image_emb, dim=-1) @ F.normalize(text_emb, dim=-1).t()
    logits = sim * base.logit_scale.exp().clamp(max=100.0)
    bias = getattr(base, "logit_bias", None)
    return logits + bias if isinstance(bias, torch.Tensor) else logits


def multi_positive_loss(logits: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    target = (group_ids[:, None] == group_ids[None, :]).to(logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, target)


def forward_embeddings(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Joint forward through the (possibly DDP-wrapped) model so gradients are synchronised."""
    kwargs = {"input_ids": batch["input_ids"], "pixel_values": batch["pixel_values"]}
    if "attention_mask" in batch:
        kwargs["attention_mask"] = batch["attention_mask"]
    out = model(**kwargs, return_loss=False)
    return out.image_embeds, out.text_embeds


def param_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if any(t in name.lower() for t in NO_DECAY) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


# --------------------------------------------------------------------------- loops
def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def validate(model, loader, device) -> float:
    model.eval()
    total = torch.zeros(2, device=device, dtype=torch.float64)
    for batch in loader:
        if batch is None:
            continue
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            img, txt = forward_embeddings(model, batch)
            loss = multi_positive_loss(contrastive_logits(model, img, txt).float(), batch["group_ids"])
        n = batch["input_ids"].shape[0]
        total += torch.tensor([loss.item() * n, n], device=device, dtype=torch.float64)
    if is_dist():
        dist.all_reduce(total)
    model.train()
    return (total[0] / total[1].clamp_min(1)).item()


def params_checksum(model: torch.nn.Module) -> float:
    base = model.module if isinstance(model, DDP) else model
    return float(sum(p.detach().double().sum().item() for p in base.parameters()))


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="HierGeo contrastive fine-tuning")
    ap.add_argument("--model_id", required=True, help="e.g. google/siglip2-large-patch16-512")
    ap.add_argument("--train_tsv", required=True)
    ap.add_argument("--val_tsv", default=None)
    ap.add_argument("--image_root", default="", help="prefix for relative image paths in the TSVs")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--grouping", choices=["image", "scene"], default="image")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--per_gpu_batch", type=int, default=128, help="images per GPU per step")
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--image_size", type=int, default=0, help="0 = the checkpoint's native resolution")
    ap.add_argument("--text_max_len", type=int, default=0, help="0 = 77 for CLIP, 64 for SigLIP")
    ap.add_argument("--max_captions", type=int, default=0)
    ap.add_argument("--attn", default="sdpa", help="sdpa | flash_attention_2 | eager")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=0, help="stop after this many optimizer steps (debugging)")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--check_sync", action="store_true", help="assert that all ranks hold identical weights after each epoch")
    args = ap.parse_args(argv)

    if "RANK" in os.environ:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=timedelta(hours=1))
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(args.seed + rank())
    torch.backends.cuda.matmul.allow_tf32 = True

    from transformers import AutoModel, AutoProcessor, get_cosine_schedule_with_warmup
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id, attn_implementation=args.attn).to(device)
    is_clip = model.config.model_type == "clip"
    image_size = args.image_size or int(model.config.vision_config.image_size)
    text_max_len = args.text_max_len or (77 if is_clip else int(model.config.text_config.max_position_embeddings))
    collate = Collator(processor, text_max_len, image_size, is_clip)

    train_set = CaptionDataset(args.train_tsv, args.image_root, args.grouping, max_captions=args.max_captions)
    train_sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed) if is_dist() else None
    train_loader = DataLoader(train_set, batch_size=args.per_gpu_batch, sampler=train_sampler, shuffle=train_sampler is None,
                              num_workers=args.num_workers, collate_fn=collate, pin_memory=True, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = None
    if args.val_tsv:
        val_set = CaptionDataset(args.val_tsv, args.image_root, args.grouping, max_captions=args.max_captions)
        val_sampler = DistributedSampler(val_set, shuffle=False) if is_dist() else None
        val_loader = DataLoader(val_set, batch_size=args.per_gpu_batch, sampler=val_sampler, shuffle=False,
                                num_workers=max(1, args.num_workers // 2), collate_fn=collate, pin_memory=True)
    log(f"model={args.model_id} ({'CLIP' if is_clip else 'SigLIP'}) image_size={image_size} text_len={text_max_len}")
    log(f"train images={len(train_set):,} (skipped lines {train_set.skipped:,}), groups={len(set(train_set.group)):,}, world={world()}")

    if is_dist():
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.98), eps=1e-8)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs if not args.max_steps else min(args.max_steps, steps_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    out_dir = Path(args.out_dir)
    if rank() == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    history: List[Dict[str, float]] = []
    best = float("inf")
    step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        running, seen, t0 = 0.0, 0, time.time()
        optimizer.zero_grad(set_to_none=True)  # drop any incomplete accumulation from the last epoch
        for it, batch in enumerate(train_loader):
            if batch is None:
                continue
            batch = to_device(batch, device)
            last_micro = (it + 1) % args.grad_accum == 0
            sync_ctx = model.no_sync() if (isinstance(model, DDP) and not last_micro) else _null()
            with sync_ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    img, txt = forward_embeddings(model, batch)
                    loss = multi_positive_loss(contrastive_logits(model, img, txt).float(), batch["group_ids"])
                (loss / args.grad_accum).backward()
            running += loss.item()
            seen += 1
            if not last_micro:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                log(f"epoch {epoch} step {step}/{total_steps} loss {running / seen:.5f} lr {scheduler.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)")
            if args.max_steps and step >= args.max_steps:
                break

        val = validate(model, val_loader, device) if val_loader is not None else float("nan")
        train_loss = running / max(seen, 1)
        log(f"epoch {epoch}: train_loss {train_loss:.5f} val_loss {val:.5f}")
        if args.check_sync and is_dist():
            sums = [None] * world()
            dist.all_gather_object(sums, params_checksum(model))
            log(f"weight checksums per rank: {sums}")
            if max(sums) - min(sums) > 1e-6 * max(1.0, abs(sums[0])):
                raise RuntimeError("ranks diverged: gradients are not synchronised")
        history.append({"epoch": epoch, "step": step, "train_loss": train_loss, "val_loss": val})
        if rank() == 0:
            base = model.module if isinstance(model, DDP) else model
            ckpt = out_dir / f"epoch_{epoch}"
            base.save_pretrained(ckpt)
            processor.save_pretrained(ckpt)
            with open(out_dir / "metrics.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            if val < best:
                best = val
                (out_dir / "best_epoch.txt").write_text(f"{epoch}\n")
        if is_dist():
            dist.barrier()
        if args.max_steps and step >= args.max_steps:
            break
    log(f"done; best validation loss {best:.5f}")
    if is_dist():
        dist.destroy_process_group()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
