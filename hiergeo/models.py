"""Thin wrappers that give CLIP, SigLIP/SigLIP2 and OpenCLIP checkpoints one interface.

``load_encoder`` returns a :class:`DualEncoder` with ``encode_text`` and
``encode_images``.

Encoder kinds
-------------
``hf_clip``    Hugging Face ``CLIPModel`` (e.g. ``openai/clip-vit-large-patch14`` or a
               fine-tuned HierGeo CLIP checkpoint). Uses the projected features.
``hf_siglip``  Hugging Face SigLIP or SigLIP2 (e.g. ``google/siglip2-large-patch16-512``
               or a fine-tuned HierGeo checkpoint). Uses the pooled tower outputs.
``open_clip``  An OpenCLIP architecture with an optional ``.pt`` state dict, used for
               RemoteCLIP and SkyCLIP.

Hugging Face models are evaluated at their native resolution by default; images
are resized to ``image_size x image_size`` (HierGeo patches are square, so this
equals the usual resize + centre crop).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ENCODER_KINDS = ("hf_clip", "hf_siglip", "open_clip")
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def _features(output: Any) -> torch.Tensor:
    """Accept both tensors and Hugging Face ``ModelOutput`` objects."""
    if isinstance(output, torch.Tensor):
        return output
    for key in ("text_embeds", "image_embeds", "pooler_output"):
        value = getattr(output, key, None)
        if isinstance(value, torch.Tensor):
            return value
    raise TypeError(f"Unsupported model output: {type(output)}")


@dataclass
class DualEncoder:
    kind: str
    model: Any
    tokenizer: Any
    image_processor: Any
    device: str
    image_size: int
    max_text_length: int
    dtype: torch.dtype = torch.float32

    def _autocast(self, dtype: Optional[torch.dtype]):
        if dtype is not None and self.device.startswith("cuda"):
            return torch.autocast(device_type="cuda", dtype=dtype)
        return contextlib.nullcontext()

    # ------------------------------------------------------------------ text
    @torch.no_grad()
    def text_features(self, texts: Sequence[str], padding: str = "max_length", autocast_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Un-normalised text features for one batch, in the model's dtype.

        ``padding`` matters for SigLIP-family models, whose text tower pools the
        last token: ``"max_length"`` matches how they were trained, ``"longest"``
        pads the batch to its longest text. It has no effect for CLIP models.
        """
        with self._autocast(autocast_dtype):
            if self.kind == "open_clip":
                return self.model.encode_text(self.tokenizer(list(texts)).to(self.device))
            enc = self.tokenizer(
                list(texts), padding=padding, truncation=True, max_length=self.max_text_length, return_tensors="pt"
            ).to(self.device)
            if self.kind == "hf_clip":
                return _features(self.model.get_text_features(**enc))
            return self.model.text_model(**enc).pooler_output

    def encode_text(
        self,
        texts: Sequence[str],
        batch_size: int = 256,
        padding: str = "max_length",
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> np.ndarray:
        """L2-normalised ``float32`` text embeddings as a numpy array."""
        chunks: List[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            feats = self.text_features(texts[start : start + batch_size], padding, autocast_dtype).float()
            chunks.append((feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu())
        return torch.cat(chunks).numpy().astype(np.float32) if chunks else np.zeros((0, 0), np.float32)

    # ---------------------------------------------------------------- images
    def preprocess(self, images: Sequence[Image.Image]) -> torch.Tensor:
        images = [im.convert("RGB") for im in images]
        if self.kind == "open_clip":
            return torch.stack([self.image_processor(im) for im in images])
        size = {"height": self.image_size, "width": self.image_size}
        kwargs = {"size": size, "crop_size": size} if self.kind == "hf_clip" else {"size": size}
        return self.image_processor(images=images, return_tensors="pt", **kwargs)["pixel_values"]

    @torch.no_grad()
    def image_features(self, pixel_values: torch.Tensor, autocast_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Un-normalised image features in the model's dtype."""
        pixel_values = pixel_values.to(self.device).to(self.dtype)
        with self._autocast(autocast_dtype):
            if self.kind == "open_clip":
                return self.model.encode_image(pixel_values)
            if self.kind == "hf_clip":
                return _features(self.model.get_image_features(pixel_values=pixel_values))
            return self.model.vision_model(pixel_values=pixel_values).pooler_output

    def encode_images(self, images: Sequence[Image.Image], batch_size: int = 64, autocast_dtype: Optional[torch.dtype] = None) -> np.ndarray:
        """L2-normalised ``float32`` image embeddings as a numpy array."""
        out = []
        for start in range(0, len(images), batch_size):
            feats = self.image_features(self.preprocess(images[start : start + batch_size]), autocast_dtype).float()
            out.append((feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu())
        return torch.cat(out).numpy().astype(np.float32)


def _load_open_clip_state(model: torch.nn.Module, checkpoint: str) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, _ = model.load_state_dict(state, strict=False)
    if len(missing) > 10:
        raise RuntimeError(f"{checkpoint}: {len(missing)} missing keys; is this the right architecture?")


def load_encoder(
    kind: str,
    model: str,
    device: Optional[str] = None,
    image_size: Optional[int] = None,
    checkpoint: Optional[str] = None,
    pretrained: Optional[str] = None,
    dtype: str | torch.dtype = "float32",
) -> DualEncoder:
    """Load a dual encoder.

    Args:
        kind: one of :data:`ENCODER_KINDS`.
        model: Hugging Face id / local directory, or an OpenCLIP architecture name.
        device: ``"cuda"``, ``"cuda:1"``, ``"cpu"`` ... (default: CUDA if available).
        image_size: evaluation resolution; defaults to the checkpoint's native size.
        checkpoint: OpenCLIP ``.pt`` state dict (RemoteCLIP, SkyCLIP).
        pretrained: OpenCLIP pretrained tag (e.g. ``"openai"``) when no checkpoint is given.
        dtype: weight dtype (``float32`` for index building, ``bfloat16`` for the
            external benchmarks, as in the paper).
    """
    if kind not in ENCODER_KINDS:
        raise ValueError(f"kind must be one of {ENCODER_KINDS}, got {kind!r}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPES[dtype] if isinstance(dtype, str) else dtype
    if not device.startswith("cuda"):
        dtype = torch.float32

    if kind == "open_clip":
        import open_clip

        net, _, preprocess = open_clip.create_model_and_transforms(model, pretrained=None if checkpoint else pretrained)
        if checkpoint:
            _load_open_clip_state(net, checkpoint)
        net = net.to(device).to(dtype).eval()
        size = net.visual.image_size
        size = int(size[0] if isinstance(size, (tuple, list)) else size)
        return DualEncoder(kind, net, open_clip.get_tokenizer(model), preprocess, device, size, 77, dtype)

    from transformers import AutoModel, AutoProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    import transformers

    major = int(transformers.__version__.split(".")[0])
    dtype_kw = {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}
    net = AutoModel.from_pretrained(model, **dtype_kw).to(device).eval()
    processor = AutoProcessor.from_pretrained(model)
    native = int(getattr(net.config.vision_config, "image_size", 224))
    max_len = int(getattr(net.config.text_config, "max_position_embeddings", 77 if kind == "hf_clip" else 64))
    return DualEncoder(
        kind=kind,
        model=net,
        tokenizer=processor.tokenizer,
        image_processor=processor.image_processor,
        device=device,
        image_size=int(image_size or native),
        max_text_length=max_len,
        dtype=dtype,
    )
