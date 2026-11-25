#!/usr/bin/env bash
# Fine-tune CLIP ViT-L/14 (224 px) on HierGeo-Data (place names removed) + SkyScript
# (paper setting, Tab. 8). Usage: bash scripts/train_clip.sh TRAIN_TSV VAL_TSV OUT_DIR [IMAGE_ROOT]
set -euo pipefail
TRAIN=${1:?train tsv}; VAL=${2:?val tsv}; OUT=${3:?output dir}; ROOT=${4:-}
NGPU=${NGPU:-3}
torchrun --standalone --nproc_per_node="${NGPU}" -m hiergeo.train \
  --model_id openai/clip-vit-large-patch14 \
  --train_tsv "${TRAIN}" --val_tsv "${VAL}" --image_root "${ROOT}" --out_dir "${OUT}" \
  --grouping image --epochs "${EPOCHS:-12}" --per_gpu_batch 24 --grad_accum 2 \
  --lr 8e-6 --weight_decay 0.05 --warmup_ratio 0.05 --attn sdpa
