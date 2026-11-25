#!/usr/bin/env bash
# Fine-tune SigLIP2-L/16 (512 px) on HierGeo-Data + SkyScript (paper setting, Tab. 8).
# Usage: bash scripts/train_siglip2.sh TRAIN_TSV VAL_TSV OUT_DIR [IMAGE_ROOT]
set -euo pipefail
TRAIN=${1:?train tsv}; VAL=${2:?val tsv}; OUT=${3:?output dir}; ROOT=${4:-}
NGPU=${NGPU:-4}
torchrun --standalone --nproc_per_node="${NGPU}" -m hiergeo.train \
  --model_id google/siglip2-large-patch16-512 \
  --train_tsv "${TRAIN}" --val_tsv "${VAL}" --image_root "${ROOT}" --out_dir "${OUT}" \
  --grouping scene --epochs "${EPOCHS:-9}" --per_gpu_batch 128 --grad_accum 2 \
  --lr 5e-6 --weight_decay 0.05 --warmup_ratio 0.05 --attn "${ATTN:-sdpa}"
