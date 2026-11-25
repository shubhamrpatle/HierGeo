#!/usr/bin/env bash
# Embed the corpus on several GPUs, then merge.
# Usage: GPUS=0,1,2,3 bash scripts/build_index.sh OUT_NPY ENCODER MODEL [extra args for build_index.py]
set -euo pipefail
OUT=${1:?output .npy}; ENC=${2:?encoder kind}; MODEL=${3:?model}; shift 3
IFS=',' read -r -a GPU <<< "${GPUS:-0}"
mkdir -p "$(dirname "${OUT}")"
for i in "${!GPU[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPU[$i]}" python scripts/build_index.py --encoder "${ENC}" --model "${MODEL}" \
    --out "${OUT}" --shard "${i}" --num-shards "${#GPU[@]}" "$@" &
done
wait
python scripts/build_index.py --out "${OUT}" --merge "$@"
rm -f "${OUT}".shard*.npz
