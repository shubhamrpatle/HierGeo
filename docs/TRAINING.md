# Training

[`hiergeo/train.py`](../hiergeo/train.py) fine-tunes a Hugging Face CLIP, SigLIP or SigLIP2 checkpoint with the multi-positive sigmoid loss of the paper.

## Objective

For a local batch of image–caption pairs, let:

- $v_i$ and $u_j$ be the L2-normalised image and text embeddings;
- $t$ and $b$ be the model's learnable temperature and bias (CLIP has no bias).

Then:

```
logits[i, j] = exp(t) * <v_i, u_j> + b
target[i, j] = 1 if pair i and pair j belong to the same group, else 0
loss         = BCEWithLogits(logits, target)
```

How groups are formed:

- **`--grouping image`:** a group is one image with all of its captions.
- **`--grouping scene`:** a group is every patch cut from the same Core-Five scene. Patch file names follow `<scene>_y<row>_x<col>.png`.

Negatives come from the local per-GPU batch.

## Input

The trainer reads two TSV files, one for training and one for validation, each with lines `<image_path>\t<caption>`.

- **Image paths:** relative paths are resolved against `--image_root`.
- **Captions:** lower-cased.
- **Repeated paths:** an image that appears on several lines is grouped with all of its captions.

## Paper settings (Tab. 8)

| | CLIP | SigLIP | SigLIP2 |
| --- | --- | --- | --- |
| Initialisation | `openai/clip-vit-large-patch14` | `google/siglip-large-patch16-384` | `google/siglip2-large-patch16-512` |
| Resolution | 224 | 384 | 512 |
| HierGeo-Data caption text | place names removed, first category | full | full |
| Records (HierGeo-Data + SkyScript) | 415,200 + 500,642 | 419,554 + 497,549 | 419,554 + 497,549 |
| Grouping | image | image | scene |
| LR / weight decay | 8e-6 / 0.05 | 5e-6 / 0.05 | 5e-6 / 0.05 |
| Images per GPU × accumulation | 24 × 2 | 128 × 2 | 128 × 2 |
| GPUs | 3 × RTX A6000 | 4 × RTX A6000 | 4 × RTX A6000 |
| Epochs run / selected (lowest validation loss) | 12 / 3 | 6 / 1 | 9 / 2 |

All three runs used:

- AdamW with β = (0.9, 0.98) and ε = 1e-8;
- no weight decay on biases, norms and the logit scale and bias;
- cosine decay with 5% warm-up;
- bf16 autocast;
- gradient clipping at 1.0;
- no image augmentation for CLIP and SigLIP2 (the setting was not recorded for the SigLIP run). The trainer here never augments.

The launchers reproduce these settings:

```bash
bash scripts/train_siglip2.sh data/train.tsv data/val.tsv checkpoints/siglip2_ft data/images
bash scripts/train_siglip.sh  data/train.tsv data/val.tsv checkpoints/siglip_ft  data/images
bash scripts/train_clip.sh    data/train_noplace.tsv data/val_noplace.tsv checkpoints/clip_ft data/images
```

Each epoch is saved as `epoch_<n>/`, in Hugging Face format with the processor. Losses go to `metrics.csv`, and the epoch with the lowest validation loss is written to `best_epoch.txt`. Select the checkpoint on validation loss only, before looking at any test metric.

## Note on the paper checkpoints

The checkpoints evaluated in the paper were trained with an earlier script that called the two encoder towers directly instead of through the `DistributedDataParallel` wrapper. As a result:

- Gradients were **not** synchronised across GPUs.
- Each saved checkpoint is a single replica that saw an effective optimiser batch of 256 images (SigLIP/SigLIP2).

This trainer runs the joint forward pass through the wrapper, so gradients are averaged across GPUs. Pass `--check_sync` to assert after every epoch that all ranks hold identical weights.

A corrected run may behave differently from the paper checkpoints. See the Limitations section of the paper.

## Quick check

The following runs on two GPUs in under a minute and prints identical weight checksums on both ranks:

```bash
torchrun --standalone --nproc_per_node=2 -m hiergeo.train \
  --model_id google/siglip-base-patch16-224 --train_tsv small_train.tsv --val_tsv small_val.tsv \
  --out_dir /tmp/hiergeo_smoke --epochs 1 --per_gpu_batch 16 --grad_accum 2 --check_sync
```

## Troubleshooting

- **Training hangs while `DistributedDataParallel` starts.** On some multi-GPU machines, NCCL peer-to-peer transfers hang. Set `NCCL_P2P_DISABLE=1` before launching. We needed this for CLIP-L on our RTX A6000 server.
- **Not enough GPU memory.** Lower `--per_gpu_batch` and raise `--grad_accum`. The loss only sees the local batch, so this also changes the number of negatives.
