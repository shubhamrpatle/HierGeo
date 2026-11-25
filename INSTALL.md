# Installation

## Environment

HierGeo needs Python ≥ 3.9 and PyTorch ≥ 2.1. SigLIP2 needs Transformers ≥ 4.49.

The code was tested with:

- Python 3.10
- PyTorch 2.6.0 (CUDA 12.4)
- Transformers 5.0.0
- OpenCLIP 3.2.0
- NVIDIA RTX A6000 GPUs

```bash
conda create -n hiergeo python=3.10 -y
conda activate hiergeo

pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"      # installs the `hiergeo` package and pytest
```

The `requirements.txt` above installs the default PyTorch build. If your system uses a different CUDA or ROCm version, first install the matching PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/), then run the commands above.

[FlashAttention-2](https://github.com/Dao-AILab/flash-attention) is optional. The paper's SigLIP runs used it (`--attn flash_attention_2`). Every script defaults to PyTorch SDPA, which needs no extra install.

## Check the installation

Run this from the repository root:

```bash
python -c "import torch, transformers, open_clip, hiergeo; print(torch.__version__, transformers.__version__, hiergeo.__version__)"
pytest -q
```

The unit tests run on CPU in a few seconds. They cover:

- the place parser, including the 24 probe queries of App. C of the paper;
- the ranking metrics and the paired bootstrap;
- the sharded search;
- the metadata baselines;
- the training loss.

## Model checkpoints

The scripts accept any Hugging Face id or local directory:

| Model | `--encoder` | `--model` |
| --- | --- | --- |
| CLIP ViT-L/14 (224 px) | `hf_clip` | `openai/clip-vit-large-patch14` |
| SigLIP-L/16 (384 px) | `hf_siglip` | `google/siglip-large-patch16-384` |
| SigLIP2-L/16 (512 px) | `hf_siglip` | `google/siglip2-large-patch16-512` |
| RemoteCLIP ViT-L/14 | `open_clip` | `ViT-L-14` with `--checkpoint RemoteCLIP-ViT-L-14.pt` ([download](https://huggingface.co/chendelong/RemoteCLIP)) |
| SkyCLIP-30 ViT-L/14 | `open_clip` | `ViT-L-14` with `--checkpoint <SkyCLIP .pt>` ([SkyScript](https://github.com/wangzhecheng/SkyScript)) |

Our fine-tuned checkpoints will be linked here when they are released.
