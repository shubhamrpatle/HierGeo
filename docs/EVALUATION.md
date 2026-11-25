# Evaluation

The commands below assume the data layout of [data/README.md](../data/README.md). Every script prints Markdown tables and writes JSON next to them.

## HierGeo-Bench

### 1. Build an image index

Embed every corpus patch once per model. Rows follow the order of `data/hiergeo_metadata.csv`.

```bash
# single GPU
python scripts/build_index.py --encoder hf_siglip --model google/siglip2-large-patch16-512 \
    --metadata data/hiergeo_metadata.csv --image-root data/images \
    --out indices/siglip2/image_embeddings.npy

# several GPUs (one shard per GPU, then merge)
GPUS=0,1,2,3 bash scripts/build_index.sh indices/siglip2/image_embeddings.npy \
    hf_siglip google/siglip2-large-patch16-512 --metadata data/hiergeo_metadata.csv --image-root data/images
```

How each model family is embedded:

| Model family | Weights | Precision | Image features |
| --- | --- | --- | --- |
| Hugging Face models | fp32 | bf16 autocast | Pooled vision output (SigLIP) or projected image features (CLIP) |
| OpenCLIP models | — | — | `encode_image` with the model's own preprocessing |

### 2. Rank, route and score

```bash
python scripts/evaluate_bench.py --encoder hf_siglip --model google/siglip2-large-patch16-512 \
    --embeddings indices/siglip2/image_embeddings.npy \
    --modes global country country_city --out results/bench/siglip2_pt
```

What the script does:

- **Inputs:** the query's `query_text` goes to the deterministic place parser ([hiergeo/routing.py](../hiergeo/routing.py)), and its `query_category_generic` is embedded with the text encoder.
- **Routing modes:**
  - `global` searches all images;
  - `country` searches the named country;
  - `country_city` searches the named city when one is given, otherwise the country. This is the reference system.
- **Scoring:** top-10 metrics over the 4,454 queries with positives, overall and per stage.
- **Outputs:**
  - `per_query.jsonl`, with the route, candidate count, ranked ids and metrics of every query;
  - `summary.json`, which includes a per-country macro average.

### 3. Metadata baselines and significance

```bash
python scripts/run_metadata_baselines.py --out results/bench/metadata_baselines.json   # ten seeds
python scripts/compare_runs.py \
    --candidate results/bench/clip_ft/country_city/per_query.jsonl \
    --baseline  results/bench/clip_pt/country_city/per_query.jsonl
```

`compare_runs.py` reports the mean nDCG@10 difference with a 95% paired bootstrap interval: 5,000 resamples of queries, seed 20260726.

### Encoder arguments for the paper's models

| Row in Tab. 6 | Arguments |
| --- | --- |
| CLIP ViT-L/14 | `--encoder hf_clip --model openai/clip-vit-large-patch14` |
| RemoteCLIP ViT-L/14 | `--encoder open_clip --model ViT-L-14 --checkpoint RemoteCLIP-ViT-L-14.pt` |
| SigLIP-L/16 | `--encoder hf_siglip --model google/siglip-large-patch16-384` |
| SigLIP2-L/16 | `--encoder hf_siglip --model google/siglip2-large-patch16-512` |
| Fine-tuned rows | the same `--encoder`, with `--model <fine-tuned checkpoint directory>` |

## External benchmarks

The external benchmarks are evaluated on the files released by [SkyScript](https://github.com/wangzhecheng/SkyScript).

Classification datasets:

- AID
- EuroSAT
- FMoW
- MillionAID
- NWPU-RESISC45
- PatternNet
- RSI-CB256

Retrieval datasets: RSICD and RSITMD.

Place them as follows:

```
benchmarks/classification/<name>/<name>/<csv>  +  classnames.txt     (name = aid, eurosat, fmow, ...)
benchmarks/retrieval/RSICD/RSICD/RSICD_img_txt_pairs_test.csv
benchmarks/retrieval/RSITMD/RSITMD/RSITMD_img_txt_pairs_test.csv
```

Image paths in the CSVs are relative to `benchmarks/classification/<name>/` and `benchmarks/retrieval/<NAME>/`. The CSV file name of each dataset is listed in [hiergeo/eval/classification.py](../hiergeo/eval/classification.py).

```bash
python scripts/evaluate_external.py --encoder hf_siglip --model google/siglip2-large-patch16-512 \
    --cls-root benchmarks/classification --ret-root benchmarks/retrieval --out results/external/siglip2_pt
```

The protocol is the same for every model:

| Setting | Value |
| --- | --- |
| Weights | bf16 |
| Resolution | native (224 / 384 / 512) |
| Classification prompt | `"a satellite image of {}"` |
| Retrieval captions | prefixed with `"a satellite image. "` (SkyScript protocol) |
| Duplicate captions | merged |
| Retrieval metric | Recall@1/5/10 in both directions |

## Reproducibility check

We re-ran the released code on the original artifacts (corpus metadata, benchmark files, image embeddings and checkpoints) and compared it with the numbers in the paper.

| What | Result |
| --- | --- |
| HierGeo-Bench rankings, 7 vision–language rankers × 3 routing modes | Top-10 lists identical to the paper's evaluator on **all 4,454 scored queries** (93,534 rankings). All metrics in Tabs. 6 and 7 match. |
| Paired bootstrap, Tab. 11 | All twelve deltas and intervals match. |
| Metadata baselines, Tab. 6 | All match (10 seeds). |
| Place parser on the benchmark | Country extracted for 2,129 / 2,129 country-stage queries. All 714 queries that name a country absent from the corpus go to global search. All 24 probe queries of App. C behave as stated (`pytest`). |
| Index building | Re-embedded patches agree with the archived embeddings (cosine ≥ 0.999). |
| External benchmarks, Tabs. 4 and 5 | Every row except MS-CLIP and CLIP-laion-RS was produced with `scripts/evaluate_external.py`. Run on the same checkpoints, the original evaluation scripts give identical numbers in the same software environment. |

**Library versions matter.** Results from older runs with earlier library versions (Transformers 4.x) differed for some checkpoints. The largest case was fine-tuned CLIP on EuroSAT: 37.00% then, 32.41% with the current stack. We therefore regenerated Tabs. 4 and 5 with one environment: PyTorch 2.6.0 and Transformers 5.0.0.

**Retrieval precision.** Retrieval recalls are computed from bf16 similarity scores, and their ties are broken by `torch.argsort`. Recalls can therefore move by a few tenths of a point across PyTorch and Transformers versions. HierGeo-Bench rankings use fp16 scores and were bit-identical in our checks.
