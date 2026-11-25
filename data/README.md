# HierGeo data

> **Download:** the dataset release (captions, metadata, benchmark queries and relevance judgments) will be linked here.

## What is released

HierGeo is released as a **pointer release**. It contains:

- captions;
- geographic and category metadata;
- benchmark queries and relevance judgments;
- identifiers that locate every patch in the source [Core-Five](https://huggingface.co/datasets/gajeshladhar/core-five) tiles.

Pixels are not redistributed. Please obtain the imagery from Core-Five under its own terms.

The POI-derived fields come from [Foursquare Open Source Places](https://opensource.foursquare.com/os-places/) (Apache 2.0).

## Expected layout

The scripts default to the following paths. Pass `--metadata`, `--queries`, `--qrels` and `--image-root` to use others.

```
data/
├── hiergeo_metadata.csv        # one row per patch; row order defines the image id
├── bench/
│   ├── queries.jsonl           # 23,470 HierGeo-Bench queries
│   └── qrels.jsonl             # relevance judgments (relevant image ids per query)
└── images/                     # optional: reconstructed 512x512 patches
```

### `hiergeo_metadata.csv`

| Column | Example |
| --- | --- |
| `image_path` | `images_batche_1/00964b34_y1536_x1024.png` (`<scene>_y<row>_x<col>`: Core-Five scene id and pixel offset of the patch) |
| `caption` | `a satellite image of Residential with residential area and apartment building in Rio das Ostras, Brazil.` |
| `city`, `country` | `Rio das Ostras`, `Brazil` |
| `poi_type` | `Residential` (one of 10 scene categories) |
| `multi_labels` | `Residential, residential area, apartment building` |

Image ids used everywhere else (embedding rows, qrels, rankings) are the **0-based row numbers** of this file. The corpus has:

- 464,553 rows;
- 140 countries;
- 2,460 country–city pairs.

### `bench/queries.jsonl`

```json
{"query_id": "b3d52e4c...", "stage": "stage1_country", "query_text": "Transportation in Afghanistan",
 "query_category_generic": "Transportation", "country": "Afghanistan", "city": null}
```

The `stage` field takes one of three values:

| `stage` | Queries | With positives |
| --- | ---: | ---: |
| `stage1_country` | 2,129 | 1,281 |
| `stage2_city` | 19,502 | 1,815 |
| `stage3_specialized` | 1,839 | 1,358 |

Fields used by the code:

- **`query_text`:** the text the place parser reads.
- **`query_category_generic`:** the text the vision–language model embeds.
- **`country` and `city`:** used only by the metadata baselines and for per-country reporting.

### `bench/qrels.jsonl`

```json
{"query_id": "b3d52e4c...", "has_positive": true, "relevant_image_ids": [318143]}
```

Only queries with `has_positive = true` (4,454) are scored.

Two kinds of query never have positives:

- **714 country-stage queries** name a country that is absent from the corpus. One example is Côte d'Ivoire, which the corpus metadata spells "Ivory Coast". The place parser sends these queries to global search.
- **Some city-stage queries** name a city that has no matching patches.

## Training TSVs

[`hiergeo/train.py`](../hiergeo/train.py) reads TSV files with one `<image_path>\t<caption>` pair per line. Repeated image paths are grouped as positives. The paper mixed HierGeo-Data training patches with [SkyScript](https://github.com/wangzhecheng/SkyScript) records. Its training and validation splits are scene-disjoint: no Core-Five scene appears in both.
