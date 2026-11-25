"""Metadata-only shortcut baselines for HierGeo-Bench (Tab. 6, top block).

These rankers never look at pixels. They use only the fields a user can see
(the country, the city, and the category text of the query) and the corpus
metadata columns ``country``, ``city``, ``poi_type`` and ``multi_labels``. Qrels
are used only for scoring. Because HierGeo-Bench relevance is derived from the
same metadata family, strong scores here are a *shortcut diagnostic*, not
evidence of visual understanding.

Strategies
----------
``geo_random``       random order inside the country (stage 1) or city shard.
``geo_poi_exact``    shard images whose ``poi_type`` equals the query category first
                     (random order), then the rest of the shard.
``geo_poi_lexical``  rank shard images by token overlap between the query category
                     and ``poi_type`` + ``multi_labels`` (+100 for an exact
                     ``poi_type`` match), random tie-breaking, then fill from the shard.
"""

from __future__ import annotations

import csv
import hashlib
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

from .metrics import mean_metrics, query_metrics

STRATEGIES = ("geo_random", "geo_poi_exact", "geo_poi_lexical")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def norm(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def tokens(value: object) -> Set[str]:
    return set(_TOKEN_RE.findall(norm(value)))


def stable_rng(seed: int, query_id: str, strategy: str) -> random.Random:
    payload = f"{seed}\0{query_id}\0{strategy}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


def _shuffled(values: Iterable[int], rng: random.Random) -> List[int]:
    out = list(values)
    rng.shuffle(out)
    return out


def _fill_unique(primary: Sequence[int], fallback: Sequence[int], k: int) -> List[int]:
    out: List[int] = []
    seen: Set[int] = set()
    for image_id in list(primary) + list(fallback):
        if image_id not in seen:
            seen.add(image_id)
            out.append(image_id)
            if len(out) >= k:
                break
    return out


class MetadataCorpus:
    def __init__(self, metadata_csv: str | Path) -> None:
        self.country_ids: MutableMapping[str, List[int]] = defaultdict(list)
        self.city_ids: MutableMapping[Tuple[str, str], List[int]] = defaultdict(list)
        self.country_exact: MutableMapping[Tuple[str, str], List[int]] = defaultdict(list)
        self.city_exact: MutableMapping[Tuple[str, str, str], List[int]] = defaultdict(list)
        self.country_tokens: MutableMapping[Tuple, List[int]] = defaultdict(list)
        self.city_tokens: MutableMapping[Tuple, List[int]] = defaultdict(list)
        with open(metadata_csv, newline="", encoding="utf-8") as handle:
            for image_id, row in enumerate(csv.DictReader(handle)):
                country, city, poi = norm(row.get("country")), norm(row.get("city")), norm(row.get("poi_type"))
                if not country:
                    continue
                self.country_ids[country].append(image_id)
                if city:
                    self.city_ids[(country, city)].append(image_id)
                if poi:
                    self.country_exact[(country, poi)].append(image_id)
                    if city:
                        self.city_exact[(country, city, poi)].append(image_id)
                for token in tokens(row.get("poi_type")) | tokens(row.get("multi_labels")):
                    self.country_tokens[(country, token)].append(image_id)
                    if city:
                        self.city_tokens[(country, city, token)].append(image_id)

    def rank(self, query: Mapping, strategy: str, seed: int, k: int = 10) -> List[int]:
        rng = stable_rng(seed, str(query["query_id"]), strategy)
        country, city = norm(query.get("country")), norm(query.get("city"))
        category = norm(query.get("query_category_generic") or query.get("query_text"))
        if city:
            geo = self.city_ids.get((country, city), [])
            exact = self.city_exact.get((country, city, category), [])
            token_index, prefix = self.city_tokens, (country, city)
        else:
            geo = self.country_ids.get(country, [])
            exact = self.country_exact.get((country, category), [])
            token_index, prefix = self.country_tokens, (country,)

        if strategy == "geo_random":
            return _shuffled(geo, rng)[:k]
        if strategy == "geo_poi_exact":
            return _fill_unique(_shuffled(exact, rng), _shuffled(geo, rng), k)
        if strategy != "geo_poi_lexical":
            raise ValueError(f"unknown strategy {strategy!r}")
        scores: MutableMapping[int, int] = defaultdict(int)
        for token in tokens(category):
            for image_id in token_index.get((*prefix, token), []):
                scores[image_id] += 1
        for image_id in exact:
            scores[image_id] += 100
        tie = {image_id: rng.random() for image_id in scores}
        ranked = sorted(scores, key=lambda i: (-scores[i], tie[i], i))
        return _fill_unique(ranked, _shuffled(geo, rng), k)


def run_metadata_baselines(
    corpus: MetadataCorpus,
    queries: Sequence[Mapping],
    qrels: Mapping[str, Mapping],
    seeds: Sequence[int] = tuple(range(10)),
    k: int = 10,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Mean and sample std over seeds, for every strategy and stage (positives only)."""
    positive = [q for q in queries if qrels[q["query_id"]].get("has_positive")]
    per_seed: Dict[Tuple[str, str], List[Dict[str, float]]] = defaultdict(list)
    for seed in seeds:
        for strategy in STRATEGIES:
            by_stage: Dict[str, List[Dict[str, float]]] = defaultdict(list)
            for query in positive:
                retrieved = corpus.rank(query, strategy, seed, k)
                m = query_metrics(retrieved, qrels[query["query_id"]]["relevant_image_ids"], k)
                by_stage[query["stage"]].append(m)
                by_stage["overall"].append(m)
            for stage, rows in by_stage.items():
                per_seed[(strategy, stage)].append(mean_metrics(rows))
    out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = defaultdict(dict)
    for (strategy, stage), runs in per_seed.items():
        out[stage][strategy] = {
            metric: {
                "mean": statistics.fmean(r[metric] for r in runs),
                "std": statistics.stdev([r[metric] for r in runs]) if len(runs) > 1 else 0.0,
            }
            for metric in runs[0]
        }
    return dict(out)
