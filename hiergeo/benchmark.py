"""HierGeo-Bench: loading, running, and scoring.

Benchmark files (JSON lines, one object per query):

``queries.jsonl``  ``query_id``, ``stage`` (``stage1_country`` | ``stage2_city`` |
                   ``stage3_specialized``), ``query_text`` (e.g. "Residential in Luanda,
                   Angola"), ``query_category_generic`` (the category shown to the
                   model, e.g. "Residential"), ``country``, ``city``.
``qrels.jsonl``    ``query_id``, ``has_positive``, ``relevant_image_ids`` (0-based rows
                   of the corpus metadata CSV).

Only queries with at least one relevant image are scored (4,454 of 23,470).
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .metrics import mean_metrics, query_metrics

STAGES = ("stage1_country", "stage2_city", "stage3_specialized")
STAGE_NAMES = {"stage1_country": "Country", "stage2_city": "City", "stage3_specialized": "Specialized"}


def read_jsonl(path: str | Path) -> List[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[Mapping]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_benchmark(queries_path: str | Path, qrels_path: str | Path):
    queries = read_jsonl(queries_path)
    qrels = {row["query_id"]: row for row in read_jsonl(qrels_path)}
    missing = [q["query_id"] for q in queries if q["query_id"] not in qrels]
    if missing:
        raise ValueError(f"{len(missing)} queries have no qrels entry")
    return queries, qrels


def score_rows(rows: List[dict], qrels: Mapping[str, dict], k: int = 10) -> List[dict]:
    """Attach per-query metrics to ranked rows (``retrieved_ids`` field)."""
    for row in rows:
        qrel = qrels[row["query_id"]]
        row["has_positive"] = bool(qrel.get("has_positive"))
        row["num_relevant"] = len(qrel.get("relevant_image_ids") or [])
        row["metrics"] = (
            query_metrics(row["retrieved_ids"], qrel["relevant_image_ids"], k) if row["has_positive"] else {}
        )
    return rows


def summarize(rows: Sequence[dict]) -> Dict[str, dict]:
    """Mean metrics over queries with positives: overall and per stage."""
    scored = [r for r in rows if r.get("has_positive")]
    out = {"overall": {"queries": len(scored), **mean_metrics([r["metrics"] for r in scored])}}
    for stage in STAGES:
        part = [r for r in scored if r["stage"] == stage]
        out[stage] = {"queries": len(part), **mean_metrics([r["metrics"] for r in part])}
    return out


def run_visual(
    index,
    encoder,
    queries: Sequence[dict],
    mode: str = "country_city",
    top_k: int = 10,
    text_field: str = "query_category_generic",
    text_padding: str = "longest",
    text_autocast=None,
    progress: bool = True,
) -> List[dict]:
    """Rank every query with a dual encoder inside its routed shard.

    The defaults (``text_padding="longest"`` over one batch of all distinct
    category texts, ``fp16`` text autocast set by the caller) reproduce the
    evaluator used for the paper's tables.
    """
    from .retrieval import encode_unique_texts

    texts = [str(q.get(text_field) or "") for q in queries]
    vectors = encode_unique_texts(encoder, texts, padding=text_padding, autocast_dtype=text_autocast)
    rows: List[dict] = []
    started = time.perf_counter()
    for i, (query, text) in enumerate(zip(queries, texts), start=1):
        t0 = time.perf_counter()
        route = index.route(query["query_text"], mode)
        result = index.search_vector(vectors[text.strip()], route, top_k)
        rows.append(
            {
                "query_id": query["query_id"],
                "stage": query["stage"],
                "query_text": query["query_text"],
                "semantic_text": text,
                "country": query.get("country"),
                "city": query.get("city"),
                "scope": route.scope,
                "routed_country": route.country,
                "routed_city": route.city,
                "num_candidates": result.num_candidates,
                "retrieved_ids": result.image_ids,
                "scores": result.scores,
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
            }
        )
        if progress and (i % 5000 == 0 or i == len(queries)):
            print(f"[{mode}] {i}/{len(queries)} queries ({time.perf_counter() - started:.1f}s)", flush=True)
    return rows


def per_country(rows: Sequence[dict], min_queries: int = 5) -> dict:
    """Macro nDCG@10 over countries with at least ``min_queries`` scored queries."""
    groups: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        if row.get("has_positive") and row.get("country"):
            groups[row["country"]].append(row["metrics"]["nDCG@10"])
    table = sorted(
        ({"country": c, "queries": len(v), "nDCG@10": sum(v) / len(v)} for c, v in groups.items()),
        key=lambda item: (item["nDCG@10"], -item["queries"], item["country"]),
    )
    supported = [t for t in table if t["queries"] >= min_queries]
    return {
        "countries_with_positives": len(table),
        f"countries_with_at_least_{min_queries}": len(supported),
        "macro_nDCG@10": sum(t["nDCG@10"] for t in supported) / len(supported) if supported else None,
        "worst_10": supported[:10],
    }


def format_summary(summary: Mapping[str, dict], title: Optional[str] = None) -> str:
    keys = ("nDCG@10", "mAP@10", "Recall@10", "Success@10", "P@10", "MRR@10")
    lines = [f"## {title}", ""] if title else []
    lines.append("| Split | Queries | " + " | ".join(keys) + " |")
    lines.append("|---|---:|" + "---:|" * len(keys))
    for split in ("overall", *STAGES):
        item = summary[split]
        name = "Overall" if split == "overall" else STAGE_NAMES[split]
        cells = " | ".join(f"{item.get(k, 0.0):.4f}" for k in keys)
        lines.append(f"| {name} | {item['queries']:,} | {cells} |")
    return "\n".join(lines)
