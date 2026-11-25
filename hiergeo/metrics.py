"""Ranking metrics used by HierGeo-Bench.

All metrics take a ranked list of image ids and the set of relevant ids, and are
computed at a cut-off ``k`` (10 in the paper). Queries without any relevant image
are excluded from the averages by the evaluator, not here.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Set

METRIC_NAMES = ("nDCG", "mAP", "Recall", "Success", "P", "MRR")


def relevance_flags(retrieved: Sequence[int], relevant: Set[int]) -> List[int]:
    """Binary relevance of each retrieved id."""
    return [1 if image_id in relevant else 0 for image_id in retrieved]


def _dcg(flags: Sequence[int], k: int) -> float:
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(flags[:k], start=1) if rel)


def ndcg_at_k(flags: Sequence[int], num_relevant: int, k: int) -> float:
    if k <= 0 or num_relevant <= 0:
        return 0.0
    ideal = _dcg([1] * min(num_relevant, k), k)
    return _dcg(flags, k) / ideal if ideal else 0.0


def average_precision_at_k(flags: Sequence[int], num_relevant: int, k: int) -> float:
    """AP@k normalised by ``min(num_relevant, k)`` so that it lies in [0, 1]."""
    if k <= 0 or num_relevant <= 0:
        return 0.0
    hits, total = 0, 0.0
    for rank, rel in enumerate(flags[:k], start=1):
        if rel:
            hits += 1
            total += hits / rank
    return total / min(num_relevant, k)


def recall_at_k(flags: Sequence[int], num_relevant: int, k: int) -> float:
    if k <= 0 or num_relevant <= 0:
        return 0.0
    return sum(flags[:k]) / num_relevant


def precision_at_k(flags: Sequence[int], k: int) -> float:
    """Precision over the returned list (shorter lists are not padded)."""
    head = list(flags[:k])
    return sum(head) / len(head) if head and k > 0 else 0.0


def success_at_k(flags: Sequence[int], k: int) -> float:
    return 1.0 if k > 0 and any(flags[:k]) else 0.0


def reciprocal_rank_at_k(flags: Sequence[int], k: int) -> float:
    for rank, rel in enumerate(flags[:k], start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def query_metrics(retrieved: Sequence[int], relevant: Iterable[int], k: int = 10) -> Dict[str, float]:
    """All HierGeo-Bench metrics for one query."""
    relevant = set(relevant)
    flags = relevance_flags(retrieved, relevant)
    n = len(relevant)
    return {
        f"nDCG@{k}": ndcg_at_k(flags, n, k),
        f"mAP@{k}": average_precision_at_k(flags, n, k),
        f"Recall@{k}": recall_at_k(flags, n, k),
        f"Success@{k}": success_at_k(flags, k),
        f"P@{k}": precision_at_k(flags, k),
        f"MRR@{k}": reciprocal_rank_at_k(flags, k),
    }


def mean_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
