"""Paired bootstrap confidence intervals for per-query metric differences."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np


def paired_bootstrap(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    samples: int = 5000,
    seed: int = 20260726,
) -> Optional[Dict[str, object]]:
    """Mean difference ``candidate - baseline`` over shared query ids, with a 95% CI.

    Queries are resampled with replacement (``samples`` times); the interval is the
    2.5th-97.5th percentile of the resampled means. The defaults are the ones used
    for Tab. 11 of the paper.
    """
    common = sorted(set(candidate) & set(baseline))
    if not common:
        return None
    a = np.asarray([candidate[q] for q in common], dtype=np.float64)
    b = np.asarray([baseline[q] for q in common], dtype=np.float64)
    deltas = a - b
    rng = np.random.default_rng(seed)
    means = rng.choice(deltas, size=(samples, len(deltas)), replace=True).mean(axis=1)
    return {
        "queries": int(len(deltas)),
        "candidate_mean": float(a.mean()),
        "baseline_mean": float(b.mean()),
        "delta": float(deltas.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    }
