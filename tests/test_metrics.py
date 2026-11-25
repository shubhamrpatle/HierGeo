import math

import numpy as np
import pytest

from hiergeo.metrics import query_metrics
from hiergeo.stats import paired_bootstrap


def test_perfect_ranking():
    m = query_metrics([1, 2, 3], {1, 2, 3}, k=10)
    assert m["nDCG@10"] == pytest.approx(1.0)
    assert m["mAP@10"] == pytest.approx(1.0)
    assert m["Recall@10"] == pytest.approx(1.0)
    assert m["P@10"] == pytest.approx(1.0)  # precision over the returned list
    assert m["MRR@10"] == 1.0 and m["Success@10"] == 1.0


def test_hand_computed_values():
    # relevant {5, 9}; hits at ranks 2 and 4
    m = query_metrics([1, 5, 2, 9, 3], {5, 9}, k=10)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3)
    assert m["nDCG@10"] == pytest.approx(dcg / idcg)
    assert m["mAP@10"] == pytest.approx((1 / 2 + 2 / 4) / 2)
    assert m["Recall@10"] == pytest.approx(1.0)
    assert m["P@10"] == pytest.approx(2 / 5)
    assert m["MRR@10"] == pytest.approx(0.5)


def test_ap_is_normalised_by_min_relevant_k():
    m = query_metrics(list(range(10)), set(range(100)), k=10)
    assert m["mAP@10"] == pytest.approx(1.0)
    assert m["Recall@10"] == pytest.approx(0.1)


def test_no_hit():
    m = query_metrics([1, 2], {7}, k=10)
    assert all(v == 0.0 for v in m.values())


def test_paired_bootstrap():
    rng = np.random.default_rng(0)
    base = {str(i): float(x) for i, x in enumerate(rng.random(2000))}
    cand = {q: v + 0.05 for q, v in base.items()}
    out = paired_bootstrap(cand, base, samples=500)
    assert out["delta"] == pytest.approx(0.05)
    assert out["ci95"][0] == pytest.approx(0.05) and out["ci95"][1] == pytest.approx(0.05)
    noisy = {q: v + rng.normal(0, 0.1) for q, v in base.items()}
    out = paired_bootstrap(noisy, base, samples=500)
    assert out["ci95"][0] < 0.0 < out["ci95"][1]
