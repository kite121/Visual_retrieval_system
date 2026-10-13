import numpy as np
import pytest

from visual_retrieval.metrics import per_query_hit, retrieval_metrics


def test_hand_computed_values():
    gallery = np.array([0] * 8 + [1] * 4)  # breed 0 has 8 relevant, breed 1 has 4
    query = np.array([0, 1])
    # query 0: 3 relevant in top5, 5 in top10 ; query 1: none in top5, 1 in top10
    retrieved = np.array([
        [0, 0, 1, 0, 1, 1, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ])
    m = retrieval_metrics(retrieved, query, gallery)
    assert m["P@5"] == pytest.approx((3 / 5 + 0 / 5) / 2)
    assert m["P@10"] == pytest.approx((5 / 10 + 1 / 10) / 2)
    assert m["R@5"] == pytest.approx((3 / 8 + 0 / 4) / 2)
    assert m["R@10"] == pytest.approx((5 / 8 + 1 / 4) / 2)
    assert m["Hit@5"] == pytest.approx(0.5)


def test_perfect_and_empty():
    gallery = np.repeat(np.arange(3), 10)
    query = np.array([0, 1, 2])
    perfect = np.repeat(query[:, None], 10, axis=1)
    m = retrieval_metrics(perfect, query, gallery)
    assert m["P@5"] == m["P@10"] == m["Hit@5"] == 1.0
    assert m["R@10"] == 1.0 and m["R@5"] == 0.5
    wrong = np.repeat(((query + 1) % 3)[:, None], 10, axis=1)
    m = retrieval_metrics(wrong, query, gallery)
    assert all(v == 0 for v in m.values())


def test_too_few_results_raises():
    with pytest.raises(ValueError):
        retrieval_metrics(np.zeros((2, 5), int), np.zeros(2, int), np.zeros(4, int), ks=(5, 10))


def test_per_query_hit():
    assert per_query_hit(np.array([[1, 1, 0], [1, 1, 1]]), np.array([0, 0]), k=3).tolist() == [True, False]
