"""Precision@K, Recall@K, and Hit@5 for breed-based relevance.

A gallery image is relevant to a query when both have the same breed label. Scores are
averaged over queries. Labels are used only here, never for retrieval.

* ``P@K``   = relevant among the first K results / K
* ``R@K``   = relevant among the first K results / all relevant images in the gallery
* ``Hit@K`` = 1 if at least one of the first K results is relevant, else 0
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def retrieval_metrics(
    retrieved_labels: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    ks: Sequence[int] = (5, 10),
    hit_ks: Sequence[int] = (5,),
) -> dict[str, float]:
    """Mean metrics over queries.

    Args:
        retrieved_labels: ``(Q, Kmax)`` breed labels of the ranked results (best first).
        query_labels: ``(Q,)`` breed label of each query.
        gallery_labels: ``(G,)`` breed labels of the whole gallery (recall denominators).
    """
    retrieved_labels = np.asarray(retrieved_labels)
    query_labels = np.asarray(query_labels)
    gallery_labels = np.asarray(gallery_labels)
    if retrieved_labels.ndim != 2 or len(retrieved_labels) != len(query_labels):
        raise ValueError("retrieved_labels must be (Q, Kmax) and match query_labels")
    kmax = retrieved_labels.shape[1]
    if max(*ks, *hit_ks) > kmax:
        raise ValueError(f"need at least {max(*ks, *hit_ks)} retrieved results per query, got {kmax}")

    relevant = retrieved_labels == query_labels[:, None]  # (Q, Kmax) bool
    n_relevant = np.bincount(gallery_labels, minlength=int(max(gallery_labels.max(), query_labels.max())) + 1)[query_labels]
    has_relevant = n_relevant > 0  # a breed absent from the gallery has undefined recall

    out: dict[str, float] = {}
    for k in ks:
        found = relevant[:, :k].sum(axis=1)
        out[f"P@{k}"] = float((found / k).mean())
        out[f"R@{k}"] = float((found[has_relevant] / n_relevant[has_relevant]).mean())
    for k in hit_ks:
        out[f"Hit@{k}"] = float(relevant[:, :k].any(axis=1).mean())
    return out


def per_query_hit(retrieved_labels: np.ndarray, query_labels: np.ndarray, k: int = 5) -> np.ndarray:
    """Boolean ``(Q,)``: does the top-``k`` contain a same-breed image? ``False`` = failed query."""
    return (np.asarray(retrieved_labels)[:, :k] == np.asarray(query_labels)[:, None]).any(axis=1)
