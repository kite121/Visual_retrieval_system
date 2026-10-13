"""Gallery-building, query-embedding, and search timing utilities.

Timing rules (identical for every model):

* same device, same fp32 precision, same query set (fixed seed);
* an untimed warm-up before every measurement (CUDA init, cuDNN autotune, allocator);
* ``torch.cuda.synchronize()`` around every timed region when the device is a GPU;
* embedding extraction and top-K search are timed **separately**, per single query
  (batch size 1), and reported as medians;
* image decoding + preprocessing is reported as a third, separate number and is *not*
  part of the embedding time;
* the gallery build time covers loading, preprocessing, the forward pass and storing all
  gallery embeddings (``batch_size`` and ``num_workers`` are recorded with it).
"""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .embeddings import encode_batch, extract_embeddings
from .retrieval import GalleryIndex


def sync(device: str | torch.device) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def hardware_info(device: str | torch.device) -> dict:
    info = {
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu_count": __import__("os").cpu_count(),
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(device)
        info["cuda"] = torch.version.cuda
    return info


def warm_file_cache(paths: Sequence[Path]) -> None:
    """Read every file once (untimed) so disk-cache state does not favour the model run last."""
    for p in paths:
        with open(p, "rb") as f:
            f.read()


@torch.inference_mode()
def warmup(encoder: nn.Module, device: str | torch.device, batch_size: int = 1, n: int = 5, size: int = 224) -> None:
    x = torch.zeros(batch_size, 3, size, size)
    for _ in range(n):
        encode_batch(encoder, x, device)
    sync(device)


def time_gallery_build(
    encoder: nn.Module,
    dataset: Dataset,
    device: str | torch.device,
    batch_size: int,
    num_workers: int,
    progress: bool = True,
) -> tuple[np.ndarray, float]:
    """Extract all gallery embeddings; returns ``(embeddings, seconds)``."""
    warmup(encoder, device, batch_size=batch_size)
    sync(device)
    t0 = time.perf_counter()
    emb = extract_embeddings(encoder, dataset, device, batch_size=batch_size, num_workers=num_workers, progress=progress)
    sync(device)
    return emb, time.perf_counter() - t0


def _summary(values_ms: list[float]) -> dict[str, float]:
    a = np.asarray(values_ms)
    return {
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p95": float(np.percentile(a, 95)),
    }


@torch.inference_mode()
def measure_query_latency(
    encoder: nn.Module,
    queries: Dataset,
    index: GalleryIndex,
    transform,
    k: int = 5,
    n_queries: int = 200,
    n_warmup: int = 10,
    seed: int = 42,
) -> dict:
    """Per-query latency in ms for single-image queries.

    ``queries`` is a ``PetDataset`` *without* a transform (raw RGB images are returned by
    ``load_image``); the same ``n_queries`` images (fixed seed) are used for every model.
    """
    from .data import load_image

    device = index.device
    n = min(n_queries, len(queries))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(queries), size=n, replace=False)

    warmup(encoder, device, batch_size=1, n=n_warmup)
    # warm the search path too
    dummy = torch.nn.functional.normalize(torch.randn(1, index.embeddings.shape[1]), dim=1)
    for _ in range(n_warmup):
        index.search(dummy, k)[1].cpu()
    sync(device)

    pre_ms, emb_ms, search_ms = [], [], []
    for i in chosen:
        t0 = time.perf_counter()
        x = transform(load_image(queries.paths[i])).unsqueeze(0)
        pre_ms.append((time.perf_counter() - t0) * 1e3)

        sync(device)
        t0 = time.perf_counter()
        e = encode_batch(encoder, x, device)  # includes host->device copy + L2 norm
        sync(device)
        emb_ms.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        _, idx = index.search(e, k)
        idx.cpu()  # results must be on the host to be usable; also acts as a sync
        sync(device)
        search_ms.append((time.perf_counter() - t0) * 1e3)

    return {
        "n_queries": int(n),
        "k": k,
        "embed_ms": _summary(emb_ms),
        "search_ms": _summary(search_ms),
        "preprocess_ms": _summary(pre_ms),
    }
