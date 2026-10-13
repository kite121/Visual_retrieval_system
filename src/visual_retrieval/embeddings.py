"""Shared image preprocessing, embedding extraction, and normalization.

Preprocessing is deterministic and identical for gallery and query images and for every
model: RGB, resize shorter side to 256, center crop 224x224, tensor, ImageNet mean/std.
No random augmentation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE_SHORT_SIDE = 256
CROP_SIZE = 224


def build_transform() -> T.Compose:
    """The single preprocessing pipeline used everywhere (bicubic, as in DINO/DeiT eval)."""
    return T.Compose(
        [
            T.Resize(RESIZE_SHORT_SIDE, interpolation=InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(CROP_SIZE),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


@torch.inference_mode()
def encode_batch(encoder: nn.Module, batch: torch.Tensor, device: str | torch.device) -> torch.Tensor:
    """Embed a preprocessed batch; returns L2-normalized float32 features on ``device``."""
    feats = encoder(batch.to(device, non_blocking=True))
    if feats.ndim != 2:
        raise ValueError(f"encoder must return (B, D) features, got {tuple(feats.shape)}")
    return l2_normalize(feats.float())


@torch.inference_mode()
def extract_embeddings(
    encoder: nn.Module,
    dataset: Dataset,
    device: str | torch.device,
    batch_size: int = 64,
    num_workers: int = 2,
    progress: bool = True,
) -> np.ndarray:
    """L2-normalized embeddings ``(N, D)`` float32 for every item, in dataset order."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=False,
    )
    it = loader
    if progress:
        try:
            from tqdm.auto import tqdm

            it = tqdm(loader, desc="embedding", leave=False)
        except ImportError:
            pass
    chunks = [encode_batch(encoder, images, device).cpu() for images, _ in it]
    return torch.cat(chunks).numpy().astype(np.float32, copy=False)


# --------------------------------------------------------------------------- caching


@dataclass
class CachedEmbeddings:
    embeddings: np.ndarray  # (N, D) float32, L2-normalized
    image_ids: list[str]
    labels: np.ndarray  # (N,) int64
    meta: dict


def cache_dir(artifacts: str | Path, model_name: str) -> Path:
    return Path(artifacts) / "embeddings" / model_name


def save_embeddings(
    artifacts: str | Path,
    model_name: str,
    split: str,
    embeddings: np.ndarray,
    image_ids: Sequence[str],
    labels: np.ndarray,
    meta: Optional[dict] = None,
) -> Path:
    """Write ``<artifacts>/embeddings/<model>/<split>.npz`` (+ ``<split>.json`` metadata)."""
    d = cache_dir(artifacts, model_name)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{split}.npz"
    np.savez(path, embeddings=embeddings, image_ids=np.asarray(image_ids), labels=np.asarray(labels))
    (d / f"{split}.json").write_text(json.dumps(meta or {}, indent=2))
    return path


def load_embeddings(
    artifacts: str | Path,
    model_name: str,
    split: str,
    expected_ids: Optional[Sequence[str]] = None,
) -> Optional[CachedEmbeddings]:
    """Load cached embeddings, or ``None`` if missing or built for a different id list."""
    d = cache_dir(artifacts, model_name)
    path = d / f"{split}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        emb, ids, labels = z["embeddings"], z["image_ids"].tolist(), z["labels"]
    if expected_ids is not None and list(expected_ids) != ids:
        return None  # the split changed since the cache was written
    meta_path = d / f"{split}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return CachedEmbeddings(emb.astype(np.float32, copy=False), ids, labels.astype(np.int64), meta)
