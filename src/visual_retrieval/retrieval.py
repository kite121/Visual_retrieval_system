"""Shared fixed-gallery nearest-neighbor search (cosine similarity, exact top-K)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn

from .data import load_image
from .embeddings import build_transform, encode_batch


@dataclass(frozen=True)
class Hit:
    rank: int  # 1-based
    image_id: str
    breed: str
    score: float  # cosine similarity


class GalleryIndex:
    """Embeddings of the fixed gallery, kept on one device for fast exact search.

    All embeddings must be L2-normalized, so cosine similarity is a plain dot product.
    """

    def __init__(
        self,
        embeddings: np.ndarray | torch.Tensor,
        image_ids: Sequence[str],
        labels: np.ndarray,
        breeds: Optional[Sequence[str]] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        emb = torch.as_tensor(embeddings, dtype=torch.float32)
        if emb.ndim != 2 or len(emb) != len(image_ids) or len(emb) != len(labels):
            raise ValueError("embeddings, image_ids and labels must describe the same N items")
        norms = emb.norm(dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
            raise ValueError("gallery embeddings must be L2-normalized")
        self.device = torch.device(device)
        self.embeddings = emb.to(self.device)
        self.image_ids = list(image_ids)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.breeds = list(breeds) if breeds is not None else [str(l) for l in self.labels]
        self._pos = {i: n for n, i in enumerate(self.image_ids)}

    def __len__(self) -> int:
        return len(self.image_ids)

    @torch.inference_mode()
    def search(
        self,
        query: torch.Tensor | np.ndarray,
        k: int = 5,
        exclude_ids: Optional[Sequence[str]] = None,
        chunk_size: int = 4096,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-``k`` gallery items per query. Returns ``(scores, indices)`` of shape ``(Q, k)``.

        ``query`` holds L2-normalized embeddings ``(Q, D)`` or ``(D,)``. ``exclude_ids``
        removes gallery images from the ranking (e.g. when a gallery image is used as the
        query in the demo). Evaluation queries are never in the gallery, so it is unused there.
        """
        q = torch.as_tensor(query, dtype=torch.float32, device=self.device)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if k > len(self):
            raise ValueError(f"k={k} exceeds gallery size {len(self)}")
        mask = None
        if exclude_ids:
            mask = torch.zeros(len(self), dtype=torch.bool, device=self.device)
            mask[[self._pos[i] for i in exclude_ids if i in self._pos]] = True
            if len(self) - int(mask.sum()) < k:
                raise ValueError("fewer than k gallery images remain after exclusion")
        scores, idx = [], []
        for start in range(0, len(q), chunk_size):
            sim = q[start : start + chunk_size] @ self.embeddings.T
            if mask is not None:
                sim = sim.masked_fill(mask, float("-inf"))
            s, i = sim.topk(k, dim=1)  # sorted, descending
            scores.append(s)
            idx.append(i)
        return torch.cat(scores), torch.cat(idx)

    def hits(self, scores: torch.Tensor, indices: torch.Tensor) -> list[Hit]:
        """Convert one query's ``(k,)`` scores/indices into ``Hit`` records."""
        return [
            Hit(rank=r + 1, image_id=self.image_ids[i], breed=self.breeds[i], score=float(s))
            for r, (s, i) in enumerate(zip(scores.tolist(), indices.tolist()))
        ]


def query(
    image: Image.Image | str | Path,
    model: nn.Module,
    index: GalleryIndex,
    k: int = 5,
    *,
    transform=None,
    exclude_ids: Optional[Sequence[str]] = None,
) -> list[Hit]:
    """Shared query function: the ``k`` gallery images most similar to ``image``.

    ``model`` is a loaded encoder (``models.load_encoder``); ``image`` is a PIL image or a
    path. The image goes through the same preprocessing as the gallery, the embedding is
    L2-normalized, and gallery images are ranked by cosine similarity.
    """
    if not isinstance(image, Image.Image):
        image = load_image(image)
    elif image.mode != "RGB":
        image = image.convert("RGB")
    transform = transform or build_transform()
    device = index.device
    emb = encode_batch(model, transform(image).unsqueeze(0), device)
    scores, idx = index.search(emb, k, exclude_ids=exclude_ids)
    return index.hits(scores[0].cpu(), idx[0].cpu())
