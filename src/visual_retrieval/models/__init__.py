"""Adapters for the pretrained image encoders and the shared model registry.

Every encoder is an ``nn.Module`` that maps a normalized image batch ``(B, 3, 224, 224)``
to raw feature vectors ``(B, D)`` -- the CLS embedding, **before** any classification
head. L2 normalization is applied later in ``embeddings.py`` so it is identical for all
models.

Adding a model (for example DINOv2 ViT-B/14 or DeiT-B/16) takes one ``register`` call in
the corresponding adapter module::

    register(ModelSpec(
        name="dinov2_vit_b14",
        display_name="DINOv2 ViT-B/14",
        embed_dim=768,
        loader=_load,          # callable(pretrained: bool) -> nn.Module
        source="torch.hub: facebookresearch/dinov2",
    ))

The ``name`` must match the entry in ``configs/default.yaml``; scripts pick up registered
models automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    name: str  # key used in configs, cache paths and result tables
    display_name: str
    embed_dim: int
    loader: Callable[[bool], nn.Module]  # loader(pretrained) -> module returning (B, D)
    source: str  # where the weights come from (for the report)


_REGISTRY: Dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    if spec.name in _REGISTRY:
        raise ValueError(f"model '{spec.name}' is already registered")
    _REGISTRY[spec.name] = spec
    return spec


def available_models() -> List[str]:
    """Names of all registered models, in registration order."""
    return list(_REGISTRY)


def get_spec(name: str) -> ModelSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"model '{name}' is not registered; available: {available_models()}") from None


def load_encoder(name: str, device: str | torch.device = "cpu", pretrained: bool = True) -> nn.Module:
    """Load a frozen encoder in eval mode on ``device``.

    ``pretrained=False`` builds the same architecture with random weights; it exists
    only for fast tests that must not download anything.
    """
    spec = get_spec(name)
    model = spec.loader(pretrained)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


# Importing the adapters registers them. Keep these at the bottom of the file.
from . import deit, dino, dinov2  # noqa: E402,F401
