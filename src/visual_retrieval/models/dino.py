"""DINO ViT-S/16 adapter (Azamat).

Self-supervised DINO ViT-S/16 (Caron et al., 2021). The official checkpoint is loaded
through ``torch.hub`` from ``facebookresearch/dino``; its forward pass returns the final
CLS token (384-d) after the last LayerNorm, which is the embedding we use. If the hub
download fails (for example GitHub rate limiting) we fall back to the identical DINO
weights published in ``timm`` (``vit_small_patch16_224.dino``).
"""

from __future__ import annotations

import warnings

import timm
import torch
from torch import nn

from . import ModelSpec, register

EMBED_DIM = 384


def _load(pretrained: bool = True) -> nn.Module:
    if not pretrained:
        return timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
    try:
        return torch.hub.load("facebookresearch/dino:main", "dino_vits16", trust_repo=True)
    except Exception as exc:  # network / hub problems -> same weights via timm
        warnings.warn(f"torch.hub load of DINO ViT-S/16 failed ({exc!r}); using timm 'vit_small_patch16_224.dino'")
        return timm.create_model("vit_small_patch16_224.dino", pretrained=True, num_classes=0)


register(
    ModelSpec(
        name="dino_vit_s16",
        display_name="DINO ViT-S/16",
        embed_dim=EMBED_DIM,
        loader=_load,
        source="torch.hub facebookresearch/dino (dino_vits16); fallback timm vit_small_patch16_224.dino",
    )
)
