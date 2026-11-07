"""DINOv2 ViT-B/14, original backbone without register tokens (Karim).

Use Meta's LVD-142M checkpoint and normalized CLS token, not a classifier.
The official model interpolates positional embeddings for the project's 224px
inputs. Its Hub code revision is pinned; we do not override its 518px training
grid when loading weights. The offline random-weight constructor uses timm and
exists only for interface tests, never for reported retrieval results.
"""

from __future__ import annotations

import timm
import torch
from torch import nn

from . import ModelSpec, register

HUB_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
HUB_REPOSITORY = f"facebookresearch/dinov2:{HUB_REVISION}"
HUB_MODEL = "dinov2_vitb14"
EMBED_DIM = 768


def _load(pretrained: bool = True) -> nn.Module:
    if not pretrained:
        return timm.create_model(
            "vit_base_patch14_dinov2", pretrained=False, num_classes=0, img_size=224,
        )
    return torch.hub.load(
        HUB_REPOSITORY, HUB_MODEL, pretrained=True, trust_repo=True,
    )


register(ModelSpec(
    name="dinov2_vit_b14",
    display_name="DINOv2 ViT-B/14",
    embed_dim=EMBED_DIM,
    loader=_load,
    source=f"torch.hub {HUB_REPOSITORY} ({HUB_MODEL}, LVD-142M, no registers)",
))
