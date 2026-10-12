"""Supervised DeiT-S/16 and DeiT-B/16 adapters (Azamat and Karim).

Azamat: DeiT-S/16 (``deit_small_patch16_224``, ImageNet-1k supervised, 384-d).
Karim:  add DeiT-B/16 by calling ``_register_deit`` once more at the bottom of this file
        (``timm_name="deit_base_patch16_224"``, ``embed_dim=768``, name
        ``deit_vit_b16_supervised``).

``num_classes=0`` removes the classification head in timm, so ``forward`` returns the
pooled CLS feature **before** the head (never class logits). We use the non-distilled
models so there is a single CLS token and no distillation token.
"""

from __future__ import annotations

import timm
from torch import nn

from . import ModelSpec, register


def _register_deit(name: str, display_name: str, timm_name: str, embed_dim: int) -> None:
    def _load(pretrained: bool = True) -> nn.Module:
        return timm.create_model(timm_name, pretrained=pretrained, num_classes=0)

    register(
        ModelSpec(
            name=name,
            display_name=display_name,
            embed_dim=embed_dim,
            loader=_load,
            source=f"timm {timm_name} (supervised ImageNet-1k, head removed)",
        )
    )


_register_deit(
    name="deit_vit_s16_supervised",
    display_name="DeiT-S/16 (supervised)",
    timm_name="deit_small_patch16_224",
    embed_dim=384,
)
