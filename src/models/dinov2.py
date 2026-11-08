"""DINOv2 ViT-B/14, original backbone without register tokens (Karim).

Use Meta's LVD-142M checkpoint and normalized CLS token, not a classifier.
The official model interpolates positional embeddings for the project's 224px
inputs. Its Hub code revision is pinned; we do not override its 518px training
grid when loading weights. The offline random-weight constructor uses timm and
exists only for interface tests, never for reported retrieval results.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import shutil
import tempfile
from urllib.error import URLError
import warnings
import zipfile

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
    try:
        return torch.hub.load(
            HUB_REPOSITORY, HUB_MODEL, pretrained=True, trust_repo=True,
        )
    except URLError as exc:
        # GitHub's zipball endpoint can return 504; codeload serves the same commit.
        warnings.warn(f"DINOv2 Hub download failed ({exc}); retrying the pinned official source via codeload")
        cache = Path(torch.hub.get_dir())
        target = cache / f"facebookresearch_dinov2_{HUB_REVISION}"
        if not target.is_dir():
            cache.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=cache) as directory:
                source_zip = Path(directory) / "source.zip"
                torch.hub.download_url_to_file(
                    f"https://codeload.github.com/facebookresearch/dinov2/zip/{HUB_REVISION}",
                    str(source_zip),
                )
                prefix = f"dinov2-{HUB_REVISION}"
                with zipfile.ZipFile(source_zip) as archive:
                    for name in archive.namelist():
                        path = PurePosixPath(name)
                        if path.is_absolute() or ".." in path.parts or path.parts[0] != prefix:
                            raise ValueError("Unexpected path in the official DINOv2 source archive")
                    archive.extractall(directory)
                shutil.move(str(Path(directory) / prefix), target)
        return torch.hub.load(str(target), HUB_MODEL, source="local", pretrained=True)


register(ModelSpec(
    name="dinov2_vit_b14",
    display_name="DINOv2 ViT-B/14",
    embed_dim=EMBED_DIM,
    loader=_load,
    source=f"torch.hub {HUB_REPOSITORY} ({HUB_MODEL}, LVD-142M, no registers)",
))
