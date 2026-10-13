"""Oxford-IIIT Pet loading and seed-42 breed-stratified split (Azamat).

Protocol (see ``docs/project_plan.pdf``):

* Labels come from ``torchvision.datasets.OxfordIIITPet(target_types="category")``.
* The official ``trainval`` partition is sorted by image id and split 50/50 within
  each breed with ``train_test_split(test_size=0.5, random_state=42, stratify=labels)``.
  The first half is the retrieval **gallery**, the second half the **val** queries.
* The official ``test`` partition is never touched by the split and is used only for
  final queries.
* The result is a manifest with columns ``image_id, breed, split`` (plus the integer
  ``label``) that every team member loads instead of re-splitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

SPLIT_GALLERY = "gallery"  # first half of official trainval ("train" in the plan)
SPLIT_VAL = "val"  # second half of official trainval (validation queries)
SPLIT_TEST = "test"  # official test (final queries)
SPLITS = (SPLIT_GALLERY, SPLIT_VAL, SPLIT_TEST)
QUERY_SPLITS = (SPLIT_VAL, SPLIT_TEST)

MANIFEST_COLUMNS = ["image_id", "breed", "label", "split"]
DATASET_FOLDER = "oxford-iiit-pet"  # folder name used by torchvision under ``root``

# Official sizes, used only for a sanity message in ``prepare_data.py``.
EXPECTED_SIZES = {"trainval": 3680, "test": 3669, "breeds": 37}


def images_dir(root: str | Path) -> Path:
    """Directory with the raw ``<image_id>.jpg`` files downloaded by torchvision."""
    return Path(root) / DATASET_FOLDER / "images"


def _official_frame(root: str | Path, official_split: str, download: bool) -> pd.DataFrame:
    """Image ids and breed labels of one official partition (``trainval`` / ``test``)."""
    from torchvision.datasets import OxfordIIITPet

    ds = OxfordIIITPet(
        root=str(root),
        split=official_split,
        target_types="category",
        download=download,
    )
    # ``_images`` is the list of image paths in the official list order; there is no
    # public accessor for it, but it is stable across torchvision releases.
    ids = [Path(p).stem for p in ds._images]
    labels = np.asarray(ds._labels, dtype=np.int64)
    classes = list(ds.classes)
    return pd.DataFrame(
        {
            "image_id": ids,
            "breed": [classes[i] for i in labels],
            "label": labels,
        }
    )


def split_trainval(
    trainval: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.5,
) -> pd.DataFrame:
    """Split ``trainval`` 50/50 within each breed. Returns ``trainval`` with a ``split`` column.

    Images are sorted by id first, so the result does not depend on the order in which
    torchvision happens to list the files.
    """
    tv = trainval.sort_values("image_id", kind="stable").reset_index(drop=True)
    idx = np.arange(len(tv))
    gallery_idx, val_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=tv["label"].to_numpy(),
    )
    split = np.empty(len(tv), dtype=object)
    split[gallery_idx] = SPLIT_GALLERY
    split[val_idx] = SPLIT_VAL
    tv["split"] = split
    return tv


def build_manifest(
    root: str | Path,
    seed: int = 42,
    test_size: float = 0.5,
    download: bool = True,
) -> pd.DataFrame:
    """Build the full manifest (gallery + val + test) from the torchvision dataset."""
    trainval = _official_frame(root, "trainval", download)
    test = _official_frame(root, "test", download=False)
    test = test.sort_values("image_id", kind="stable").reset_index(drop=True)
    test["split"] = SPLIT_TEST
    manifest = pd.concat([split_trainval(trainval, seed, test_size), test], ignore_index=True)
    manifest = manifest.sort_values("image_id", kind="stable").reset_index(drop=True)
    validate_manifest(manifest)
    return manifest[MANIFEST_COLUMNS]


def validate_manifest(manifest: pd.DataFrame) -> None:
    """Raise ``ValueError`` if the manifest violates the agreed protocol."""
    missing = [c for c in MANIFEST_COLUMNS if c not in manifest.columns]
    if missing:
        raise ValueError(f"manifest is missing columns: {missing}")
    if manifest["image_id"].duplicated().any():
        raise ValueError("an image_id appears more than once (query/gallery overlap?)")
    unknown = set(manifest["split"]) - set(SPLITS)
    if unknown:
        raise ValueError(f"unknown split names: {sorted(unknown)}")
    # every breed must be present in the gallery and in both query sets
    breeds = set(manifest["breed"])
    for s in SPLITS:
        if set(manifest.loc[manifest["split"] == s, "breed"]) != breeds:
            raise ValueError(f"split '{s}' does not contain all {len(breeds)} breeds")


def save_manifest(manifest: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest[MANIFEST_COLUMNS].to_csv(path, index=False)
    return path


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load the shared manifest (the single source of truth for the split)."""
    manifest = pd.read_csv(path, dtype={"image_id": str, "breed": str, "split": str})
    manifest["label"] = manifest["label"].astype(np.int64)
    validate_manifest(manifest)
    return manifest


def subset(manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    """Rows of one split, ordered by image id (this order defines embedding row order)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got '{split}'")
    return manifest[manifest["split"] == split].sort_values("image_id", kind="stable").reset_index(drop=True)


def image_path(root: str | Path, image_id: str) -> Path:
    return images_dir(root) / f"{image_id}.jpg"


def load_image(path: str | Path) -> Image.Image:
    """Open an image and convert it to RGB (a few Oxford-Pet files are grayscale/CMYK)."""
    with Image.open(path) as im:
        return im.convert("RGB")


class PetDataset(Dataset):
    """One manifest split as a torch ``Dataset`` yielding ``(tensor, label)``."""

    def __init__(
        self,
        frame: pd.DataFrame,
        root: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.image_ids: list[str] = self.frame["image_id"].tolist()
        self.labels = self.frame["label"].to_numpy(dtype=np.int64)
        self.paths: list[Path] = [image_path(self.root, i) for i in self.image_ids]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, i: int):
        img = load_image(self.paths[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])
