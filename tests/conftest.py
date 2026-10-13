import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visual_retrieval import data  # noqa: E402

N_BREEDS, N_TRAINVAL, N_TEST = 5, 12, 6


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory):
    """Tiny fake Oxford-Pet layout: ``<root>/oxford-iiit-pet/images/*.jpg`` + manifest + yaml config."""
    root = tmp_path_factory.mktemp("pets")
    img_dir = data.images_dir(root)
    img_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for b in range(N_BREEDS):
        base = rng.integers(40, 215, size=3)
        for j in range(N_TRAINVAL + N_TEST):
            noise = rng.integers(-25, 25, size=(60, 80, 3))
            arr = np.clip(base + noise, 0, 255).astype(np.uint8)
            image_id = f"Breed{b}_{j + 1}"
            Image.fromarray(arr).save(img_dir / f"{image_id}.jpg")
            rows.append((image_id, f"Breed{b}", b, "test" if j >= N_TRAINVAL else "trainval"))
    df = pd.DataFrame(rows, columns=["image_id", "breed", "label", "official"])
    tv = data.split_trainval(df[df.official == "trainval"][["image_id", "breed", "label"]])
    te = df[df.official == "test"][["image_id", "breed", "label"]].assign(split=data.SPLIT_TEST)
    manifest = pd.concat([tv, te]).sort_values("image_id").reset_index(drop=True)
    manifest_path = data.save_manifest(manifest, root / "manifest.csv")
    return {"root": root, "manifest": manifest, "manifest_path": manifest_path}
