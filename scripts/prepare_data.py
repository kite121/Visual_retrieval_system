"""Download Oxford-IIIT Pet and write the shared split manifest.

    python scripts/prepare_data.py                 # create configs/split_manifest.csv
    python scripts/prepare_data.py --check         # verify the committed manifest is reproducible

The manifest (``image_id, breed, label, split``) is committed to Git so the whole team
uses exactly the same gallery / validation / test partition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
import yaml

from visual_retrieval import data

REPO_ROOT = _bootstrap.REPO_ROOT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--root", default=str(REPO_ROOT / "data"), help="where torchvision stores the dataset")
    ap.add_argument("--manifest", default=None, help="output path (default: from config)")
    ap.add_argument("--check", action="store_true", help="rebuild and compare with the existing manifest")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = cfg["dataset"]["seed"]
    test_size = cfg["dataset"]["validation_fraction_of_trainval"]
    manifest_path = Path(args.manifest or REPO_ROOT / cfg["paths"]["manifest"])

    manifest = data.build_manifest(args.root, seed=seed, test_size=test_size, download=True)

    counts = manifest["split"].value_counts().to_dict()
    n_trainval = counts.get(data.SPLIT_GALLERY, 0) + counts.get(data.SPLIT_VAL, 0)
    print(f"images: {len(manifest)}  breeds: {manifest['breed'].nunique()}")
    print(f"gallery: {counts.get(data.SPLIT_GALLERY)}  val: {counts.get(data.SPLIT_VAL)}  test: {counts.get(data.SPLIT_TEST)}")
    exp = data.EXPECTED_SIZES
    if (n_trainval, counts.get(data.SPLIT_TEST), manifest["breed"].nunique()) != (exp["trainval"], exp["test"], exp["breeds"]):
        print("WARNING: sizes differ from the official Oxford-IIIT Pet numbers "
              f"(trainval={exp['trainval']}, test={exp['test']}, breeds={exp['breeds']})")

    if args.check:
        if not manifest_path.exists():
            print(f"no manifest at {manifest_path}")
            return 1
        old = data.load_manifest(manifest_path)
        same = old.equals(manifest[data.MANIFEST_COLUMNS].reset_index(drop=True))
        print("manifest is reproducible" if same else "MISMATCH between rebuilt and existing manifest")
        return 0 if same else 1

    data.save_manifest(manifest, manifest_path)
    print(f"saved {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
