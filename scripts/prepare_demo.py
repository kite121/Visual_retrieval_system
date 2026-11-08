"""Prepare demo gallery caches, example images and pretrained weights.

    python scripts/prepare_demo.py --embeddings-zip /path/to/second_models_embeddings.zip
    python scripts/prepare_demo.py --verify-models

Dataset images and model weights are downloaded once. No training is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import _bootstrap  # noqa: F401

from demo import service
from src import data, timing


def install_gallery(archive_path: Path, settings: service.Settings) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        archive_manifest = archive.read("configs/split_manifest.csv")
        if archive_manifest != settings.manifest.read_bytes():
            raise ValueError("The embedding archive uses a different dataset split.")
        records = [name for name in archive.namelist() if name.endswith("/run_metadata.json")]
        if len(records) != 1:
            raise ValueError("Expected exactly one run_metadata.json in the embedding archive.")
        metadata = json.loads(archive.read(records[0]))
        if metadata["dataset"]["manifest_sha256"] != hashlib.sha256(archive_manifest).hexdigest():
            raise ValueError("Embedding archive manifest hash is invalid.")
        archive_models = {item["model_id"] for item in metadata["models"]}
        if not set(settings.model_ids).issubset(archive_models):
            raise ValueError("The archive is missing one or more configured models.")
        for model_id in settings.model_ids:
            for filename in ("gallery.npz", "gallery.json"):
                suffix = f"/embeddings/{model_id}/{filename}"
                candidates = [name for name in archive.namelist() if name.endswith(suffix)]
                if len(candidates) != 1:
                    raise ValueError(f"Expected one gallery cache entry for {model_id}/{filename}.")
                destination = settings.embeddings_root / "embeddings" / model_id / filename
                payload = archive.read(candidates[0])
                if destination.exists() and destination.read_bytes() != payload:
                    raise ValueError(f"A different cache already exists at {destination}; move it before installing another run.")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings-zip", type=Path)
    parser.add_argument("--skip-model-download", action="store_true", help="prepare images and gallery caches only")
    parser.add_argument("--verify-models", action="store_true", help="run all four real encoders on all prepared queries")
    args = parser.parse_args()
    if args.skip_model_download and args.verify_models:
        parser.error("--verify-models cannot be combined with --skip-model-download")
    settings = service.load_settings()
    if args.embeddings_zip:
        install_gallery(args.embeddings_zip, settings)
    print("Preparing Oxford-IIIT Pet and checking the committed split…", flush=True)
    rebuilt = data.build_manifest(settings.dataset_root, seed=42, test_size=0.5, download=True)
    committed = data.load_manifest(settings.manifest)
    if not rebuilt.equals(committed):
        raise ValueError("Downloaded dataset does not reproduce the committed manifest.")
    cases = service.load_cases(settings)
    for case in cases:
        source = data.image_path(settings.dataset_root, case["image_id"])
        destination = settings.cases_path.parent / case["filename"]
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
            destination.chmod(0o644)
    missing = service.missing_assets(settings)
    if missing:
        raise ValueError("Missing demo assets:\n" + "\n".join(missing))
    signatures = service.gallery_signatures(settings) if args.verify_models else {}
    verification = {}
    for model_id in settings.model_ids:
        index = service.load_gallery(model_id, settings)
        print(f"Gallery OK: {model_id}, {tuple(index.embeddings.shape)}", flush=True)
        if args.skip_model_download:
            continue
        print(f"Loading pretrained weights: {model_id}", flush=True)
        encoder = service.load_encoder(model_id, settings.device)
        if args.verify_models:
            verification[model_id] = {}
            for case in cases:
                payload = (settings.cases_path.parent / case["filename"]).read_bytes()
                query_info = service.identify_query(payload, settings, signatures)
                result = service.search_image(payload, model_id, encoder, index, settings, query_info)
                verification[model_id][case["image_id"]] = {
                    "correct_top5": result["correct"],
                    "hit_ids": [hit.image_id for hit in result["hits"]],
                    "embedding_ms": result["embedding_ms"],
                }
                print(f"  {case['image_id']}: {result['correct']}/5", flush=True)
        del encoder, index
    if verification:
        service.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (service.CACHE_DIR / "verification.json").write_text(json.dumps({
            "hardware": timing.hardware_info(settings.device),
            "cases": verification,
        }, indent=2) + "\n")
    print("Demo assets ready. Launch: streamlit run demo/app.py", flush=True)


if __name__ == "__main__":
    main()
