"""Demo asset validation and live queries using the shared retrieval pipeline."""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "demo" / ".cache"
# Keep downloaded model assets beside the demo and outside version control.
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
import yaml

from src import data, embeddings, models, retrieval, timing


@dataclass(frozen=True)
class Settings:
    manifest: Path
    dataset_root: Path
    embeddings_root: Path
    cases_path: Path
    results: Path
    model_ids: tuple[str, ...]
    default_model: str
    top_k: int
    device: str


def load_settings(path: Path | None = None) -> Settings:
    config = yaml.safe_load((path or ROOT / "demo" / "config.yaml").read_text())
    paths = config["paths"]
    names = tuple(config["models"])
    if not names or len(set(names)) != len(names):
        raise ValueError("The model list must be nonempty and contain no duplicates.")
    for name in names:
        models.get_spec(name)
    if config["default_model"] not in names:
        raise ValueError("The default model is not in the configured model list.")
    top_k = int(config["top_k"])
    if top_k != 5:
        raise ValueError("This demo uses Top-5 to match the prepared cases and report.")
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("Choose auto, cpu, or cuda in demo/config.yaml.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is configured but unavailable; choose auto or cpu.")
    if device == "cpu":
        torch.set_num_threads(int(config.get("cpu_threads", 4)))
    return Settings(
        manifest=(ROOT / paths["manifest"]).resolve(),
        dataset_root=(ROOT / paths["dataset_root"]).resolve(),
        embeddings_root=(ROOT / paths["embeddings_root"]).resolve(),
        cases_path=(ROOT / paths["cases"]).resolve(),
        results=(ROOT / paths["results"]).resolve(),
        model_ids=names,
        default_model=config["default_model"],
        top_k=top_k,
        device=device,
    )


def load_cases(settings: Settings) -> list[dict]:
    cases = json.loads(settings.cases_path.read_text())
    manifest = data.load_manifest(settings.manifest).set_index("image_id")
    for case in cases:
        row = manifest.loc[case["image_id"]]
        if row["split"] != "test" or row["breed"] != case["breed"]:
            raise ValueError(f"Prepared case disagrees with the manifest: {case['image_id']}")
        if Path(case["filename"]).name != case["filename"]:
            raise ValueError("Prepared-case filenames must not contain directories.")
    return cases


def gallery_frame(settings: Settings) -> pd.DataFrame:
    manifest = data.load_manifest(settings.manifest)
    return data.subset(manifest, "gallery")


def missing_assets(settings: Settings) -> list[str]:
    missing = []
    for name in settings.model_ids:
        for suffix in ("npz", "json"):
            path = settings.embeddings_root / "embeddings" / name / f"gallery.{suffix}"
            if not path.is_file():
                missing.append(str(path.relative_to(ROOT)))
    gallery = gallery_frame(settings)
    absent = sum(not data.image_path(settings.dataset_root, image_id).is_file() for image_id in gallery.image_id)
    if absent:
        missing.append(f"{absent} gallery images in {data.images_dir(settings.dataset_root)}")
    for case in load_cases(settings):
        if not (settings.cases_path.parent / case["filename"]).is_file():
            missing.append(f"demo/example_queries/{case['filename']}")
    return missing


def load_gallery(model_id: str, settings: Settings) -> retrieval.GalleryIndex:
    if model_id not in settings.model_ids:
        raise ValueError("Unknown demo model.")
    frame = gallery_frame(settings)
    cached = embeddings.load_embeddings(settings.embeddings_root, model_id, "gallery", frame.image_id.tolist())
    if cached is None:
        raise ValueError(f"Gallery cache missing or inconsistent with the manifest: {model_id}")
    spec = models.get_spec(model_id)
    if cached.embeddings.shape != (len(frame), spec.embed_dim):
        raise ValueError(f"Unexpected gallery embedding shape for {model_id}.")
    if not np.array_equal(cached.labels, frame.label.to_numpy()):
        raise ValueError(f"Gallery labels disagree with the manifest for {model_id}.")
    if not np.isfinite(cached.embeddings).all() or not cached.meta.get("pretrained", False):
        raise ValueError(f"Invalid or non-pretrained gallery cache for {model_id}.")
    return retrieval.GalleryIndex(
        cached.embeddings, cached.image_ids, cached.labels,
        frame.breed.tolist(), device=settings.device,
    )


def load_encoder(model_id: str, device: str):
    encoder = models.load_encoder(model_id, device)
    timing.warmup(encoder, device, n=2)
    return encoder


def decode_image(image_bytes: bytes) -> Image.Image:
    if len(image_bytes) > 20 * 1024 * 1024:
        raise ValueError("Choose an image smaller than 20 MiB.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The file could not be opened as a valid image.") from exc


def gallery_signatures(settings: Settings) -> dict[str, list[str]]:
    """Identify byte-identical gallery uploads so they can be excluded from search."""
    signatures: dict[str, list[str]] = {}
    for image_id in gallery_frame(settings).image_id:
        digest = hashlib.sha256(data.image_path(settings.dataset_root, image_id).read_bytes()).hexdigest()
        signatures.setdefault(digest, []).append(image_id)
    return signatures


def identify_query(image_bytes: bytes, settings: Settings, signatures: dict[str, list[str]]) -> dict:
    digest = hashlib.sha256(image_bytes).hexdigest()
    for case in load_cases(settings):
        path = settings.cases_path.parent / case["filename"]
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            return {"breed": case["breed"], "image_id": case["image_id"], "exclude_ids": []}
    excluded = signatures.get(digest, [])
    if excluded:
        manifest = data.load_manifest(settings.manifest).set_index("image_id")
        return {"breed": str(manifest.loc[excluded[0], "breed"]), "image_id": excluded[0], "exclude_ids": excluded}
    return {"breed": None, "image_id": None, "exclude_ids": []}


def search_image(image_bytes: bytes, model_id: str, encoder, index, settings: Settings, query_info: dict) -> dict:
    """Time live preprocessing, embedding and search; model loading is excluded."""
    start = time.perf_counter()
    tensor = embeddings.build_transform()(decode_image(image_bytes)).unsqueeze(0)
    preprocessing_ms = (time.perf_counter() - start) * 1000
    timing.sync(settings.device)
    start = time.perf_counter()
    vector = embeddings.encode_batch(encoder, tensor, settings.device)
    timing.sync(settings.device)
    embedding_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    scores, indices = index.search(vector, settings.top_k, exclude_ids=query_info["exclude_ids"])
    scores, indices = scores.cpu(), indices.cpu()
    timing.sync(settings.device)
    search_ms = (time.perf_counter() - start) * 1000
    hits = index.hits(scores[0], indices[0])
    breed = query_info["breed"]
    correct = sum(hit.breed == breed for hit in hits) if breed else None
    return {
        "model_id": model_id,
        "display_name": models.get_spec(model_id).display_name,
        "hits": hits,
        "correct": correct,
        "precision_at_5": correct / settings.top_k if correct is not None else None,
        "preprocessing_ms": preprocessing_ms,
        "embedding_ms": embedding_ms,
        "search_ms": search_ms,
        "total_ms": preprocessing_ms + embedding_ms + search_ms,
    }
