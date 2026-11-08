"""Validate canonical experiment tables and assemble the final analysis (no inference)."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import _bootstrap
import numpy as np
import pandas as pd
from src import data

ROOT = _bootstrap.REPO_ROOT
DEFAULT_SOURCE = "results/02_second_models"
DEFAULT_OUTPUT = "results/03_final_analysis"
METRICS = {"precision_at_5": "P@5", "recall_at_5": "R@5",
           "precision_at_10": "P@10", "recall_at_10": "R@10", "hit_at_5": "Hit@5"}
MODELS = ("dino_vit_s16", "deit_vit_s16_supervised", "dinov2_vit_b14", "deit_vit_b16_supervised")


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def fingerprint(path):
    p = resolve(path).resolve()
    return {"path": p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT, manifest="configs/split_manifest.csv"):
    source, output, manifest = map(resolve, (source, output, manifest))
    meta_path = source / "run_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frame = data.load_manifest(manifest)
    digest = fingerprint(manifest)["sha256"]
    if digest != meta["dataset"]["manifest_sha256"]:
        raise ValueError("Source metadata and manifest SHA-256 differ")
    for split, key in (("gallery", "gallery"), ("val", "validation"), ("test", "test")):
        if len(data.subset(frame, split)) != meta["dataset"]["sizes"][key]:
            raise ValueError(f"Source/manifest size mismatch: {split}")
    if {m["model_id"] for m in meta["models"] if m["pretrained"]} != set(MODELS):
        raise ValueError("Expected four pretrained models in source metadata")
    sources, tables = [meta_path, manifest], []
    for split in ("validation", "test"):
        p = source / f"metrics_{split}.csv"
        t = pd.read_csv(p)
        if len(t) != 4 or set(t.model_id) != set(MODELS):
            raise ValueError(f"Expected exactly four unique models: {p}")
        values = t[list(METRICS)].to_numpy(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Invalid quality metrics: {p}")
        t = t.set_index("model_id").loc[list(MODELS)].reset_index()
        t.insert(2, "split", split)
        tables.append(t)
        sources.append(p)
    runtime_path = source / "runtime_summary.csv"
    runtime = pd.read_csv(runtime_path)
    if len(runtime) != 4 or set(runtime.model_id) != set(MODELS):
        raise ValueError("Runtime must contain four unique models")
    for prefix in ("gallery_build_seconds", "preprocess", "embed", "search"):
        cols = [f"{prefix}_{s}" if prefix == "gallery_build_seconds" else f"{prefix}_{s}_ms"
                for s in ("min", "median", "max")]
        a = runtime[cols].to_numpy(float)
        if not np.isfinite(a).all() or (a < 0).any() or (np.diff(a, axis=1) < 0).any():
            raise ValueError(f"Invalid repeated runtime statistics: {prefix}")
    context = ["device", "gpu", "batch_size", "num_workers", "n_queries", "search_k", "repeats"]
    if runtime[context].isna().any().any() or len(runtime[context].drop_duplicates()) != 1:
        raise ValueError("Incomparable or incomplete runtime contexts")
    if (runtime.repeats != meta["timing"]["repeats_per_model"]).any():
        raise ValueError("Repeat count differs from metadata")
    sources.append(runtime_path)
    previous_failures = {}
    previous_metadata = output / "run_metadata.json"
    if previous_metadata.exists() and (output / "failure_cases.csv").exists():
        previous = json.loads(previous_metadata.read_text(encoding="utf-8"))
        if previous.get("manifest_sha256") != digest:
            raise ValueError("Saved failure cases use a different manifest")
        previous_failures = previous.get("failure_cases", {})
    output.mkdir(parents=True, exist_ok=True)
    quality = pd.concat(tables, ignore_index=True)
    quality.to_csv(output / "final_quality_metrics.csv", index=False)
    runtime.set_index("model_id").loc[list(MODELS)].reset_index().to_csv(output / "runtime_summary.csv", index=False)
    metadata = {"schema_version": 1, "analysis_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "manifest_sha256": digest, "source_files": [fingerprint(p) for p in sources],
                "source_run_metadata": meta, "quality_rows": len(quality),
                "timing_aggregation": "Copied canonical repeated statistics, not legacy stage3 timings",
                "failure_cases": {**previous_failures, "status": "pending"}}
    save_json(output / "run_metadata.json", metadata)
    return quality, runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--manifest", default="configs/split_manifest.csv")
    a = p.parse_args()
    build(a.source, a.output, a.manifest)
    print(f"Validated 8 quality rows and 4 repeated runtime rows -> {resolve(a.output)}")


if __name__ == "__main__":
    main()
