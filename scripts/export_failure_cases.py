"""Validate and reuse the three committed diagnostic cases without full rankings."""
from __future__ import annotations

import argparse
import json
import shutil

import numpy as np
import pandas as pd
from PIL import Image

from build_final_results import DEFAULT_OUTPUT, MODELS, fingerprint, resolve, save_json
from src import data, metrics


def export(output=DEFAULT_OUTPUT, source=DEFAULT_OUTPUT, manifest="configs/split_manifest.csv"):
    output, source, manifest = map(resolve, (output, source, manifest))
    source_metadata = json.loads((source / "run_metadata.json").read_text(encoding="utf-8"))
    metadata_path = output / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = fingerprint(manifest)["sha256"]
    if source_metadata.get("manifest_sha256") != digest or metadata.get("manifest_sha256") != digest:
        raise ValueError("Saved cases and final analysis must use the same manifest")
    frame = data.load_manifest(manifest)
    by_id = frame.set_index("image_id")
    gallery = data.subset(frame, "gallery")
    gallery_ids = set(gallery.image_id)
    cases = pd.read_csv(source / "failure_cases.csv")
    if len(cases) != 3 or cases.query_id.nunique() != 3:
        raise ValueError("Expected three distinct saved diagnostic cases")

    figures = []
    for number, row in enumerate(cases.to_dict("records"), 1):
        query_id = row["query_id"]
        if row["model_id"] not in MODELS or row["split"] != "test":
            raise ValueError("Invalid model or split in saved cases")
        if query_id not in by_id.index or by_id.loc[query_id, "split"] != "test":
            raise ValueError(f"Query is not in the official test split: {query_id}")
        if row["query_breed"] != by_id.loc[query_id, "breed"]:
            raise ValueError(f"Query breed differs from manifest: {query_id}")
        hits = json.loads(row["hits_json"])
        if len(hits) != 5 or [h["rank"] for h in hits] != list(range(1, 6)):
            raise ValueError(f"Expected five ordered neighbors: {query_id}")
        ids = [h["image_id"] for h in hits]
        if len(set(ids)) != 5 or not set(ids) <= gallery_ids:
            raise ValueError(f"Neighbors must be distinct gallery images: {query_id}")
        scores = np.asarray([h["score"] for h in hits], dtype=float)
        if not np.isfinite(scores).all() or (np.abs(scores) > 1.00001).any() or (np.diff(scores) > 1e-6).any():
            raise ValueError(f"Invalid cosine scores: {query_id}")
        breeds = by_id.loc[ids, "breed"].tolist()
        for hit, breed in zip(hits, breeds):
            if hit["breed"] != breed or hit["same_breed"] != (breed == row["query_breed"]):
                raise ValueError(f"Neighbor annotations differ from manifest: {query_id}")
        measured = metrics.retrieval_metrics(
            by_id.loc[ids, "label"].to_numpy()[None, :],
            by_id.loc[[query_id], "label"].to_numpy(),
            gallery.label.to_numpy(), ks=(5,), hit_ks=(5,),
        )
        for column, key in (("precision_at_5", "P@5"), ("hit_at_5", "Hit@5")):
            value = float(row[column])
            if not np.isfinite(value) or abs(value - measured[key]) > 1e-6:
                raise ValueError(f"Saved {column} differs from the five neighbors: {query_id}")
        if measured["P@5"] == 1:
            raise ValueError(f"Selected query is not a failure: {query_id}")
        if not all(isinstance(row[k], str) and row[k].strip() for k in ("observation", "explanation")):
            raise ValueError(f"Missing diagnostic explanation: {query_id}")
        expected_figure = f"figures/failure_case_{number:02d}.png"
        if row["figure"] != expected_figure:
            raise ValueError(f"Unexpected figure path: {row['figure']}")
        image_path = source / expected_figure
        with Image.open(image_path) as image:
            if image.format != "PNG":
                raise ValueError(f"Expected a PNG figure: {image_path}")
            image.verify()
        figures.append(expected_figure)

    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    for name in figures:
        if (source / name).resolve() != (output / name).resolve():
            shutil.copyfile(source / name, output / name)
    cases.to_csv(output / "failure_cases.csv", index=False)
    previous = source_metadata.get("failure_cases", {})
    metadata["failure_cases"] = {
        "status": "complete",
        "mode": "reuse_saved_cases",
        "selection": previous.get("selection", {}),
        "interpretation": "Saved diagnostic examples from the separate 2026-10-31 run. Explanations are hypotheses, not causal findings.",
        "case_metrics_verified": True,
        "full_test_rankings_recomputed": False,
        "historical_test_metric_checks": previous.get("historical_test_metric_checks", previous.get("test_metric_checks", [])),
        "historical_source_files": previous.get("historical_source_files", previous.get("source_files", [])),
        "source_files": [fingerprint(output / p) for p in ["failure_cases.csv", *figures]],
    }
    save_json(metadata_path, metadata)
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--source", default=DEFAULT_OUTPUT, help="directory with the saved failure CSV, metadata, and three PNGs")
    parser.add_argument("--manifest", default="configs/split_manifest.csv")
    args = parser.parse_args()
    print(export(args.output, args.source, args.manifest)[["model_id", "query_id", "precision_at_5", "hit_at_5"]].to_string(index=False))
