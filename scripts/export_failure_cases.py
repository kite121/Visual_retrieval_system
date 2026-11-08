"""Verify saved test rankings against canonical metrics and export three real failures."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from build_final_results import DEFAULT_OUTPUT, METRICS, MODELS, fingerprint, resolve, save_json
from src import data, metrics

DEFAULT_EVIDENCE = "results/final_analysis_evidence"


def load_rankings(artifacts, model, manifest):
    """Use saved rankings, or shared embedding loader and search without loading weights."""
    path = artifacts / "runs" / model / "topk_test.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in ("query_ids", "gallery_ids", "scores")}, [path]
    from src import embeddings, retrieval
    frames = {s: data.subset(manifest, s) for s in ("gallery", "test")}
    cache = {s: embeddings.load_embeddings(artifacts, model, s, frames[s].image_id.tolist()) for s in frames}
    if any(c is None for c in cache.values()):
        raise FileNotFoundError(f"Missing {path} and compatible gallery/test embeddings. Restore embeddings or run scripts/run_evaluation.py --save-rankings")
    for s, c in cache.items():
        if not np.array_equal(c.labels, frames[s].label.to_numpy()):
            raise ValueError(f"Cached labels differ from manifest: {model}/{s}")
    g, q = cache["gallery"], cache["test"]
    index = retrieval.GalleryIndex(g.embeddings, g.image_ids, g.labels, device="cpu")
    scores, indices = index.search(q.embeddings, k=10)
    files = [artifacts / "embeddings" / model / f"{s}{ext}" for s in frames for ext in (".npz", ".json")]
    return {"query_ids": np.asarray(q.image_ids), "gallery_ids": np.asarray(g.image_ids)[indices.numpy()],
            "scores": scores.numpy()}, [p for p in files if p.exists()]


def export(output=DEFAULT_OUTPUT, evidence=DEFAULT_EVIDENCE, artifacts=None, root=None,
           manifest="configs/split_manifest.csv"):
    output, evidence, manifest_path = map(resolve, (output, evidence, manifest))
    artifacts = resolve(artifacts) if artifacts else evidence / "artifacts"
    root = resolve(root) if root else evidence / "data"
    plan_path = evidence / "selection.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    selected = plan["cases"]
    if len(selected) != 3 or len({c["query_id"] for c in selected}) != 3:
        raise ValueError("Selection must specify three distinct query IDs")
    manifest = data.load_manifest(manifest_path)
    by_id = manifest.set_index("image_id")
    expected = data.subset(manifest, "test").image_id.to_numpy()
    gal = data.subset(manifest, "gallery")
    provenance_files, rankings, comparisons = [plan_path], {}, []
    canonical = pd.read_csv(output / "final_quality_metrics.csv")
    metadata_path = output / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = fingerprint(manifest_path)["sha256"]
    if metadata["manifest_sha256"] != digest:
        raise ValueError("Final analysis manifest differs")
    for model in MODELS:
        provenance_path = artifacts / "runs" / model / "evaluation.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance["model"] != model or not provenance["pretrained"] or provenance["manifest_sha256"] != digest:
            raise ValueError(f"Invalid ranking provenance: {model}")
        provenance_files.append(provenance_path)
        r, paths = load_rankings(artifacts, model, manifest)
        provenance_files.extend(paths)
        q, g, scores = r["query_ids"].astype(str), r["gallery_ids"].astype(str), r["scores"]
        if not np.array_equal(q, expected) or g.shape != (len(q), 10) or scores.shape != g.shape:
            raise ValueError(f"Ranking IDs/shapes differ from test protocol: {model}")
        if not set(g.flat) <= set(gal.image_id) or any(len(set(row)) != 10 for row in g):
            raise ValueError(f"Invalid or duplicate gallery IDs: {model}")
        if not np.isfinite(scores).all() or (np.diff(scores, axis=1) > 1e-6).any() or (np.abs(scores) > 1.00001).any():
            raise ValueError(f"Invalid similarity scores: {model}")
        labels = by_id.loc[g.ravel(), "label"].to_numpy().reshape(g.shape)
        qlabels = by_id.loc[q, "label"].to_numpy()
        calculated = metrics.retrieval_metrics(labels, qlabels, gal.label.to_numpy())
        row = canonical[(canonical.model_id == model) & (canonical.split == "test")].iloc[0]
        differences = {key: abs(calculated[name] - float(row[key])) for key, name in METRICS.items()}
        if max(differences.values()) > 1e-6:
            raise ValueError(f"Ranking metrics differ from canonical test results: {model}: {differences}")
        comparisons.append({"model_id": model, "max_absolute_difference": max(differences.values()),
                            "ranking_run_date": provenance["date"], "ranking_hardware": provenance["hardware"]})
        rankings[model] = r
    # Validate every selected image before writing a CSV or a figure.
    image_files = []
    for c in selected:
        r = rankings[c["model_id"]]
        i = list(r["query_ids"]).index(c["query_id"])
        ids = [c["query_id"]] + list(r["gallery_ids"][i, :5])
        for image_id in ids:
            p = data.image_path(root, str(image_id))
            data.load_image(p)
            image_files.append(p)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, c in enumerate(selected, 1):
        model, query_id = c["model_id"], c["query_id"]
        r = rankings[model]
        i = list(r["query_ids"]).index(query_id)
        ids = [query_id] + [str(g) for g in r["gallery_ids"][i, :5]]
        breeds = by_id.loc[ids, "breed"].tolist()
        labels = by_id.loc[ids, "label"].to_numpy()
        qm = metrics.retrieval_metrics(labels[None, 1:], labels[:1], gal.label.to_numpy(), ks=(5,), hit_ks=(5,))
        if qm["P@5"] == 1:
            raise ValueError(f"Selected query is not a failure: {model}/{query_id}")
        sheet = Image.new("RGB", (6*256, 338), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f"{model} | test | P@5={qm['P@5']:.1f} Hit@5={qm['Hit@5']:.0f}", fill="black")
        for j, image_id in enumerate(ids):
            tile = ImageOps.contain(data.load_image(data.image_path(root, image_id)), (248, 240))
            x = j*256
            sheet.paste(tile, (x+(256-tile.width)//2, 34+(240-tile.height)//2))
            title = "QUERY" if j == 0 else f"#{j} cosine={float(r['scores'][i,j-1]):.4f}"
            draw.text((x+4, 279), title, fill="black")
            draw.text((x+4, 296), breeds[j], fill="black")
            draw.text((x+4, 313), image_id, fill="black")
        figure = f"figures/failure_case_{number:02d}.png"
        sheet.save(output / figure)
        hits = [{"rank": j, "image_id": ids[j], "breed": breeds[j],
                 "score": float(r["scores"][i,j-1]), "same_breed": breeds[j] == breeds[0]} for j in range(1,6)]
        rows.append({"model_id": model, "split": "test", "query_id": query_id,
                     "query_breed": breeds[0], "precision_at_5": qm["P@5"], "hit_at_5": qm["Hit@5"],
                     "observation": c["observation"], "explanation": c["explanation"], "figure": figure,
                     "hits_json": json.dumps(hits)})
    pd.DataFrame(rows).to_csv(output / "failure_cases.csv", index=False)
    metadata["failure_cases"] = {"status": "complete", "selection": plan,
        "interpretation": "Diagnostic examples; explanations are hypotheses, not causal findings. Rankings are from a separate verified run, not the canonical timing run.",
        "metric_tolerance": 1e-6, "test_metric_checks": comparisons,
        "source_files": [fingerprint(p) for p in dict.fromkeys(provenance_files + image_files)]}
    save_json(metadata_path, metadata)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--evidence", default=DEFAULT_EVIDENCE)
    p.add_argument("--artifacts", default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--manifest", default="configs/split_manifest.csv")
    a = p.parse_args()
    print(export(a.output, a.evidence, a.artifacts, a.root, a.manifest)[["model_id", "query_id", "precision_at_5", "hit_at_5"]].to_string(index=False))
