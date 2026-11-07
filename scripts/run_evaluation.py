"""Shared evaluation entry point: gallery -> embeddings -> retrieval metrics -> timings.

    python scripts/run_evaluation.py                          # all models from the config
    python scripts/run_evaluation.py --models dino_vit_s16    # a single model
    python scripts/run_evaluation.py --rebuild                # ignore cached embeddings
    python scripts/run_evaluation.py --output-format canonical # final Stage 2 file layout

For every model it

1. builds the gallery embeddings (timed; cached in ``artifacts/embeddings/<model>/``),
2. embeds the validation and test queries,
3. ranks the gallery by cosine similarity and computes P@5/10, R@5/10, Hit@5 on both query sets,
4. measures single-query embedding / search latency (batch size 1, fixed query sample),
5. writes either the legacy per-model files or the canonical Stage 2 result files.

The default ``legacy`` output is retained for the existing Stage 3 workflow. The
``canonical`` output writes ``metrics_validation.csv``, ``metrics_test.csv``,
``runtime_summary.csv``, and ``run_metadata.json`` without placing intermediate
per-model JSON files in the results directory.

All four planned models are registered. Explicitly requested unknown model names
are errors; unregistered entries in a custom config are still reported and skipped.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
import numpy as np
import pandas as pd
import torch
import yaml

from visual_retrieval import data, embeddings, metrics, models, retrieval, timing

REPO_ROOT = _bootstrap.REPO_ROOT


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def get_embeddings(artifacts, model_name, split, dataset, encoder, args, pretrained, timed):
    """Cached embeddings of one split, computing (and timing, if ``timed``) them when needed."""
    cached = None if args.rebuild else embeddings.load_embeddings(artifacts, model_name, split, dataset.image_ids)
    if cached is not None and cached.meta.get("pretrained", True) == pretrained and (not timed or "build_seconds" in cached.meta):
        return cached.embeddings, {**cached.meta, "cache_reused": True}
    timing.warm_file_cache(dataset.paths)
    if timed:
        emb, seconds = timing.time_gallery_build(encoder, dataset, args.device, args.batch_size, args.num_workers)
    else:
        emb = embeddings.extract_embeddings(encoder, dataset, args.device, args.batch_size, args.num_workers)
        seconds = None
    meta = {
        "pretrained": pretrained,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": args.device,
        "n_images": len(dataset),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "cache_reused": False,
    }
    if seconds is not None:
        meta["build_seconds"] = seconds
    embeddings.save_embeddings(artifacts, model_name, split, emb, dataset.image_ids, dataset.labels, meta)
    return emb, meta


def evaluate_model(name, cfg, manifest, manifest_sha, args, paths) -> dict:
    spec = models.get_spec(name)
    pretrained = not args.random_weights
    print(f"\n=== {spec.display_name} ({name}) on {args.device} ===", flush=True)
    encoder = models.load_encoder(name, args.device, pretrained=pretrained)
    transform = embeddings.build_transform()

    frames = {s: data.subset(manifest, s) for s in data.SPLITS}
    datasets = {s: data.PetDataset(frames[s], paths["data_root"], transform) for s in data.SPLITS}

    # No query image may be in the gallery (plan: query/gallery are disjoint).
    gallery_ids = set(frames[data.SPLIT_GALLERY]["image_id"])
    for s in data.QUERY_SPLITS:
        assert gallery_ids.isdisjoint(frames[s]["image_id"]), f"{s} queries overlap with the gallery"

    gal_emb, gal_meta = get_embeddings(paths["artifacts"], name, data.SPLIT_GALLERY, datasets[data.SPLIT_GALLERY], encoder, args, pretrained, timed=True)
    print(f"gallery: {gal_emb.shape}, build time {gal_meta['build_seconds']:.1f} s", flush=True)

    gal = datasets[data.SPLIT_GALLERY]
    breed_of = dict(zip(manifest["label"], manifest["breed"]))
    index = retrieval.GalleryIndex(gal_emb, gal.image_ids, gal.labels, [breed_of[l] for l in gal.labels], device=args.device)

    ks, hit_ks = cfg["retrieval"]["ks"], cfg["retrieval"]["hit_ks"]
    kmax = max(*ks, *hit_ks)
    result_metrics, topk_dir = {}, paths["artifacts"] / "runs" / name
    topk_dir.mkdir(parents=True, exist_ok=True)
    for split in data.QUERY_SPLITS:
        ds = datasets[split]
        q_emb, _ = get_embeddings(paths["artifacts"], name, split, ds, encoder, args, pretrained, timed=False)
        scores, idx = index.search(q_emb, kmax)
        idx_np = idx.cpu().numpy()
        retrieved_labels = gal.labels[idx_np]
        result_metrics[split] = metrics.retrieval_metrics(retrieved_labels, ds.labels, gal.labels, ks, hit_ks)
        # Save rankings so failed queries can be analysed later without re-running models.
        np.savez(topk_dir / f"topk_{split}.npz", query_ids=np.asarray(ds.image_ids), query_labels=ds.labels,
                 gallery_ids=np.asarray(gal.image_ids)[idx_np], scores=scores.cpu().numpy())
        print(f"{split:>4}: " + "  ".join(f"{k}={v:.4f}" for k, v in result_metrics[split].items()), flush=True)

    t = cfg["timing"]
    latency = timing.measure_query_latency(
        encoder, data.PetDataset(frames[data.SPLIT_TEST], paths["data_root"], None), index, transform,
        k=t["search_k"], n_queries=args.n_queries or t["n_queries"], n_warmup=t["n_warmup"], seed=cfg["dataset"]["seed"],
    )
    print(f"query latency (median): embed {latency['embed_ms']['median']:.2f} ms, "
          f"search {latency['search_ms']['median']:.3f} ms, preprocess {latency['preprocess_ms']['median']:.1f} ms", flush=True)

    return {
        "model": name,
        "display_name": spec.display_name,
        "weights_source": spec.source if pretrained else "RANDOM WEIGHTS (smoke test, numbers are meaningless)",
        "embed_dim": spec.embed_dim,
        "pretrained": pretrained,
        "dataset_seed": cfg["dataset"]["seed"],
        "manifest_sha256": manifest_sha,
        "sizes": {s: len(frames[s]) for s in data.SPLITS},
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "precision": "fp32",
        "hardware": timing.hardware_info(args.device),
        "date": dt.date.today().isoformat(),
        "gallery_build_seconds": gal_meta["build_seconds"],
        "gallery_build_cache_reused": gal_meta.get("cache_reused", False),
        "timing_protocol": {
            "gallery": "warm file cache; loading, transforms, forward, L2 norm, host transfer; excludes disk save",
            "embedding": "batch 1; host-to-device transfer, forward and L2 norm; excludes preprocessing",
            "search": "exact cosine top-5 and transfer of indices to host",
        },
        "metrics": result_metrics,
        "query_latency": latency,
    }


def write_tables(results_dir: Path) -> None:
    """Aggregate every ``<model>.json`` in ``results_dir`` into ``metrics.csv`` and ``metrics.md``."""
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        if "metrics" in payload and "query_latency" in payload and "display_name" in payload:
            runs.append(payload)
    if not runs:
        return
    rows, trows = [], []
    for r in runs:
        for split, m in r["metrics"].items():
            rows.append({"model": r["display_name"], "split": split, **m})
        q = r["query_latency"]
        trows.append({
            "model": r["display_name"],
            "gallery_build_s": r["gallery_build_seconds"],
            "embed_ms_median": q["embed_ms"]["median"],
            "search_ms_median": q["search_ms"]["median"],
            "preprocess_ms_median": q["preprocess_ms"]["median"],
            "n_timing_queries": q["n_queries"],
            "device": r["hardware"].get("gpu", r["hardware"]["device"]),
            "batch_size": r["batch_size"],
            "date": r["date"],
        })
    mdf, tdf = pd.DataFrame(rows), pd.DataFrame(trows)
    mdf.to_csv(results_dir / "metrics.csv", index=False)
    tdf.to_csv(results_dir / "timing.csv", index=False)

    def md(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) and abs(v) < 10 else (f"{v:.2f}" if isinstance(v, float) else str(v))
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
        return "\n".join(lines)

    (results_dir / "metrics.md").write_text(
        "# Retrieval quality and runtime\n\n"
        "Generated by `scripts/run_evaluation.py`. `val` = second half of trainval, `test` = official test; "
        "gallery = first half of trainval.\n\n## Retrieval quality\n\n" + md(mdf) + "\n\n## Runtime\n\n" + md(tdf) + "\n"
    )


def write_canonical_results(results_dir: Path, runs: list[dict], cfg: dict, manifest_path: Path) -> None:
    """Write the four canonical Stage 2 result files from the completed in-memory runs."""
    if not runs:
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    metric_columns = [
        "model_id", "model", "precision_at_5", "recall_at_5",
        "precision_at_10", "recall_at_10", "hit_at_5",
    ]
    metric_names = {
        "precision_at_5": "P@5",
        "recall_at_5": "R@5",
        "precision_at_10": "P@10",
        "recall_at_10": "R@10",
        "hit_at_5": "Hit@5",
    }
    for split, filename in (("val", "metrics_validation.csv"), ("test", "metrics_test.csv")):
        rows = []
        for run in runs:
            values = run["metrics"][split]
            row = {"model_id": run["model"], "model": run["display_name"]}
            row.update({column: values[source] for column, source in metric_names.items()})
            rows.append(row)
        pd.DataFrame(rows, columns=metric_columns).to_csv(results_dir / filename, index=False)

    runtime_rows = []
    for run in runs:
        latency = run["query_latency"]
        runtime_rows.append({
            "model_id": run["model"],
            "model": run["display_name"],
            "gallery_build_seconds": run["gallery_build_seconds"],
            "gallery_build_cache_reused": run.get("gallery_build_cache_reused", False),
            "n_queries": latency["n_queries"],
            "search_k": latency["k"],
            "embed_median_ms": latency["embed_ms"]["median"],
            "embed_mean_ms": latency["embed_ms"]["mean"],
            "embed_p95_ms": latency["embed_ms"]["p95"],
            "search_median_ms": latency["search_ms"]["median"],
            "search_mean_ms": latency["search_ms"]["mean"],
            "search_p95_ms": latency["search_ms"]["p95"],
            "preprocess_median_ms": latency["preprocess_ms"]["median"],
            "preprocess_mean_ms": latency["preprocess_ms"]["mean"],
            "preprocess_p95_ms": latency["preprocess_ms"]["p95"],
            "device": run["hardware"]["device"],
            "gpu": run["hardware"].get("gpu"),
            "batch_size": run["batch_size"],
            "date": run["date"],
        })
    pd.DataFrame(runtime_rows).to_csv(results_dir / "runtime_summary.csv", index=False)

    first = runs[0]
    metadata = {
        "schema_version": 1,
        "experiment_id": "01_first_models",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": {
            "name": cfg["dataset"]["name"],
            "target_type": cfg["dataset"]["target_type"],
            "seed": cfg["dataset"]["seed"],
            "manifest_path": manifest_path.relative_to(REPO_ROOT).as_posix()
            if manifest_path.is_relative_to(REPO_ROOT) else str(manifest_path),
            "manifest_sha256": first["manifest_sha256"],
            "gallery_fraction_of_trainval": cfg["dataset"]["gallery_fraction_of_trainval"],
            "validation_fraction_of_trainval": cfg["dataset"]["validation_fraction_of_trainval"],
            "final_query_split": cfg["dataset"]["final_query_split"],
            "sizes": {
                "gallery": first["sizes"]["gallery"],
                "validation": first["sizes"]["val"],
                "test": first["sizes"]["test"],
            },
        },
        "preprocessing": cfg["preprocessing"],
        "retrieval": {
            "similarity": cfg["retrieval"]["similarity"],
            "exclude_query_from_gallery": cfg["retrieval"]["exclude_query_from_gallery"],
            "precision_recall_at_k": cfg["retrieval"]["ks"],
            "hit_at_k": cfg["retrieval"]["hit_ks"],
        },
        "embedding": {
            "batch_size": first["batch_size"],
            "num_workers": first["num_workers"],
            "precision": first["precision"],
        },
        "timing": {
            "n_queries": first["query_latency"]["n_queries"],
            "n_warmup": cfg["timing"]["n_warmup"],
            "search_k": first["query_latency"]["k"],
            "protocol": first["timing_protocol"],
        },
        "environment": first["hardware"],
        "models": [
            {
                "model_id": run["model"],
                "display_name": run["display_name"],
                "weights_source": run["weights_source"],
                "embedding_dimension": run["embed_dim"],
                "pretrained": run["pretrained"],
            }
            for run in runs
        ],
    }
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--models", nargs="+", default=None, help="model names (default: registered ones from the config)")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--root", default=None, help="dataset root (default: from config)")
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--n-queries", type=int, default=None, help="number of single-image timing queries")
    ap.add_argument("--rebuild", action="store_true", help="recompute embeddings even if cached")
    ap.add_argument("--random-weights", action="store_true", help="smoke test only: skip pretrained weights")
    ap.add_argument(
        "--output-format",
        choices=("legacy", "canonical"),
        default="legacy",
        help="legacy keeps Stage 3 per-model JSON files; canonical writes the four Stage 2 files",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.batch_size = args.batch_size or cfg["embedding"]["batch_size"]
    args.num_workers = cfg["embedding"]["num_workers"] if args.num_workers is None else args.num_workers
    paths = {
        "manifest": resolve(args.manifest or cfg["paths"]["manifest"]),
        "data_root": resolve(args.root or cfg["paths"]["data_root"]),
        "artifacts": resolve(args.artifacts or cfg["paths"]["artifacts"]),
        "results": resolve(args.results_dir or cfg["paths"]["results"]),
    }
    if not paths["manifest"].exists():
        sys.exit(f"manifest not found: {paths['manifest']} -- run scripts/prepare_dataset.py first")
    manifest = data.load_manifest(paths["manifest"])
    manifest_sha = hashlib.sha256(paths["manifest"].read_bytes()).hexdigest()

    wanted = args.models or cfg["models"]
    registered = set(models.available_models())
    unknown = set(wanted) - registered
    if args.models and unknown:
        sys.exit(f"unknown model names: {sorted(unknown)}; available: {sorted(registered)}")
    todo = [m for m in wanted if m in registered]
    for m in wanted:
        if m not in registered:
            print(f"skipping '{m}': not registered yet (adapter not implemented)")
    if not todo:
        sys.exit("no registered models to evaluate")

    paths["results"].mkdir(parents=True, exist_ok=True)
    completed_runs = []
    for name in todo:
        result = evaluate_model(name, cfg, manifest, manifest_sha, args, paths)
        completed_runs.append(result)
        if args.output_format == "legacy":
            (paths["results"] / f"{name}.json").write_text(json.dumps(result, indent=2))
        else:
            raw_path = paths["artifacts"] / "runs" / name / "evaluation.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps(result, indent=2) + "\n")
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    if args.output_format == "canonical":
        write_canonical_results(paths["results"], completed_runs, cfg, paths["manifest"])
    else:
        write_tables(paths["results"])
    print(f"\nwrote tables to {paths['results']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
