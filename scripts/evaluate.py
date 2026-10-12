"""Shared evaluation entry point: gallery -> embeddings -> retrieval metrics -> timings.

    python scripts/evaluate.py                          # all registered models from the config
    python scripts/evaluate.py --models dino_vit_s16    # a single model
    python scripts/evaluate.py --rebuild                # ignore cached embeddings

For every model it

1. builds the gallery embeddings (timed; cached in ``artifacts/embeddings/<model>/``),
2. embeds the validation and test queries,
3. ranks the gallery by cosine similarity and computes P@5/10, R@5/10, Hit@5 on both query sets,
4. measures single-query embedding / search latency (batch size 1, fixed query sample),
5. writes ``results/first_models/<model>.json`` and refreshes ``metrics.csv`` / ``metrics.md``.

Models that are in the config but not registered yet (DINOv2, DeiT-B) are skipped, so the
same script works unchanged once Karim registers them.
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
        return cached.embeddings, cached.meta
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
        "metrics": result_metrics,
        "query_latency": latency,
    }


def write_tables(results_dir: Path) -> None:
    """Aggregate every ``<model>.json`` in ``results_dir`` into ``metrics.csv`` and ``metrics.md``."""
    runs = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]
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
        "# First models: quality and runtime\n\n"
        "Generated by `scripts/evaluate.py`. `val` = second half of trainval, `test` = official test; "
        "gallery = first half of trainval.\n\n## Retrieval quality\n\n" + md(mdf) + "\n\n## Runtime\n\n" + md(tdf) + "\n"
    )


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
        sys.exit(f"manifest not found: {paths['manifest']} -- run scripts/prepare_data.py first")
    manifest = data.load_manifest(paths["manifest"])
    manifest_sha = hashlib.sha256(paths["manifest"].read_bytes()).hexdigest()

    wanted = args.models or cfg["models"]
    registered = set(models.available_models())
    todo = [m for m in wanted if m in registered]
    for m in wanted:
        if m not in registered:
            print(f"skipping '{m}': not registered yet (adapter not implemented)")
    if not todo:
        sys.exit("no registered models to evaluate")

    paths["results"].mkdir(parents=True, exist_ok=True)
    for name in todo:
        result = evaluate_model(name, cfg, manifest, manifest_sha, args, paths)
        (paths["results"] / f"{name}.json").write_text(json.dumps(result, indent=2))
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    write_tables(paths["results"])
    print(f"\nwrote tables to {paths['results']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
