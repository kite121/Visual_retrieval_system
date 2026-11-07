"""Repeated fresh timings on the same gallery and query sample; no quality tuning.

Example: python scripts/benchmark_timing.py --device cuda --repeats 3
Every repeat rebuilds gallery features without an embedding cache. Files are
checkpointed after every model/repeat and kept separate from quality results.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch
import yaml

from visual_retrieval import data, embeddings, models, retrieval, timing


def summarize(records: list[dict]) -> pd.DataFrame:
    rows = []
    for name in dict.fromkeys(r["model"] for r in records):
        subset = [r for r in records if r["model"] == name]
        row = {"model": name, "repeats": len(subset)}
        values = {"gallery_build_s": [r["gallery_build_seconds"] for r in subset]}
        for kind in ("embed", "search", "preprocess"):
            values[f"{kind}_ms"] = [r["query_latency"][f"{kind}_ms"]["median"] for r in subset]
        for key, numbers in values.items():
            row.update({f"{key}_median": float(np.median(numbers)),
                        f"{key}_min": float(np.min(numbers)), f"{key}_max": float(np.max(numbers))})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(_bootstrap.REPO_ROOT / "configs/default.yaml"))
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--output-dir", default="results/stage3/timing_repeats")
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error("repeats must be positive")
    cfg = yaml.safe_load(Path(args.config).read_text())
    resolve = lambda p: Path(p) if Path(p).is_absolute() else _bootstrap.REPO_ROOT / p
    manifest_path = resolve(args.manifest or cfg["paths"]["manifest"])
    root = resolve(args.root or cfg["paths"]["data_root"])
    dest = resolve(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    names = args.models or cfg["models"]
    for name in names:
        models.get_spec(name)
    manifest = data.load_manifest(manifest_path)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    transform = embeddings.build_transform()
    gal = data.PetDataset(data.subset(manifest, "gallery"), root, transform)
    queries = data.PetDataset(data.subset(manifest, "test"), root, None)
    batch = args.batch_size or cfg["embedding"]["batch_size"]
    workers = cfg["embedding"]["num_workers"] if args.num_workers is None else args.num_workers
    records = []
    tcfg = cfg["timing"]
    for repeat in range(1, args.repeats + 1):
        # Rotate the order between repeats to reduce systematic ordering effects.
        offset = (repeat - 1) % len(names)
        for name in names[offset:] + names[:offset]:
            print(f"repeat {repeat}/{args.repeats}: {name}", flush=True)
            encoder = models.load_encoder(name, args.device)
            timing.warm_file_cache(gal.paths)
            vectors, seconds = timing.time_gallery_build(encoder, gal, args.device, batch, workers)
            index = retrieval.GalleryIndex(vectors, gal.image_ids, gal.labels, device=args.device)
            latency = timing.measure_query_latency(
                encoder, queries, index, transform, k=tcfg["search_k"],
                n_queries=tcfg["n_queries"], n_warmup=tcfg["n_warmup"], seed=cfg["dataset"]["seed"],
            )
            record = {"model": name, "repeat": repeat, "pretrained": True,
                      "weights_source": models.get_spec(name).source,
                      "manifest_sha256": digest, "hardware": timing.hardware_info(args.device),
                      "precision": "fp32", "batch_size": batch, "num_workers": workers,
                      "gallery_size": len(gal), "date": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "gallery_build_seconds": seconds, "gallery_build_cache_reused": False,
                      "query_latency": latency}
            records.append(record)
            (dest / f"{name}_repeat{repeat}.json").write_text(json.dumps(record, indent=2) + "\n")
            summarize(records).to_csv(dest / "summary.csv", index=False)
            print(f"saved {name}: gallery {seconds:.2f}s, embed {latency['embed_ms']['median']:.2f}ms", flush=True)
            del encoder, vectors, index
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    print(f"completed {len(records)} timing measurements: {dest}")


if __name__ == "__main__":
    main()
