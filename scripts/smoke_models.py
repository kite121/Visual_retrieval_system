"""Verify the new encoders and shared retrieval on synthetic images, not quality.

Default: download real pretrained weights and run on CPU. --random-weights is
an offline interface check. No Oxford-Pet metrics are produced by this script.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch
from PIL import Image

from visual_retrieval import embeddings, models, retrieval, timing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["dinov2_vit_b14", "deit_vit_b16_supervised"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--output", default="artifacts/local_checks/pretrained_smoke.json")
    args = ap.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rng = np.random.default_rng(42)
    images = [Image.fromarray(rng.integers(0, 256, (256, 320, 3), dtype=np.uint8)) for _ in range(6)]
    transform = embeddings.build_transform()
    batch = torch.stack([transform(image) for image in images])
    rows = []
    for name in args.models:
        print(f"checking {name}, pretrained={not args.random_weights}", flush=True)
        encoder = models.load_encoder(name, args.device, pretrained=not args.random_weights)
        vectors = embeddings.encode_batch(encoder, batch, args.device).cpu()
        assert vectors.shape == (6, models.get_spec(name).embed_dim)
        assert torch.isfinite(vectors).all()
        assert torch.allclose(vectors.norm(dim=1), torch.ones(6), atol=1e-5)
        assert not encoder.training and not any(p.requires_grad for p in encoder.parameters())
        ids = [f"synthetic_{i}" for i in range(6)]
        index = retrieval.GalleryIndex(vectors, ids, np.arange(6), device=args.device)
        hits = retrieval.query(images[2], encoder, index, k=5)
        assert len(hits) == 5 and hits[0].image_id == ids[2]
        assert abs(hits[0].score - 1.0) < 1e-4
        excluded = retrieval.query(images[2], encoder, index, k=5, exclude_ids=[ids[2]])
        assert ids[2] not in [h.image_id for h in excluded]
        rows.append({"model": name, "source": models.get_spec(name).source,
                     "shape": list(vectors.shape), "finite": True, "unit_norm": True,
                     "frozen": True, "self_match": True, "exclusion": True})
        del encoder, index, vectors
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    result = {"purpose": "synthetic interface smoke, not Oxford-Pet retrieval quality",
              "pretrained": not args.random_weights, "hardware": timing.hardware_info(args.device),
              "checks": rows}
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS: {len(rows)} encoders; evidence: {dest}")


if __name__ == "__main__":
    main()
