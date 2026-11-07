"""Export query plus top-5 contact sheets from saved validation rankings.

These are illustrations of an already completed evaluation, not settings chosen
from test results. Also save query/hit IDs, breed agreement and similarity scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from visual_retrieval import data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["dinov2_vit_b14", "deit_vit_b16_supervised"])
    ap.add_argument("--manifest", default="configs/split_manifest.csv")
    ap.add_argument("--root", default="data")
    ap.add_argument("--artifacts", default="artifacts/stage3")
    ap.add_argument("--output-dir", default="results/stage3/examples")
    args = ap.parse_args()
    manifest = data.load_manifest(args.manifest)
    breed = manifest.set_index("image_id")["breed"].to_dict()
    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        with np.load(Path(args.artifacts) / "runs" / name / "topk_val.npz", allow_pickle=False) as z:
            query_ids, gallery_ids, scores = z["query_ids"], z["gallery_ids"], z["scores"]
        # First query is a fixed illustration; first Hit@5 failure, if any, is a diagnostic.
        failures = [i for i, q in enumerate(query_ids)
                    if all(breed[str(g)] != breed[str(q)] for g in gallery_ids[i, :5])]
        chosen = [(0, "fixed")]
        if failures and failures[0] != 0:
            chosen.append((failures[0], "failure"))
        descriptions = []
        for i, kind in chosen:
            q = str(query_ids[i])
            ids = [q] + [str(g) for g in gallery_ids[i, :5]]
            sheet = Image.new("RGB", (6 * 224, 288), "white")
            draw = ImageDraw.Draw(sheet)
            hits = []
            for j, image_id in enumerate(ids):
                tile = ImageOps.contain(data.load_image(data.image_path(args.root, image_id)), (224, 224))
                sheet.paste(tile, (j * 224 + (224 - tile.width) // 2, (224 - tile.height) // 2))
                title = "query" if j == 0 else f"#{j} score={float(scores[i, j-1]):.3f}"
                draw.text((j * 224 + 5, 230), title, fill="black")
                draw.text((j * 224 + 5, 250), breed[image_id], fill="black")
                if j:
                    hits.append({"rank": j, "image_id": image_id, "breed": breed[image_id],
                                 "score": float(scores[i, j-1]), "same_breed": breed[image_id] == breed[q]})
            path = dest / f"{name}_{kind}.png"
            sheet.save(path)
            descriptions.append({"split": "val", "kind": kind, "query_id": q,
                                 "query_breed": breed[q], "image": path.name, "hits": hits})
        (dest / f"{name}.json").write_text(json.dumps(descriptions, indent=2) + "\n")
        print(f"{name}: {len(chosen)} illustrations, {len(failures)} Hit@5 failures on validation")


if __name__ == "__main__":
    main()
