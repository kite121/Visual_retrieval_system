# Visual Retrieval System

Image search on the [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) dataset using pretrained image embeddings. The project compares self-supervised DINO and DINOv2 with supervised DeiT models under one retrieval protocol.

## Project question

How do retrieval quality and runtime of pretrained DINO and DINOv2 embeddings compare with supervised embeddings on Oxford-IIIT Pet?

## Agreed experimental protocol

- Use all 37 breed classes and the official `trainval` and `test` partitions.
- Split `trainval` 50/50 **within each breed** with seed `42`. Use one half as the fixed gallery and the other as validation queries. Use the official `test` partition only for final queries.
- A gallery image is relevant when its breed label matches the query breed.
- Compare DINO ViT-S/16, supervised DeiT-S/16, DINOv2 ViT-B/14, and supervised DeiT-B/16. Use pretrained weights without fine-tuning in the initial comparison.
- Report Precision@5/10, Recall@5/10, Hit@5, gallery embedding time, and per-query embedding and search time. Record hardware and batch size alongside timing results.
- The demo accepts a query image, lets the user select a model, and shows the five nearest gallery images.

Breed agreement is an operational proxy for visual similarity; it does not capture every human judgment about appearance.

## Repository layout

| Path | Purpose |
| --- | --- |
| `configs/default.yaml` | Shared split, models, and evaluation settings |
| `src/visual_retrieval/data.py` | Dataset loading and deterministic split |
| `src/visual_retrieval/models/` | Pretrained model adapters |
| `src/visual_retrieval/embeddings.py` | Embedding extraction and normalization |
| `src/visual_retrieval/retrieval.py` | Gallery indexing and nearest-neighbor search |
| `src/visual_retrieval/metrics.py` | Retrieval metrics |
| `src/visual_retrieval/timing.py` | Runtime measurements |
| `scripts/` | Data preparation and evaluation entry points |
| `demo/app.py` | Query-by-image demo |
| `results/` | Final tables, figures, and failure examples |
| `docs/` | Project plan and documentation |

## Team and deadlines

| Date (2026) | Owner | Deliverable |
| --- | --- | --- |
| 6 October | Danil | Project plan and shared experiment protocol |
| 13 October | Azamat | Dataset pipeline, gallery, DINO ViT-S/16 and supervised DeiT-S/16 baselines |
| 20 October | Karim | DINOv2 ViT-B/14 and supervised DeiT-B/16 using the same gallery and metrics |
| 27 October | Aslan | Preliminary comparison and remaining experiment list |
| 3 November | Aslan | Final results, runtimes, figures, and at least three failure examples |
| 10 November | Danil | Technical summary, reproducible code, README, slides, demo, and contributions |

## Getting started

Create a Python environment and install the dependencies in `requirements.txt`. The repository currently contains the agreed **structure and protocol**; model adapters, data preparation, evaluation, and demo are tasks for the upcoming sprints. Downloaded images, model weights, and generated caches belong in `data/` or `artifacts/` and are excluded from Git.

See [`docs/project_plan.pdf`](docs/project_plan.pdf) for the one-page team plan.
