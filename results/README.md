# Experiment Results

These files contain the recorded outputs of the two executed experiment notebooks. All models use frozen pretrained embeddings, the same Oxford-IIIT Pet split (`configs/split_manifest.csv`, seed 42), a gallery of 1,840 images, 1,840 validation queries, and 3,669 test queries. An image is relevant when its breed matches the query breed. The runs were performed on a Tesla T4 on 7 November 2026.

```text
results/
├── README.md
├── 01_first_models/
│   ├── metrics_validation.csv
│   ├── metrics_test.csv
│   ├── runtime_summary.csv
│   └── run_metadata.json
└── 02_second_models/
    ├── metrics_validation.csv
    ├── metrics_test.csv
    ├── runtime_summary.csv
    └── run_metadata.json
```

## 01_first_models

Source: [notebook 01](../notebooks/01_first_models_colab.ipynb), using `scripts/run_evaluation.py`. Contains the first baseline comparison: **DINO ViT-S/16** and supervised **DeiT-S/16**.

| File | Contents |
| --- | --- |
| `metrics_validation.csv` | Two model rows: validation Precision@5/10, Recall@5/10, and Hit@5. |
| `metrics_test.csv` | The same five metrics for the two models on the official test queries. |
| `runtime_summary.csv` | One timing run per model: gallery build time in seconds; preprocessing, embedding, and search median/mean/p95 in milliseconds over 200 queries; hardware, batch size, and date. |
| `run_metadata.json` | Run configuration, split information, model settings, environment versions, hardware, and provenance needed to interpret the results. |

## 02_second_models

Source: [notebook 02](../notebooks/02_second_models_colab.ipynb), using `scripts/run_evaluation.py` and `scripts/benchmark_timing.py`. Contains **DINO ViT-S/16**, supervised **DeiT-S/16**, **DINOv2 ViT-B/14**, and supervised **DeiT-B/16**, evaluated together under the same protocol. Use this folder for the four-model comparison.

| File | Contents |
| --- | --- |
| `metrics_validation.csv` | Four model rows: validation Precision@5/10, Recall@5/10, and Hit@5. |
| `metrics_test.csv` | The same five metrics for all four models on the official test queries. |
| `runtime_summary.csv` | Three timing repeats per model on the same 200 queries. Contains median/min/max gallery build time in seconds and median/min/max of the per-repeat preprocessing, embedding, and search medians in milliseconds; also hardware, batch size, and worker count. |
| `run_metadata.json` | Source commit, manifest hash, configuration, environment, timing query IDs, repeat settings, and the recorded comparison with previous results. |

Quality values are fractions in `[0, 1]`; Hit@5 is the fraction of queries with at least one relevant result in the Top-5. Runtime formats differ between the two runs: the min/max values in `02_second_models` describe variation between repeats, while p95 in `01_first_models` describes variation between individual query timings. For time comparisons across all four models, use `02_second_models/runtime_summary.csv`.

Recorded metadata may reference historical paths such as `results/stage3/`; these identify the original comparison sources retained in Git history. The current result tables are in the two folders above. Gallery embeddings and pretrained weights are local demo assets, not stored here. Final analysis figures and failure cases have not yet been added to this folder.
