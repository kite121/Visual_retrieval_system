# Experiment Results

These files contain the recorded outputs of the experiment notebooks and their final analysis. All models use frozen pretrained embeddings, the same Oxford-IIIT Pet split (`configs/split_manifest.csv`, seed 42), a gallery of 1,840 images, 1,840 validation queries, and 3,669 test queries. An image is relevant when its breed matches the query breed. The quality and runtime tables in notebooks 01 and 02 come from Tesla T4 runs on 7 November 2026; the separate origin of the diagnostic rankings is described below.

```text
results/
├── README.md
├── 01_first_models/
│   ├── metrics_validation.csv
│   ├── metrics_test.csv
│   ├── runtime_summary.csv
│   └── run_metadata.json
├── 02_second_models/
│   ├── metrics_validation.csv
│   ├── metrics_test.csv
│   ├── runtime_summary.csv
│   └── run_metadata.json
├── 03_final_analysis/
│   ├── final_quality_metrics.csv
│   ├── runtime_summary.csv
│   ├── failure_cases.csv
│   ├── run_metadata.json
│   └── figures/                     # two comparison charts and three failure cases
└── final_analysis_evidence/
    ├── selection.json
    ├── dataset_verification.json
    ├── artifacts/runs/<model>/      # saved test rankings and their provenance
    └── data/oxford-iiit-pet/images/  # only images needed for the three failure cases
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

## 03_final_analysis

Source: [notebook 03](../notebooks/03_final_analysis.ipynb), using `scripts/build_final_results.py`, `scripts/generate_final_figures.py`, and `scripts/export_failure_cases.py`. It reuses the four-model results from `02_second_models` without retraining or repeating inference.

| File | Contents |
| --- | --- |
| `final_quality_metrics.csv` | Eight rows: four models × validation/test, with Precision@5/10, Recall@5/10, and Hit@5. |
| `runtime_summary.csv` | The four-model repeated timing summary copied from `02_second_models`. |
| `failure_cases.csv` | Three diagnostic queries: model, breed, P@5, Hit@5, observation, explanation, figure path, and the five retrieved neighbors with cosine scores. |
| `run_metadata.json` | Analysis date and source commit, input hashes, original experiment metadata, selection protocol, and ranking validation results. |
| `figures/precision_at_5_comparison.png` | Test Precision@5 for all four models. |
| `figures/runtime_comparison.png` | Gallery build time and separate preprocessing, embedding, and search timings with min/max across repeats. |
| `figures/failure_case_01.png`–`failure_case_03.png` | Query + Top-5, breed labels, image IDs, and cosine scores for each failure case. |

## final_analysis_evidence

These are inputs for reproducing the diagnostic figures on CPU, not another set of final metrics. The rankings come from a separate pretrained evaluation on 31 October 2026 (GTX 1060 3GB, batch size 8). Their five test metrics are recomputed for each model and checked against the current quality tables within `1e-6`. Matching aggregate metrics does not establish identical rankings between runs. Final timings use only the canonical Tesla T4 run.

| File or path | Contents |
| --- | --- |
| `selection.json` | Three fixed diagnostic cases, selection rationale, observations, and explanatory hypotheses. |
| `dataset_verification.json` | The original dataset verification record and manifest hash. |
| `artifacts/runs/<model>/evaluation.json` | Model, manifest, and environment metadata for each historical ranking run. |
| `artifacts/runs/<model>/topk_test.npz` | Query IDs and Top-10 gallery IDs with cosine scores for all 3,669 test queries. |
| `data/oxford-iiit-pet/images/` | Eighteen original dataset images needed to render the selected queries and neighbors. |

Recorded metadata may reference historical paths such as `results/stage3/`; these identify original sources retained in Git history. Gallery embeddings and pretrained weights are local demo assets, not stored here.
