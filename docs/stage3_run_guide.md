# Stage 3 run guide

Karim adds DINOv2 ViT-B/14 and supervised DeiT-B/16 to the shared retrieval
pipeline. Run frozen pretrained backbones; do not fit a classifier or fine-tune.

## Files to open

- `notebooks/02_second_models_colab.ipynb`: complete GPU experiment.
- `Visual_retrieval_system_stage3.zip`: source snapshot packaged locally, including
  the new adapters. Upstream GitHub main does not include these local changes yet.
- `scripts/package_stage3.py`: rebuild the source ZIP after any code change.

## Running in Colab

1. Upload the notebook at https://colab.research.google.com/.
2. Select **Runtime > Change runtime type > T4 GPU** (another CUDA GPU is also
   supported; its identity will be recorded).
3. Upload `Visual_retrieval_system_stage3.zip` through the **Files** panel into
   `/content`, then choose **Run all**. No Drive mount is required.
4. Keep the default four-model list for a fresh comparison on one machine.
5. Wait until the completeness checks and ZIP validation pass. Results are saved
   after each model under `/content/icv_stage3_results/<run id>/`. Download the results ZIP
   and the **executed** notebook via **File > Download > Download .ipynb**.

The notebook uses an isolated environment and keeps Colab's matching CUDA
PyTorch/Torchvision pair. There is no need to replace CUDA packages or install
xFormers for the official DINOv2 inference backbone. Dependencies are recorded
in `evidence/environment.txt`, hardware in `evidence/hardware.json`.

## Experimental protocol

- Oxford-IIIT Pet: all 37 breeds; the committed manifest has 7,349 unique IDs.
- Gallery: 1,840 images; validation queries: 1,840; test queries: 3,669.
- Manifest SHA-256:
  `82f40e25888189aba46d1c1ff2c74894284c1ff7fa50362026c8abbea0ec9f80`.
- RGB, shorter side resized to 256 with bicubic interpolation, center crop 224,
  ImageNet normalization; no random augmentation.
- DINOv2: Meta's original `dinov2_vitb14` LVD-142M backbone without registers,
  pinned Hub revision recorded by the adapter; its final normalized CLS token
  is a 768-dimensional feature. Keep the original pretrained position grid;
  the official forward pass interpolates it for 224px inputs.
- DeiT-B: `deit_base_patch16_224.fb_in1k`, non-distilled supervised ImageNet-1k
  weights, classifier removed, 768-dimensional CLS feature.
- Shared L2 normalization and exact cosine top-K; breed labels used only to
  evaluate relevance. Report P@5/10, R@5/10 and Hit@5 averaged over all queries.

Stage 2 already reported test results. Preserve settings chosen from the plan
or validation; do not claim the test has never been inspected. This experiment
compares available backbones, not an isolated causal effect of supervision.

## Runtime measurements

The quality run forces a fresh gallery build with `--rebuild`. Three additional
timing repeats also rebuild the complete gallery and rotate model order.
Every model uses the same 200 test-query IDs at batch size 1 for latency, with
warm-up and CUDA synchronization. The sample IDs are saved and checked.

- Gallery build: warm file cache, decoding, preprocessing, forward passes,
  L2 normalization and transfer to a CPU array. Saving to disk is excluded.
- Query embedding: host-to-GPU copy, forward and L2 normalization.
- Search: exact cosine top-5 and returning indices to the CPU.
- Query preprocessing: decoding and transforms, measured separately.

`timing_repeats/summary.csv` reports median and min/max of the per-repeat
measurements. A small latency difference alone is not evidence of a stable
speed advantage. Use the fresh four-model results, keeping historical Stage 2
timing separate.

## Evidence for the short report

- `quality/<model>.json` and `quality/metrics.csv`: quality on both complete query sets.
- `quality/timing.csv`: timings from the full quality run.
- `timing_repeats/<model>_repeat<N>.json` and `summary.csv`: repeated fresh timings.
- `artifacts/embeddings/<model>/`: embeddings, labels, image IDs and metadata.
- `artifacts/runs/<model>/topk_val.npz` and `topk_test.npz`: complete top-10 rankings.
- `examples/`: fixed validation query contact sheets and first Hit@5 failures,
  if present; JSON contains query/hit IDs, scores and breed agreement.
- `logs/`, `evidence/`, `source_snapshot.zip`, `bundle_manifest.json`,
  `run_settings.json`, `run_status.json`, `file_hashes.json` and ZIP checksum.

The notebook checks all expected model files, exact query/image counts,
normalized finite embeddings, rank shapes, manifest identity and timing sample
identity before marking evaluation complete. Download the executed notebook
separately to preserve its cell outputs. The prepared blank notebook and local
synthetic smoke results are not a full Oxford-Pet experiment.

## Local commands

Run from the repository root after installing PyTorch/Torchvision and
`requirements-stage3.txt`:

```bash
python -m pytest tests -q
python scripts/smoke_models.py
python scripts/prepare_data.py --check
python scripts/evaluate.py --models dinov2_vit_b14 deit_vit_b16_supervised \
  --device cuda --artifacts artifacts/stage3 --results-dir results/stage3/quality --rebuild
python scripts/benchmark_timing.py --device cuda --repeats 3
python scripts/package_stage3.py
```

`smoke_models.py` verifies both new real pretrained encoders on synthetic RGB
images, including `(N, 768)`, finite unit-normalized features, frozen parameters,
top-5 self retrieval and self exclusion. `--random-weights` is an offline
interface check and does not produce reportable model-quality results.

If the GPU runs out of memory, reduce `BATCH_SIZE` for **all** models and start
a fresh run. Keep the fixed manifest and preprocessing. Download completed
results before ending the VM; files in its session storage are temporary. The
notebook starts a new named run on re-execution, so incomplete and complete runs
stay separate.
