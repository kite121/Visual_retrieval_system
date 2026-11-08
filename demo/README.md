# Visual Similarity Search Demo

The Streamlit app searches a fixed gallery of 1,840 Oxford-IIIT Pet images using DINO ViT-S/16, DeiT-S/16, DINOv2 ViT-B/14, or DeiT-B/16. It supports uploaded JPG/PNG images and four prepared examples, with single-model and four-model comparison modes.

## Setup

Run commands from the repository root. Python 3.12 is recommended for the locally verified environment; CUDA is optional.

Obtain `second_models_embeddings.zip` from the completed `notebooks/02_second_models_colab.ipynb` run. It is a generated artifact and is not stored in Git.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/prepare_demo.py --embeddings-zip /path/to/second_models_embeddings.zip --verify-models
```

The preparation command verifies the archive's split manifest, installs gallery embeddings, downloads Oxford-IIIT Pet and pretrained weights, copies four test queries, and runs all 16 query/model combinations. It records observed results without comparing them with predefined accuracy targets. It does not train models or regenerate gallery embeddings.

If gallery caches have already been installed, omit `--embeddings-zip`. To prepare images and caches without downloading model weights, use `--skip-model-download`; uploaded-image inference still requires the real weights before presentation.

## Run

```bash
streamlit run demo/app.py
```

Open the local address printed by Streamlit, normally `http://localhost:8501`.

Choose **Example** or **Upload**, select **One model** or **Compare 4 models**, then press **Search**. The mode and model controls are on the main page. The app shows five images, breed labels, similarity scores, and live embedding/search/total timings. Known examples additionally show query-level P@5. For arbitrary uploads, no breed or quality score is assumed.

The single-model selector is hidden in comparison mode. Each image shows its numeric **Cosine Similarity**. Help icons explain **Embedding** and **Search** timings; the dataset title at the top links to the official Oxford page.

In comparison mode, the table summarizes all four models for the current query. Select a model under **View results**, or use **Previous** and **Next**, to browse each Top-5 without rerunning inference. Only one model's images are displayed at a time. Aggregate test-set tables remain in `results/` and are not displayed in the demo.

Model loading is cached and excluded from the displayed timings. Total time includes preprocessing, embedding, and search. The app defaults to offline Hugging Face cache access; run the preparation command before starting it. Live timings use the current CPU/CUDA device. Cosine scores are not calibrated probabilities and should not be compared directly across models.

## Assets and paths

```text
demo/
├── app.py
├── service.py
├── config.yaml
├── README.md
├── demo_script.md              # optional local presentation notes; ignored by Git
├── example_queries/
│   ├── cases.json
│   ├── Abyssinian_224.jpg
│   ├── Abyssinian_21.jpg
│   ├── Siamese_228.jpg
│   └── leonberger_73.jpg
├── embeddings/                 # local; ignored by Git
│   └── <model_id>/
│       ├── gallery.npz
│       └── gallery.json
└── .cache/                     # local; ignored by Git
    ├── torch/
    ├── huggingface/
    └── verification.json
```

`gallery.npz` contains L2-normalized vectors, image IDs, and labels, not images or encoder weights. The app uses raw gallery images under `data/oxford-iiit-pet/images/`. Validation/test embeddings are not needed by the live app.

The default model caches are inside `demo/.cache/`. Existing `TORCH_HOME` or `HF_HOME` settings take precedence. Dataset images, gallery caches, and model weights are not committed. The four small query JPG files and their descriptions are versioned.

`demo/config.yaml` controls paths, the default model, CPU thread count, and automatic CPU/CUDA selection. The existing shared code in `src/` supplies preprocessing, encoder loading, normalized embeddings, and exact cosine search.

## Presentation preparation

Download assets on the actual presentation machine in advance. Run the preparation command with `--verify-models`, start the app, and use **Compare 4 models** once to warm the app's model caches. The regular search then needs no network access as long as the installed model libraries and cached weights remain available.

Prepared queries belong to the test split and are disjoint from the gallery. Byte-identical gallery images are excluded if a gallery image is uploaded. Recompression, crops, and near-duplicates are not detected as exact duplicates.

The `--verify-models` option checks that each encoder can complete retrieval for the prepared queries and records the observed outputs. The case descriptions contain no expected scores, and the command does not enforce fixed retrieval results or overwrite the recorded experiment metrics.

## Local verification

Verified on macOS with CPU inference, Python 3.12.3, PyTorch 2.15.0, torchvision 0.29.2, timm 1.0.31, and Streamlit 1.67.0. All 16 prepared query/model combinations completed live retrieval. Streamlit application checks covered all model selections, all four comparison cases, a JPEG upload, a previously unseen cropped PNG, exclusion of an uploaded gallery image, and an invalid-image error. Those application checks also passed with network connections disabled. The running browser interface was checked separately.
