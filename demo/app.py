"""Run with: streamlit run demo/app.py"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pandas as pd
import streamlit as st

from demo import service
from src import data, models

st.set_page_config(page_title="Visual Similarity Search", page_icon="🔎", layout="wide")
st.markdown(
    """<style>
    .block-container {max-width: 1450px; padding-top: 2rem;}
    [data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: 12px; border-radius: 10px;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def cached_index(model_id, settings):
    return service.load_gallery(model_id, settings)


@st.cache_resource(show_spinner=False)
def cached_encoder(model_id, device):
    return service.load_encoder(model_id, device), threading.Lock()


@st.cache_data(show_spinner=False)
def cached_signatures(settings):
    return service.gallery_signatures(settings)


@st.cache_data(show_spinner=False, ttl=5)
def asset_status(settings):
    return service.missing_assets(settings)


def render_result(result, query_info, settings):
    st.subheader(result["display_name"])
    columns = st.columns(4)
    if result["correct"] is not None:
        columns[0].metric("Precision@5", f"{result['precision_at_5']:.0%}", help="Same-breed results among the five retrieved images.")
    else:
        columns[0].metric("Retrieved images", len(result["hits"]))
    columns[1].metric("Preprocessing", f"{result['preprocessing_ms']:.1f} ms")
    columns[2].metric("Embedding", f"{result['embedding_ms']:.1f} ms")
    columns[3].metric("Search", f"{result['search_ms']:.2f} ms")
    for column, hit in zip(st.columns(5), result["hits"]):
        with column:
            st.image(data.load_image(data.image_path(settings.dataset_root, hit.image_id)), width="stretch")
            correct = query_info["breed"] and hit.breed == query_info["breed"]
            marker = ("✅ " if correct else "❌ ") if query_info["breed"] else ""
            st.markdown(f"**#{hit.rank} · {marker}{hit.breed}**")
            st.caption(f"Cosine similarity {hit.score:.3f}")
    st.caption(f"Live processing: {result['total_ms']:.1f} ms · model loading and page rendering excluded")


def main():
    st.title("Visual Similarity Search")
    st.markdown("Find similar pets with **DINO, DINOv2 and supervised DeiT embeddings**.")
    try:
        settings = service.load_settings()
        cases = service.load_cases(settings)
        missing = asset_status(settings)
    except Exception as exc:
        st.error(f"Demo configuration could not be loaded: {exc}")
        return
    if missing:
        st.warning("Prepare the gallery and example images before starting the demo.")
        st.code('python scripts/prepare_demo.py --embeddings-zip /path/to/second_models_embeddings.zip', language="bash")
        with st.expander("Missing files", expanded=True):
            for item in missing:
                st.write(item)
        return

    with st.sidebar:
        st.header("Search settings")
        mode = st.radio("Mode", ["Single model", "Compare all four"], key="search_mode")
        selected = st.selectbox(
            "Model", settings.model_ids,
            index=settings.model_ids.index(settings.default_model),
            format_func=lambda value: models.get_spec(value).display_name,
            disabled=mode == "Compare all four", key="selected_model",
        )
        st.divider()
        st.caption("Oxford-IIIT Pet · 37 breeds")
        st.caption("Fixed gallery · 1,840 images")
        st.caption("Top-5 · exact cosine search")
        st.caption(f"Current device: {settings.device.upper()}")

    source = st.radio("Query source", ["Prepared examples", "Upload an image"], horizontal=True, key="query_source")
    image_bytes = None
    chosen_case = None
    if source == "Prepared examples":
        case_id = st.selectbox(
            "Prepared example", [case["id"] for case in cases],
            format_func=lambda value: next(c["title"] for c in cases if c["id"] == value),
            key="prepared_case",
        )
        chosen_case = next(case for case in cases if case["id"] == case_id)
        image_bytes = (settings.cases_path.parent / chosen_case["filename"]).read_bytes()
    else:
        uploaded = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png"], key="uploaded_query")
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
        st.caption("Breed labels evaluate known examples; arbitrary uploads are searched without assuming a breed.")

    if image_bytes is None:
        st.info("Upload an image to search the gallery.")
        return
    try:
        query_image = service.decode_image(image_bytes)
    except ValueError as exc:
        st.error(str(exc))
        return
    query_info = service.identify_query(image_bytes, settings, cached_signatures(settings))
    preview, explanation = st.columns([1, 3], gap="large")
    with preview:
        st.image(query_image, caption="Query image", width=240)
    with explanation:
        if chosen_case:
            st.subheader(chosen_case["title"])
            st.write(chosen_case["description"])
        else:
            st.subheader("Your query")
            st.write("The selected encoder turns this image into a vector and searches the same fixed gallery.")
        if query_info["breed"]:
            st.write(f"Known breed: **{query_info['breed']}**")
        if query_info["exclude_ids"]:
            st.caption("The uploaded image is in the gallery. Byte-identical gallery images are excluded.")
        st.caption("Similarity scores are not probabilities and should not be compared directly across encoders.")

    names = settings.model_ids if mode == "Compare all four" else (selected,)
    signature = (hashlib.sha256(image_bytes).hexdigest(), names)
    if st.button("Find similar images", type="primary", key="search_button"):
        st.session_state.pop("search_run", None)
        results = []
        try:
            for name in names:
                with st.spinner(f"Searching with {models.get_spec(name).display_name}…"):
                    index = cached_index(name, settings)
                    encoder, lock = cached_encoder(name, settings.device)
                    with lock:
                        results.append(service.search_image(image_bytes, name, encoder, index, settings, query_info))
        except Exception as exc:
            st.error("Search could not be completed. Run the preparation command to download and verify model weights.")
            with st.expander("Error details"):
                st.code(str(exc))
            return
        st.session_state["search_run"] = {"signature": signature, "results": results}

    current = st.session_state.get("search_run")
    if current and current["signature"] == signature:
        results = current["results"]
        if len(results) > 1:
            st.subheader("Same query · same gallery · four encoders")
            rows = [{
                "Model": r["display_name"],
                "Correct / 5": f"{r['correct']}/5" if r["correct"] is not None else "Unknown breed",
                "Embedding (ms)": round(r["embedding_ms"], 2),
                "Search (ms)": round(r["search_ms"], 3),
            } for r in results]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        for result in results:
            st.divider()
            render_result(result, query_info, settings)

    with st.expander("Full test-set results"):
        metrics_path = settings.results / "metrics_test.csv"
        runtime_path = settings.results / "runtime_summary.csv"
        metrics = pd.read_csv(metrics_path)
        display_metrics = metrics[["model", "precision_at_5", "precision_at_10", "recall_at_5", "recall_at_10", "hit_at_5"]].rename(columns={
            "model": "Model", "precision_at_5": "P@5", "precision_at_10": "P@10",
            "recall_at_5": "R@5", "recall_at_10": "R@10", "hit_at_5": "Hit@5",
        })
        st.dataframe(display_metrics.round(4), hide_index=True, width="stretch")
        runtime = pd.read_csv(runtime_path)
        display_runtime = runtime[["model_id", "embed_median_ms", "search_median_ms", "gpu"]].copy()
        display_runtime["model_id"] = display_runtime["model_id"].map(lambda value: models.get_spec(value).display_name)
        display_runtime.columns = ["Model", "Embedding (ms)", "Search (ms)", "GPU"]
        st.dataframe(display_runtime.round(3), hide_index=True, width="stretch")
        st.caption("Historical benchmark: all 3,669 test queries; latency measured on 200 fixed queries, three repeats, Tesla T4. Live timings above use the current device.")
        st.write("DINOv2 has the highest test Precision@5, with a larger embedding-time cost. Prepared examples illustrate behavior; aggregate conclusions come from the full test set.")
        st.caption("Same-breed relevance is a proxy for visual similarity. Different architectures and pretraining data prevent attributing the result to self-supervision alone.")


if __name__ == "__main__":
    main()
