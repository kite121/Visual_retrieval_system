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

st.set_page_config(page_title="Image Search", page_icon="🔎", layout="wide")
st.markdown(
    """<style>
    .block-container {max-width: 1200px; padding-top: 1.5rem;}
    h1 {font-size: 2.2rem !important;}
    [data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: 12px; border-radius: 10px;}
    [data-testid="stMetricValue"] {font-size: 1.6rem;}
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
        color: inherit !important; opacity: 1 !important;
    }
    [data-testid="stWidgetLabel"] p, [data-testid="stMetricLabel"] p {font-weight: 600;}
    button:disabled {color: inherit !important; opacity: 1 !important; cursor: not-allowed;}
    .st-key-query_preview img {width: 100%; max-height: 220px; object-fit: contain; border-radius: 10px;}
    .st-key-result_grid img {width: 100%; height: 150px; object-fit: contain; border-radius: 8px;}
    .cosine-score {display: flex; align-items: baseline; gap: .35rem; flex-wrap: wrap; font-size: .75rem; line-height: 1.6;}
    .cosine-score span, .cosine-score strong {white-space: nowrap;}
    </style>""",
    unsafe_allow_html=True,
)

MODEL_LABELS = {
    "dino_vit_s16": "DINO-S",
    "deit_vit_s16_supervised": "DeiT-S",
    "dinov2_vit_b14": "DINOv2-B",
    "deit_vit_b16_supervised": "DeiT-B",
}
CASE_HINTS = {
    "successful_search": "A clear match for every model.",
    "model_comparison": "Compare how the encoders see the same image.",
    "shared_failure": "A difficult example with visually similar breeds.",
    "larger_is_not_always_better": "A larger model can still make mistakes.",
}


def move_result(direction, names):
    current = st.session_state.get("result_model", names[0])
    position = names.index(current)
    st.session_state["result_model"] = names[position + direction]


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
    st.subheader(f"{MODEL_LABELS[result['model_id']]} · Top 5")
    columns = st.columns(4)
    if result["correct"] is not None:
        columns[0].metric("P@5", f"{result['precision_at_5']:.0%}", help="Fraction of Top-5 results with the same breed as the query.")
    else:
        columns[0].metric("Matches", len(result["hits"]))
    columns[1].metric("Embedding", f"{result['embedding_ms']:.1f} ms", help="Time to convert the query image into a normalized feature vector. Excludes preprocessing and model loading.")
    columns[2].metric("Search", f"{result['search_ms']:.2f} ms", help="Time to compare the query vector with 1,840 gallery vectors and select the Top 5.")
    columns[3].metric("Total", f"{result['total_ms']:.1f} ms", help="Preprocessing + embedding + search. Model loading is excluded.")
    with st.container(key="result_grid"):
        for column, hit in zip(st.columns(5), result["hits"]):
            with column:
                st.image(data.load_image(data.image_path(settings.dataset_root, hit.image_id)), width="stretch")
                correct = query_info["breed"] and hit.breed == query_info["breed"]
                marker = ("✅ " if correct else "❌ ") if query_info["breed"] else ""
                st.markdown(f"**#{hit.rank} {marker}{hit.breed}**")
                st.markdown(f'<div class="cosine-score"><span>Cosine Similarity</span><strong>{hit.score:.3f}</strong></div>', unsafe_allow_html=True)


def main():
    st.title("Image Search")
    st.markdown("**Oxford-IIIT Pet** · [Dataset website ↗](https://www.robots.ox.ac.uk/~vgg/data/pets/)")
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

    controls, model_control = st.columns([3, 2], gap="large")
    with controls:
        mode = st.radio("Search mode", ["One model", "Compare 4 models"], horizontal=True, key="search_mode")
    selected = st.session_state.get("preferred_model", settings.default_model)
    if mode == "One model":
        with model_control:
            selected = st.selectbox(
                "Model", settings.model_ids,
                index=settings.model_ids.index(selected),
                format_func=MODEL_LABELS.get, key="selected_model",
            )
        st.session_state["preferred_model"] = selected

    source = st.radio("Image", ["Example", "Upload"], horizontal=True, key="query_source")
    image_bytes = None
    chosen_case = None
    if source == "Example":
        case_id = st.selectbox(
            "Example", [case["id"] for case in cases],
            format_func=lambda value: next(c["title"] for c in cases if c["id"] == value),
            key="prepared_case",
        )
        chosen_case = next(case for case in cases if case["id"] == case_id)
        image_bytes = (settings.cases_path.parent / chosen_case["filename"]).read_bytes()
    else:
        uploaded = st.file_uploader("Choose a JPG or PNG", type=["jpg", "jpeg", "png"], key="uploaded_query")
        if uploaded is not None:
            image_bytes = uploaded.getvalue()

    if image_bytes is None:
        st.info("Choose an image to start.")
        return
    try:
        query_image = service.decode_image(image_bytes)
    except ValueError as exc:
        st.error(str(exc))
        return
    query_info = service.identify_query(image_bytes, settings, cached_signatures(settings))
    preview, explanation = st.columns([2, 3], gap="large", vertical_alignment="center")
    with preview:
        with st.container(key="query_preview"):
            st.image(query_image, width="stretch")
    with explanation:
        if chosen_case:
            st.subheader(chosen_case["title"])
            st.write(CASE_HINTS[chosen_case["id"]])
        else:
            st.subheader("Your image")
        if query_info["breed"]:
            st.caption(f"Breed: {query_info['breed']}")
        if query_info["exclude_ids"]:
            st.caption("The query itself is excluded from results.")

    names = settings.model_ids if mode == "Compare 4 models" else (selected,)
    signature = (hashlib.sha256(image_bytes).hexdigest(), names)
    if st.button("Search", type="primary", key="search_button"):
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
            st.subheader("Comparison")
            rows = [{
                "Model": MODEL_LABELS[r["model_id"]],
                "Correct / 5": f"{r['correct']}/5" if r["correct"] is not None else "—",
                "Embedding (ms)": round(r["embedding_ms"], 2),
                "Search (ms)": round(r["search_ms"], 3),
            } for r in results]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            if st.session_state.get("result_model") not in names:
                st.session_state["result_model"] = names[0]
            active = st.radio("View results", names, format_func=MODEL_LABELS.get, horizontal=True, key="result_model")
            position = names.index(active)
            previous, counter, following = st.columns([1, 2, 1])
            previous.button("← Previous", disabled=position == 0, on_click=move_result, args=(-1, names), key="previous_result")
            counter.caption(f"Model {position + 1} of {len(names)}")
            following.button("Next →", disabled=position == len(names) - 1, on_click=move_result, args=(1, names), key="next_result")
            render_result(next(result for result in results if result["model_id"] == active), query_info, settings)
        else:
            st.divider()
            render_result(results[0], query_info, settings)


if __name__ == "__main__":
    main()
