import io
import os
import traceback
from typing import List, Dict, Any, Optional
import streamlit as st

from utils.preprocessing import read_text_from_upload, clean_text
from utils.inference import get_supported_models, load_ner_pipeline, run_ner_on_text, tokenize_text
from utils.visualization import (
    render_entities_html,
    entities_to_dataframe,
    token_importance_bar,
)

print("✅ Streamlit app starting...")

st.set_page_config(page_title="NER App", page_icon="🔎", layout="wide")

st.title(" Named Entity Recognition (NER) - Multi-Model")
st.caption("Upload text/PDF, choose one or more models (BERT / RoBERTa / XLM-RoBERTa / custom HF id), compare entities side-by-side, visualize token importance, and export results.")

# ---------------- Sidebar ----------------
st.sidebar.header("Models")
SUPPORTED = get_supported_models()
print(f"✅ Supported models: {list(SUPPORTED.keys())}")

model_labels = st.sidebar.multiselect(
    "Choose models for comparison",
    list(SUPPORTED.keys()),
    default=["BERT (dslim/bert-base-NER)"],
    help="Select multiple to compare side-by-side."
)

custom_model_id = st.sidebar.text_input(
    "Or add a custom Hugging Face model id (token-classification)",
    value="",
    help="Example: dslim/bert-base-NER (press Enter after typing to use)"
)
if custom_model_id.strip():
    SUPPORTED[f"Custom ({custom_model_id.strip()})"] = custom_model_id.strip()
    if f"Custom ({custom_model_id.strip()})" not in model_labels:
        model_labels.append(f"Custom ({custom_model_id.strip()})")

aggregation = st.sidebar.selectbox(
    "Aggregation strategy",
    ["simple", "max", "first", "average"],
    index=0,
    help="How subword tokens are grouped into entity spans."
)

device_map = st.sidebar.selectbox(
    "Device",
    ["auto", "cpu"],
    index=0,
    help="Set 'auto' to use GPU if available, else CPU."
)

max_chunk_chars = st.sidebar.slider(
    "Max chunk size (characters)",
    min_value=500, max_value=4000, value=1500, step=100,
    help="Long texts are split into chunks to avoid model max length limits."
)

st.sidebar.header("Input")
uploaded = st.sidebar.file_uploader("Upload a .txt or .pdf", type=["txt", "pdf"])
pasted_text = st.sidebar.text_area("…or paste text here", height=160, placeholder="Paste any article here…")

show_heatmap = st.sidebar.checkbox("Show token importance heatmap (per model)", value=True,
                                   help="Proxy heatmap using entity scores over tokens.")
run_btn = st.sidebar.button("Run NER", type="primary")

print("✅ Sidebar ready.")

# ---------------- Input handling ----------------
def get_input_text() -> Optional[str]:
    if uploaded is not None:
        print(f"📂 Uploaded file detected: {uploaded.name}")
        txt = read_text_from_upload(uploaded)
        if txt:
            print(f"📄 File read OK ({len(txt)} chars)")
            return txt
    if pasted_text.strip():
        print(f"📝 Pasted text detected ({len(pasted_text)} chars)")
        return pasted_text
    print("⚠️ No input text found.")
    return None

# ---------------- Main UI ----------------
col_left, col_right = st.columns([0.55, 0.45], gap="large")

with col_left:
    st.subheader("1) Input")
    text_in = get_input_text()
    if text_in:
        st.text_area("Detected text (first 4000 chars)", value=text_in[:4000], height=220)
    else:
        st.info("Upload a .txt/.pdf or paste text in the sidebar.")

with col_right:
    st.subheader("2) Models & Settings")
    st.write("**Selected models:**")
    if model_labels:
        for lbl in model_labels:
            st.write(f"- `{lbl}` → `{SUPPORTED[lbl]}`")
    else:
        st.warning("Select at least one model in the sidebar.")
    st.write(f"Aggregation: `{aggregation}` · Device: `{device_map}` · Chunk: `{max_chunk_chars}`")

# ---------------- Run ----------------
if run_btn:
    print("🚀 Run clicked.")
    if not text_in or len(text_in.strip()) == 0:
        st.error("Please provide some text via upload or paste.")
        st.stop()
    if not model_labels:
        st.error("Please select at least one model to run.")
        st.stop()

    with st.spinner("Loading models and running NER…"):
        try:
            clean = clean_text(text_in)
            print(f"🧹 Cleaned text length: {len(clean)}")

            comparison_results: Dict[str, List[dict]] = {}
            token_views: Dict[str, Dict[str, Any]] = {}

            for lbl in model_labels:
                model_id = SUPPORTED[lbl]
                print(f"🧠 Loading pipeline: {lbl} [{model_id}]")
                pipe = load_ner_pipeline(
                    model_id=model_id,
                    aggregation_strategy=aggregation,
                    device_map=device_map,
                )
                print(f"🔍 Running NER for: {lbl}")
                ents = run_ner_on_text(
                    pipe=pipe,
                    text=clean,
                    max_chunk_chars=max_chunk_chars,
                    allow_nested=True,   # keep overlapping spans
                )
                print(f"✅ {lbl}: {len(ents)} entities")
                comparison_results[lbl] = ents

                if show_heatmap:
                    toks = tokenize_text(model_id, clean)
                    token_views[lbl] = {
                        "tokens": toks["tokens"],
                        "offsets": toks["offsets"],
                        "scores": toks["scores_from_entities"](ents),  # proxy importance
                    }

        except Exception as e:
            print(f"❌ ERROR: {e}")
            traceback.print_exc()
            st.error(f"NER failed: {e}")
            st.code(traceback.format_exc())
            st.stop()

    st.success("Done!")
    print("📊 Rendering comparison tabs...")

    # --------- Comparison Tabs ---------
    tabs = st.tabs(model_labels)
    combined_rows = []
    for i, lbl in enumerate(model_labels):
        with tabs[i]:
            st.markdown(f"### {lbl}")
            df = entities_to_dataframe(comparison_results[lbl])
            st.dataframe(df, use_container_width=True)

            # CSV download for this model
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download {lbl} results (CSV)",
                data=csv_bytes,
                file_name=f"ner_results_{lbl.replace(' ', '_').replace('/', '_')}.csv",
                mime="text/csv",
            )

            st.markdown("#### Highlighted text")
            html = render_entities_html(clean, comparison_results[lbl])
            st.markdown(html, unsafe_allow_html=True)

            if show_heatmap and lbl in token_views:
                st.markdown("#### Token importance (proxy heatmap)")
                fig = token_importance_bar(
                    tokens=token_views[lbl]["tokens"],
                    scores=token_views[lbl]["scores"],
                    max_tokens=128,  # prevent super-wide plots
                )
                st.plotly_chart(fig, use_container_width=True)

            # Collect combined rows
            if not df.empty:
                tmp = df.copy()
                tmp.insert(0, "model", lbl)
                combined_rows.append(tmp)

    # Combined CSV for all models
    if combined_rows:
        import pandas as pd
        combined = pd.concat(combined_rows, ignore_index=True)
        st.markdown("### Download combined results")
        st.dataframe(combined, use_container_width=True)
        st.download_button(
            label=" Download ALL models (CSV)",
            data=combined.to_csv(index=False).encode("utf-8"),
            file_name="ner_results_all_models.csv",
            mime="text/csv",
        )

print("✅ App ready.")
