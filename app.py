import io
import os
import traceback
from typing import List, Dict, Any, Optional
import streamlit as st

from utils.preprocessing import read_text_from_upload, clean_text
from utils.inference import get_supported_models, load_ner_pipeline, run_ner_on_text
from utils.visualization import render_entities_html, entities_to_dataframe

print("✅ Streamlit app starting...")

st.set_page_config(page_title="NER App", page_icon="🔎", layout="wide")

st.title("Named Entity Recognition (NER) – Multi-Model")
st.caption("Upload an article (TXT/PDF) or paste text, choose a model, and get entities with highlights.")

print("📦 Sidebar setup starting...")

# ---------------- Sidebar ----------------
st.sidebar.header("Model")
SUPPORTED = get_supported_models()
print(f"✅ Supported models loaded: {list(SUPPORTED.keys())}")

model_label = st.sidebar.selectbox(
    "Choose a model",
    list(SUPPORTED.keys()),
    index=0
)
default_model_id = SUPPORTED[model_label]

custom_model_id = st.sidebar.text_input(
    "Or use a custom Hugging Face model id (token-classification)",
    value="",
    help="Example: dslim/bert-base-NER"
)
selected_model_id = custom_model_id.strip() or default_model_id

aggregation = st.sidebar.selectbox(
    "Aggregation strategy",
    ["simple", "max", "first", "average"],
    index=0,
)

device_map = st.sidebar.selectbox(
    "Device",
    ["auto", "cpu"],
    index=0,
)

max_chunk_chars = st.sidebar.slider(
    "Max chunk size (characters)",
    min_value=500, max_value=4000, value=1500, step=100,
)

st.sidebar.header("Input")
uploaded = st.sidebar.file_uploader("Upload a .txt or .pdf", type=["txt", "pdf"])
pasted_text = st.sidebar.text_area("…or paste text here", height=160, placeholder="Paste any article here…")

run_btn = st.sidebar.button("Run NER", type="primary")

print("✅ Sidebar setup complete.")

# ---------------- Input handling ----------------
def get_input_text() -> Optional[str]:
    if uploaded is not None:
        print(f"📂 Uploaded file detected: {uploaded.name}")
        txt = read_text_from_upload(uploaded)
        if txt:
            print(f"📄 File read successfully ({len(txt)} chars).")
            return txt
    if pasted_text.strip():
        print(f"📝 Pasted text detected ({len(pasted_text)} chars).")
        return pasted_text
    print("⚠️ No input text detected.")
    return None


# ---------------- Main area ----------------
col_left, col_right = st.columns([0.55, 0.45], gap="large")

with col_left:
    st.subheader("1) Input")
    text_in = get_input_text()
    if text_in:
        st.text_area("Detected text", value=text_in[:4000], height=220)
    else:
        st.info("Upload a .txt/.pdf or paste text in the sidebar.")

with col_right:
    st.subheader("2) Model")
    st.write(f"**Selected model id:** `{selected_model_id}`")
    st.write(f"Aggregation: `{aggregation}` · Device: `{device_map}`")

# ---------------- Run ----------------
if run_btn:
    print("🚀 Run button clicked.")
    if not text_in or len(text_in.strip()) == 0:
        st.error("Please provide some text via upload or paste.")
        st.stop()

    with st.spinner("Loading model and running NER…"):
        try:
            print(f"🧠 Loading model: {selected_model_id}")
            pipe = load_ner_pipeline(
                model_id=selected_model_id,
                aggregation_strategy=aggregation,
                device_map=device_map
            )
            print("✅ Model loaded successfully. Cleaning text...")
            clean = clean_text(text_in)
            print(f"🧹 Cleaned text length: {len(clean)}")
            print("🔍 Running NER pipeline...")
            results = run_ner_on_text(pipe=pipe, text=clean, max_chunk_chars=max_chunk_chars)
            print(f"✅ NER completed. Entities found: {len(results)}")

        except Exception as e:
            print(f"❌ ERROR: {e}")
            traceback.print_exc()
            st.error(f"NER failed: {e}")
            st.code(traceback.format_exc())
            st.stop()

    st.success("Done!")
    print("📊 Rendering results...")

    st.subheader("Entities (table)")
    df = entities_to_dataframe(results)
    st.dataframe(df, use_container_width=True)
    print("✅ Entity table rendered.")

    st.subheader("Entities (highlighted)")
    html = render_entities_html(clean, results)
    st.markdown(html, unsafe_allow_html=True)
    print("✅ Entity highlighting rendered.")

    with st.expander("Raw results (debug)", expanded=False):
        st.json(results)
        print("🧾 Raw JSON results displayed.")

print("✅ App fully loaded and waiting for user input.")
