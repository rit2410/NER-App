from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def get_supported_models() -> Dict[str, str]:
    print("🔧 Loading supported model list...")
    return {
        "BERT (dslim/bert-base-NER)": "dslim/bert-base-NER",
        "RoBERTa (Jean-Baptiste/roberta-large-ner-english)": "Jean-Baptiste/roberta-large-ner-english",
        "XLM-RoBERTa (xlm-roberta-large-finetuned-conll03-english)": "xlm-roberta-large-finetuned-conll03-english",
    }

_PIPELINE_CACHE: Dict[str, Any] = {}

def load_ner_pipeline(model_id: str, aggregation_strategy: str = "simple", device_map: str = "auto"):
    print(f"🧠 Loading model: {model_id}, aggregation={aggregation_strategy}, device={device_map}")
    cache_key = f"{model_id}|{aggregation_strategy}|{device_map}"
    if cache_key in _PIPELINE_CACHE:
        print("⚡ Using cached pipeline.")
        return _PIPELINE_CACHE[cache_key]

    print("⬇️ Downloading model and tokenizer from Hugging Face...")
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForTokenClassification.from_pretrained(model_id)
    print("✅ Model and tokenizer loaded.")

    print("⚙️ Building pipeline...")
    pipe = pipeline(
        task="token-classification",
        model=mdl,
        tokenizer=tok,
        aggregation_strategy=aggregation_strategy,
        device_map=(None if device_map == "cpu" else device_map),
    )

    _PIPELINE_CACHE[cache_key] = pipe
    print("✅ Pipeline ready and cached.")
    return pipe


def _chunk_text(text: str, max_chars: int) -> List[str]:
    print(f"✂️ Splitting text into chunks (max {max_chars} chars per chunk)...")
    if len(text) <= max_chars:
        print("✅ Text fits in one chunk.")
        return [text]

    chunks = []
    paragraph_splits = text.split("\n\n")
    buf, cur_len = [], 0
    for para in paragraph_splits:
        add = (para + "\n\n")
        if cur_len + len(add) > max_chars and buf:
            chunks.append("".join(buf).strip())
            print(f"📦 Created chunk {len(chunks)} with {cur_len} chars.")
            buf, cur_len = [add], len(add)
        else:
            buf.append(add)
            cur_len += len(add)
    if buf:
        chunks.append("".join(buf).strip())
        print(f"📦 Created final chunk {len(chunks)} with {cur_len} chars.")
    print(f"✅ Total chunks: {len(chunks)}")
    return chunks


def run_ner_on_text(pipe, text: str, max_chunk_chars: int = 1500) -> List[dict]:
    print(f"🚀 Running NER on text ({len(text)} chars)...")
    chunks = _chunk_text(text, max_chunk_chars)
    print(f"📊 Processing {len(chunks)} chunks...")
    results: List[dict] = []
    cursor = 0

    for i, ch in enumerate(chunks):
        print(f"🔹 Processing chunk {i+1}/{len(chunks)} (len={len(ch)})...")
        out = pipe(ch)
        print(f"✅ Chunk {i+1}: {len(out)} entities found.")
        for ent in out:
            results.append({
                "text": ent.get("word"),
                "label": ent.get("entity_group") or ent.get("entity"),
                "score": float(ent.get("score", 0.0)),
                "start": ent["start"] + cursor,
                "end": ent["end"] + cursor,
            })
        cursor += len(ch)

    print(f"🔍 Deduplicating overlapping entities...")
    results.sort(key=lambda r: (r["start"], -r["score"]))
    dedup: List[dict] = []
    last_end = -1
    for r in results:
        if r["start"] >= last_end:
            dedup.append(r)
            last_end = r["end"]
        else:
            if r["score"] > dedup[-1]["score"]:
                dedup[-1] = r
                last_end = r["end"]
    print(f"🎯 Final entities count: {len(dedup)}")
    return dedup
