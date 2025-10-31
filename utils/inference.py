from typing import Dict, Any, List, Callable
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def get_supported_models() -> Dict[str, str]:
    print("🔧 Supported model list requested.")
    return {
        "BERT (dslim/bert-base-NER)": "dslim/bert-base-NER",
        "RoBERTa (Jean-Baptiste/roberta-large-ner-english)": "Jean-Baptiste/roberta-large-ner-english",
        "XLM-RoBERTa (xlm-roberta-large-finetuned-conll03-english)": "xlm-roberta-large-finetuned-conll03-english",
    }

_PIPELINE_CACHE: Dict[str, Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}

def load_ner_pipeline(model_id: str, aggregation_strategy: str = "simple", device_map: str = "auto"):
    print(f"🧠 Loading model: {model_id}, aggregation={aggregation_strategy}, device={device_map}")
    cache_key = f"{model_id}|{aggregation_strategy}|{device_map}"
    if cache_key in _PIPELINE_CACHE:
        print("⚡ Using cached pipeline.")
        return _PIPELINE_CACHE[cache_key]

    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForTokenClassification.from_pretrained(model_id)
    print("✅ Model & tokenizer loaded.")

    pipe = pipeline(
        task="token-classification",
        model=mdl,
        tokenizer=tok,
        aggregation_strategy=aggregation_strategy,
        device_map=(None if device_map == "cpu" else device_map),
    )
    _PIPELINE_CACHE[cache_key] = pipe
    _TOKENIZER_CACHE[model_id] = tok
    print("✅ Pipeline ready and cached.")
    return pipe

def _chunk_text(text: str, max_chars: int) -> List[str]:
    print(f"✂️ Chunking text (max {max_chars} chars)...")
    if len(text) <= max_chars:
        print("✅ Single chunk.")
        return [text]
    chunks = []
    paragraph_splits = text.split("\n\n")
    buf, cur_len = [], 0
    for para in paragraph_splits:
        add = (para + "\n\n")
        if cur_len + len(add) > max_chars and buf:
            chunks.append("".join(buf).strip())
            print(f"📦 New chunk ({len(chunks)}), size={cur_len}")
            buf, cur_len = [add], len(add)
        else:
            buf.append(add)
            cur_len += len(add)
    if buf:
        chunks.append("".join(buf).strip())
        print(f"📦 Final chunk ({len(chunks)}), size={cur_len}")
    print(f"✅ Total chunks: {len(chunks)}")
    return chunks

def run_ner_on_text(
    pipe,
    text: str,
    max_chunk_chars: int = 1500,
    allow_nested: bool = True,
) -> List[dict]:
    """
    Returns list of entities with original text offsets:
    { 'text', 'label', 'score', 'start', 'end' }
    """
    print(f"🚀 NER inference on {len(text)} chars...")
    chunks = _chunk_text(text, max_chars=max_chunk_chars)
    results: List[dict] = []
    cursor = 0
    for i, ch in enumerate(chunks):
        print(f"🔹 Chunk {i+1}/{len(chunks)} (len={len(ch)})")
        out = pipe(ch)
        print(f"✅ Chunk {i+1}: {len(out)} entities")
        for ent in out:
            results.append({
                "text": ent.get("word"),
                "label": ent.get("entity_group") or ent.get("entity"),
                "score": float(ent.get("score", 0.0)),
                "start": ent["start"] + cursor,
                "end": ent["end"] + cursor,
            })
        cursor += len(ch)

    # Handle nested entities
    results.sort(key=lambda r: (r["start"], -r["score"]))
    if allow_nested:
        print("🧬 Nested-entity mode: keeping overlaps.")
        # No dedup; keep overlaps as-is
        return results

    # Otherwise: deduplicate by dropping overlapping lower-score spans
    print("🧹 Deduplicating overlaps (keep highest score).")
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
    print(f"🎯 Final entities: {len(dedup)}")
    return dedup

def tokenize_text(model_id: str, text: str) -> Dict[str, Any]:
    """
    Tokenize full text and expose a function to create a proxy
    token-importance array from detected entities (using entity scores).
    """
    print(f"🔤 Tokenizing text for token-importance view with model: {model_id}")
    tok = _TOKENIZER_CACHE.get(model_id) or AutoTokenizer.from_pretrained(model_id)
    _TOKENIZER_CACHE[model_id] = tok

    enc = tok(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=4096,  # generous for visualization only
    )
    tokens = tok.convert_ids_to_tokens(enc["input_ids"])
    offsets = enc["offset_mapping"]  # list of (start, end) per token

    def scores_from_entities(entities: List[dict]) -> List[float]:
        # Proxy importance: for each token, max entity score covering token span (else 0)
        scores = [0.0] * len(tokens)
        for ent in entities:
            es, ee, sc = ent["start"], ent["end"], float(ent["score"])
            for i, (ts, te) in enumerate(offsets):
                if ts >= ee or te <= es:
                    continue
                scores[i] = max(scores[i], sc)
        return scores

    print(f"✅ Tokenized: {len(tokens)} tokens")
    return {"tokens": tokens, "offsets": offsets, "scores_from_entities": scores_from_entities}
