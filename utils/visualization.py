from typing import List, Dict
import html

LABEL_COLORS = {
    "PER": "#ffd166", "PERSON": "#ffd166",
    "ORG": "#06d6a0", "ORGANIZATION": "#06d6a0",
    "LOC": "#118ab2", "LOCATION": "#118ab2", "GPE": "#118ab2",
    "MISC": "#ef476f", "DATE": "#a78bfa", "TIME": "#60a5fa",
}

def _label_color(label: str) -> str:
    return LABEL_COLORS.get(label, "#e5e7eb")

def render_entities_html(text: str, entities: List[Dict]) -> str:
    print(f"🎨 Rendering {len(entities)} entities to HTML...")
    pieces, last = [], 0
    for ent in entities:
        s, e = ent["start"], ent["end"]
        label = ent["label"]
        color = _label_color(label)
        pieces.append(html.escape(text[last:s]))
        span_txt = html.escape(text[s:e])
        pieces.append(
            f'<span style="background:{color}; padding:2px 4px; border-radius:4px; margin:0 1px;">'
            f'{span_txt}<span style="opacity:.7; font-size:.8em; margin-left:4px;">{html.escape(label)}</span>'
            f"</span>"
        )
        last = e
    pieces.append(html.escape(text[last:]))
    print("✅ Rendering complete.")
    return (
        '<div style="font-family: system-ui; line-height:1.6; white-space:pre-wrap;">'
        + "".join(pieces)
        + "</div>"
    )

def entities_to_dataframe(entities: List[Dict]):
    import pandas as pd
    print(f"🧾 Converting {len(entities)} entities to dataframe...")
    rows = [{"text": e["text"], "label": e["label"], "score": round(float(e["score"]), 4),
             "start": e["start"], "end": e["end"]} for e in entities]
    print("✅ DataFrame created.")
    return pd.DataFrame(rows)
