from typing import List, Dict, Sequence
import html
import plotly.graph_objects as go

LABEL_COLORS = {
    "PER": "#ffd166", "PERSON": "#ffd166",
    "ORG": "#06d6a0", "ORGANIZATION": "#06d6a0",
    "LOC": "#118ab2", "LOCATION": "#118ab2", "GPE": "#118ab2",
    "MISC": "#ef476f", "DATE": "#a78bfa", "TIME": "#60a5fa",
    "LAW": "#f59e0b", "EVENT": "#34d399", "PRODUCT": "#f472b6",
    "NORP": "#93c5fd", "FAC": "#c7d2fe", "WORK_OF_ART": "#fbcfe8",
    "LANGUAGE": "#fde68a", "PERCENT": "#fca5a5", "MONEY": "#86efac",
    "QUANTITY": "#67e8f9", "ORDINAL": "#f9a8d4", "CARDINAL": "#fef08a",
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
            f'{span_txt}<span style="opacity:.7; font-size:.75em; margin-left:4px;">{html.escape(label)}</span>'
            f"</span>"
        )
        last = e
    pieces.append(html.escape(text[last:]))
    print("✅ Rendering complete.")
    return (
        '<div style="font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto; '
        'line-height:1.6; white-space:pre-wrap;">'
        + "".join(pieces)
        + "</div>"
    )

def entities_to_dataframe(entities: List[Dict]):
    import pandas as pd
    print(f"🧾 Converting {len(entities)} entities to DataFrame...")
    rows = [{"text": e["text"], "label": e["label"], "score": round(float(e["score"]), 4),
             "start": e["start"], "end": e["end"]} for e in entities]
    return pd.DataFrame(rows)

def token_importance_bar(tokens: Sequence[str], scores: Sequence[float], max_tokens: int = 128):
    """
    Simple proxy 'attention' view:
    - x-axis: tokens
    - y-axis: importance (derived from entity scores covering the token)
    """
    print(f"🔥 Building token-importance bar: tokens={len(tokens)}, capped={max_tokens}")
    toks = list(tokens)[:max_tokens]
    scs = list(scores)[:max_tokens]
    fig = go.Figure(
        data=[go.Bar(x=list(range(len(toks))), y=scs, hovertext=toks)]
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=30, b=60),
        xaxis=dict(title="Token index (hover to see token)", tickmode="auto"),
        yaxis=dict(title="Importance (proxy from entity scores)"),
        title="Token Importance (Proxy Heatmap)",
    )
    return fig
