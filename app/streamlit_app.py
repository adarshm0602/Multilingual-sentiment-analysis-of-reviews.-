"""
Streamlit App — Kannada Product Review Sentiment Analyzer

Run from the project root:
    venv/bin/streamlit run app/streamlit_app.py
"""

import io
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# ── Make the project root importable ─────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

# ── Page config — must be the very first Streamlit call ──────────────────────
st.set_page_config(
    page_title="Kannada Sentiment Analyzer",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Plotly (optional — graceful fallback message if missing) ──────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False

# =============================================================================
# ── Styling ───────────────────────────────────────────────────────────────────
# =============================================================================
_CSS = """
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+Kannada:wght@400;600&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Noto Sans Kannada', sans-serif;
}

/* ── Main container ── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 40%, #0f3460 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #d4dde8 !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #a9bcd0 !important;
    font-size: 0.9rem;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
}

/* ── App header ── */
.app-header {
    background: linear-gradient(135deg, #e63946 0%, #c1121f 40%, #780000 100%);
    padding: 1.75rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(198, 18, 31, 0.28);
    text-align: center;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: "ಕ";
    position: absolute;
    top: -20px; right: 30px;
    font-size: 9rem;
    opacity: 0.06;
    font-family: 'Noto Sans Kannada', sans-serif;
    color: #fff;
    pointer-events: none;
}
.app-header h1 {
    color: #ffffff !important;
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0 0 0.2rem 0;
    letter-spacing: -0.5px;
}
.app-header p {
    color: rgba(255,255,255,0.82) !important;
    font-size: 0.95rem;
    margin: 0;
}
.header-scripts {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 0.65rem;
    flex-wrap: wrap;
}
.script-pill {
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.22);
    color: #fff !important;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 500;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin-bottom: 0.5rem;
}

/* ── Sentiment verdict card ── */
.verdict-card {
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.10);
}
.verdict-positive { background: linear-gradient(135deg,#f0fdf4,#dcfce7); border: 2px solid #22c55e; }
.verdict-negative { background: linear-gradient(135deg,#fff1f2,#ffe4e6); border: 2px solid #ef4444; }
.verdict-neutral  { background: linear-gradient(135deg,#f9fafb,#f3f4f6); border: 2px solid #9ca3af; }
.verdict-emoji { font-size: 3.5rem; line-height: 1; flex-shrink: 0; }
.verdict-label { font-size: 2rem; font-weight: 700; line-height: 1.1; margin-bottom: 0.2rem; }
.verdict-positive .verdict-label { color: #15803d; }
.verdict-negative .verdict-label { color: #b91c1c; }
.verdict-neutral  .verdict-label { color: #4b5563; }
.verdict-sub { font-size: 0.9rem; color: #6b7280; font-weight: 400; }
.conf-bar-wrap {
    background: #e5e7eb; border-radius: 999px; height: 10px;
    margin-top: 0.5rem; overflow: hidden; width: 100%; max-width: 300px;
}
.conf-bar-fill-positive { background: linear-gradient(90deg,#4ade80,#16a34a); }
.conf-bar-fill-negative { background: linear-gradient(90deg,#f87171,#b91c1c); }
.conf-bar-fill-neutral  { background: linear-gradient(90deg,#d1d5db,#6b7280); }
.conf-bar-fill { height: 100%; border-radius: 999px; }
.conf-pct { font-size: 1.5rem; font-weight: 700; margin-top: 0.25rem; }
.verdict-positive .conf-pct { color: #15803d; }
.verdict-negative .conf-pct { color: #b91c1c; }
.verdict-neutral  .conf-pct { color: #4b5563; }

/* ── Pipeline flow ── */
.pipeline-section-title {
    font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: #9ca3af; margin-bottom: 0.75rem;
}
.pipeline-flow { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; margin-bottom: 1.25rem; }
.pipe-step {
    display: flex; align-items: center; gap: 6px;
    background: #f8fafc; border: 1.5px solid #e2e8f0;
    border-radius: 8px; padding: 6px 14px; font-size: 0.82rem;
    font-weight: 500; color: #374151;
}
.pipe-step.active { background: #eff6ff; border-color: #3b82f6; color: #1d4ed8; }
.pipe-step .step-num {
    background: #3b82f6; color: #fff; border-radius: 50%;
    width: 18px; height: 18px; display: inline-flex; align-items: center;
    justify-content: center; font-size: 0.68rem; font-weight: 700; flex-shrink: 0;
}
.pipe-arrow { color: #9ca3af; font-size: 1rem; flex-shrink: 0; }

/* ── Language badge ── */
.lang-badge {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 0.76rem; font-weight: 600; letter-spacing: 0.03em; margin-left: 6px;
}
.lang-kannada_script    { background:#fff7ed; color:#c2410c; border:1px solid #fed7aa; }
.lang-romanized_kannada { background:#f0fdfa; color:#0f766e; border:1px solid #99f6e4; }
.lang-english           { background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; }
.lang-mixed             { background:#fefce8; color:#a16207; border:1px solid #fef08a; }
.lang-unknown           { background:#f9fafb; color:#6b7280; border:1px solid #e5e7eb; }

/* ── Output cards ── */
.output-card {
    background: #ffffff; border: 1.5px solid #e5e7eb; border-radius: 10px;
    padding: 1rem 1.25rem; margin-bottom: 0.75rem;
}
.output-card-label {
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: #9ca3af; margin-bottom: 0.35rem;
}
.output-card-value {
    font-size: 1rem; color: #111827;
    font-family: 'Noto Sans Kannada','Inter',sans-serif; line-height: 1.5;
}

/* ── Timing grid ── */
.timing-grid { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 0.5rem; }
.timing-cell {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 8px 14px; font-size: 0.78rem; color: #374151; text-align: center;
}
.timing-cell strong { display: block; font-size: 1.05rem; color: #111827; }

/* ── Warning banner ── */
.warn-banner {
    background: #fffbeb; border: 1px solid #fcd34d; border-radius: 8px;
    padding: 0.75rem 1rem; font-size: 0.84rem; color: #92400e; margin-bottom: 0.75rem;
}

/* ── Sidebar team card ── */
.team-card {
    background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px; padding: 0.6rem 0.9rem; margin-bottom: 0.4rem;
    font-size: 0.85rem; color: #d4dde8 !important;
    display: flex; align-items: center; gap: 8px;
}

/* ── Batch section ── */
.upload-zone {
    border: 2px dashed #d1d5db; border-radius: 12px;
    padding: 1.5rem; text-align: center;
    background: #fafafa; margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.preview-card {
    background: #ffffff; border: 1.5px solid #e5e7eb;
    border-radius: 10px; overflow: hidden; margin-bottom: 1rem;
}
.preview-header {
    background: #f8fafc; padding: 0.6rem 1rem;
    font-size: 0.78rem; font-weight: 600; color: #374151;
    border-bottom: 1px solid #e5e7eb;
}
.stat-card {
    background: #ffffff; border: 1.5px solid #e5e7eb; border-radius: 12px;
    padding: 1.25rem; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.stat-val { font-size: 2rem; font-weight: 700; color: #111827; line-height: 1.1; }
.stat-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.07em; color: #6b7280; margin-top: 4px; }
.stat-positive .stat-val { color: #15803d; }
.stat-negative .stat-val { color: #b91c1c; }
.stat-neutral  .stat-val { color: #4b5563; }
.results-table-wrap { border: 1.5px solid #e5e7eb; border-radius: 10px; overflow: hidden; }
.row-limit-warn {
    background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px;
    padding: 0.75rem 1rem; font-size: 0.84rem; color: #1e40af; margin-bottom: 0.75rem;
}
.dashboard-title {
    font-size: 1.1rem; font-weight: 700; color: #111827;
    margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem;
    border-bottom: 2px solid #e5e7eb;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
"""

st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

# =============================================================================
# ── Constants ─────────────────────────────────────────────────────────────────
# =============================================================================

TEAM_MEMBERS = [
    ("🎓", "Adarsh M"),
    ("🎓", "Team Member 2"),   # ← edit
    ("🎓", "Team Member 3"),   # ← edit
]

SAMPLE_REVIEWS = [
    {
        "label": "ಕನ್ನಡ ✍",
        "text": "ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ, ನಾನು ತುಂಬಾ ಖುಷಿಯಾಗಿದ್ದೇನೆ!",
        "hint": "Kannada script — positive",
    },
    {
        "label": "Romanized 🔤",
        "text": "Tumba ketta product, waste of money, baralla",
        "hint": "Romanized Kannada — negative",
    },
    {
        "label": "English 🇬🇧",
        "text": "Excellent quality! Delivery was fast and packaging was great.",
        "hint": "English — positive",
    },
    {
        "label": "Negative 🙁",
        "text": "ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಕೆಟ್ಟದು, ಮರಳಿ ಖರೀದಿಸುವುದಿಲ್ಲ.",
        "hint": "Kannada script — negative",
    },
]

_LANG_LABELS: Dict[str, str] = {
    "kannada_script":    "Kannada Script",
    "romanized_kannada": "Romanized Kannada",
    "english":           "English",
    "mixed":             "Mixed",
    "unknown":           "Unknown",
}

_LANG_DISPLAY: Dict[str, str] = {  # human-readable for charts
    "kannada_script":    "Kannada Script",
    "romanized_kannada": "Romanized Kannada",
    "english":           "English",
    "mixed":             "Mixed",
    "unknown":           "Unknown",
}

_SENTIMENT_CONFIG: Dict[str, Dict] = {
    "Positive": {"emoji": "😊", "css": "positive"},
    "Negative": {"emoji": "😞", "css": "negative"},
    "Neutral":  {"emoji": "😐", "css": "neutral"},
}

_STEP_ICONS: Dict[str, str] = {
    "language_detection":       "🔍 Language Detection",
    "transliteration":          "🔤 Transliteration",
    "translation":              "🌐 Translation",
    "sentiment_classification": "💬 Sentiment",
}

_BACKEND_LABELS: Dict[str, str] = {
    "indictrans2": "IndicTrans2 (offline, best quality)",
    "google":      "Google Translate (online, needs API key)",
    "fallback":    "Dictionary Fallback (offline, fast)",
}

# Colour palettes reused in charts
_SENTIMENT_COLORS = {"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#9ca3af"}
_LANG_COLORS = {
    "Kannada Script":    "#ff6b35",
    "Romanized Kannada": "#4ecdc4",
    "English":           "#45b7d1",
    "Mixed":             "#f5a623",
    "Unknown":           "#a8a8a8",
}

_MAX_ROWS = 500   # hard cap to prevent runaway processing
_DEMO_CSV_PATH = _ROOT / "data" / "demo_data.csv"

# =============================================================================
# ── Pipeline loader ───────────────────────────────────────────────────────────
# =============================================================================

@st.cache_resource(show_spinner=False)
def _load_pipeline(translation_backend: str):
    """Initialize the pipeline once per backend selection (cached per backend)."""
    from src.pipeline import KannadaSentimentPipeline
    return KannadaSentimentPipeline(
        translation_backend=translation_backend,
        use_transliteration_model=True,
        auto_fallback=True,
    )


# =============================================================================
# ── Sidebar ───────────────────────────────────────────────────────────────────
# =============================================================================

def _render_sidebar() -> str:
    """Render the sidebar. Returns selected translation backend."""
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;padding:1rem 0 0.5rem 0;">
                <div style="font-size:3rem;line-height:1;">🏷️</div>
                <div style="font-size:1.05rem;font-weight:700;color:#e2e8f0;
                            margin-top:4px;letter-spacing:-0.3px;">
                    Kannada Sentiment<br>Analyzer
                </div>
                <div style="font-size:0.75rem;color:#64748b;margin-top:4px;">
                    Multilingual NLP Pipeline
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.markdown(
            """
            <p style="font-size:0.82rem;color:#94a3b8;line-height:1.6;margin:0;">
            Analyzes product reviews written in
            <strong style="color:#e2e8f0;">Kannada script</strong>,
            <strong style="color:#e2e8f0;">Romanized Kannada</strong>, or
            <strong style="color:#e2e8f0;">English</strong>
            using a three-stage NLP pipeline:
            </p>
            <ul style="font-size:0.8rem;color:#94a3b8;margin:0.6rem 0 0 0;
                       padding-left:1.1rem;line-height:1.8;">
                <li>Script &amp; language detection</li>
                <li>Transliteration (Romanized → Kannada)</li>
                <li>Machine translation (Kannada → English)</li>
                <li>Transformer-based sentiment classification</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.markdown(
            '<p style="font-size:0.72rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:0.1em;color:#64748b;margin-bottom:6px;">⚙️ Settings</p>',
            unsafe_allow_html=True,
        )
        backend = st.radio(
            "Translation Engine",
            options=list(_BACKEND_LABELS.keys()),
            format_func=lambda x: _BACKEND_LABELS[x],
            index=0,
            help=(
                "IndicTrans2 — high-quality offline model (slow first load).\n"
                "Google — cloud API, requires GOOGLE_TRANSLATE_API_KEY.\n"
                "Fallback — fast dictionary lookup, lower accuracy."
            ),
        )
        if backend == "google":
            st.info(
                "🔑 Set `GOOGLE_TRANSLATE_API_KEY` in your environment "
                "before launching.",
                icon="ℹ️",
            )

        st.markdown("---")

        st.markdown(
            '<p style="font-size:0.72rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:0.1em;color:#64748b;margin-bottom:8px;">👥 Team</p>',
            unsafe_allow_html=True,
        )
        for icon, name in TEAM_MEMBERS:
            st.markdown(
                f'<div class="team-card">{icon}&nbsp;{name}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        st.markdown(
            """
            <p style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
               letter-spacing:0.1em;color:#64748b;margin-bottom:6px;">🌐 Supported Input</p>
            <table style="font-size:0.78rem;color:#94a3b8;border-collapse:collapse;width:100%;">
                <tr><td style="padding:3px 0;">🔴</td><td style="padding:3px 6px;">ಕನ್ನಡ (Kannada script)</td></tr>
                <tr><td style="padding:3px 0;">🟢</td><td style="padding:3px 6px;">Romanized Kannada</td></tr>
                <tr><td style="padding:3px 0;">🔵</td><td style="padding:3px 6px;">English</td></tr>
                <tr><td style="padding:3px 0;">🟡</td><td style="padding:3px 6px;">Mixed language</td></tr>
            </table>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="font-size:0.68rem;color:#475569;margin-top:1.5rem;text-align:center;">'
            'Built with Streamlit · PyTorch · HuggingFace</div>',
            unsafe_allow_html=True,
        )

    return backend


# =============================================================================
# ── Single-review result renderers ────────────────────────────────────────────
# =============================================================================

def _render_verdict(result: Dict[str, Any]) -> None:
    label = result.get("sentiment_label", "Neutral")
    conf  = result.get("confidence_score", 0.0)
    cfg   = _SENTIMENT_CONFIG.get(label, _SENTIMENT_CONFIG["Neutral"])
    pct   = int(conf * 100)
    css   = cfg["css"]
    st.markdown(
        f"""
        <div class="verdict-card verdict-{css}">
            <div class="verdict-emoji">{cfg["emoji"]}</div>
            <div style="flex:1;">
                <div class="verdict-label">{label}</div>
                <div class="verdict-sub">Sentiment prediction</div>
                <div class="conf-pct">{pct}% confidence</div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar-fill conf-bar-fill-{css}" style="width:{pct}%;"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_pipeline_flow(result: Dict[str, Any]) -> None:
    steps = result.get("pipeline_steps", [])
    if not steps:
        return
    st.markdown('<p class="pipeline-section-title">Processing pipeline</p>', unsafe_allow_html=True)
    parts = []
    for i, step in enumerate(steps):
        label = _STEP_ICONS.get(step, step.replace("_", " ").title())
        parts.append(
            f'<div class="pipe-step active"><span class="step-num">{i+1}</span> {label}</div>'
        )
        if i < len(steps) - 1:
            parts.append('<span class="pipe-arrow">→</span>')
    st.markdown(f'<div class="pipeline-flow">{"".join(parts)}</div>', unsafe_allow_html=True)


def _render_language_row(result: Dict[str, Any]) -> None:
    lang  = result.get("detected_language", "unknown")
    conf  = result.get("detection_confidence", 0.0)
    label = _LANG_LABELS.get(lang, lang)
    st.markdown(
        f'<div class="output-card">'
        f'<div class="output-card-label">Detected Language</div>'
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<span class="output-card-value">{label}</span>'
        f'<span class="lang-badge lang-{lang}">{int(conf*100)}% confidence</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _render_intermediates(result: Dict[str, Any]) -> None:
    tlit = result.get("transliterated_text")
    tran = result.get("translated_text")
    col1, col2 = st.columns(2)
    with col1:
        if tlit:
            st.markdown(
                f'<div class="output-card"><div class="output-card-label">'
                f'🔤 Transliterated (Romanized → Kannada script)</div>'
                f'<div class="output-card-value">{tlit}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="output-card" style="opacity:0.45;"><div class="output-card-label">'
                '🔤 Transliteration</div>'
                '<div class="output-card-value" style="color:#9ca3af;font-style:italic;">'
                'Not required</div></div>',
                unsafe_allow_html=True,
            )
    with col2:
        if tran:
            st.markdown(
                f'<div class="output-card"><div class="output-card-label">'
                f'🌐 Translated to English</div>'
                f'<div class="output-card-value">{tran}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="output-card" style="opacity:0.45;"><div class="output-card-label">'
                '🌐 Translation</div>'
                '<div class="output-card-value" style="color:#9ca3af;font-style:italic;">'
                'Not required</div></div>',
                unsafe_allow_html=True,
            )


def _render_errors(result: Dict[str, Any]) -> None:
    for err in result.get("errors", []):
        st.markdown(f'<div class="warn-banner">⚠️ {err}</div>', unsafe_allow_html=True)


def _render_timings(result: Dict[str, Any]) -> None:
    timings = result.get("timings", {})
    if not timings:
        return
    labels = {
        "language_detection":       "Language Detection",
        "transliteration":          "Transliteration",
        "translation":              "Translation",
        "sentiment_classification": "Sentiment Classification",
        "total":                    "⏱ Total",
    }
    with st.expander("⏱ Timing breakdown", expanded=False):
        cells = "".join(
            f'<div class="timing-cell"><strong>{timings[k]*1000:.0f} ms</strong>{v}</div>'
            for k, v in labels.items()
            if k in timings
        )
        st.markdown(f'<div class="timing-grid">{cells}</div>', unsafe_allow_html=True)


def _render_single_result(result: Dict[str, Any]) -> None:
    st.markdown('<p class="section-label">Analysis Results</p>', unsafe_allow_html=True)
    _render_verdict(result)
    _render_pipeline_flow(result)
    _render_language_row(result)
    _render_intermediates(result)
    _render_errors(result)
    _render_timings(result)


# =============================================================================
# ── Batch helpers ─────────────────────────────────────────────────────────────
# =============================================================================

def _read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Parse CSV or Excel upload into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError(f"Unsupported file type '{uploaded_file.name}'. Use CSV or Excel.")


def _flatten_result(raw: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a pipeline result dict into a single-level dict for DataFrame rows."""
    errors = result.get("errors", [])
    return {
        "original_text":        result.get("original_text", raw),
        "detected_language":    result.get("detected_language", "unknown"),
        "detection_confidence": round(result.get("detection_confidence", 0.0), 4),
        "transliterated_text":  result.get("transliterated_text") or "",
        "translated_text":      result.get("translated_text") or "",
        "sentiment_label":      result.get("sentiment_label", "Neutral"),
        "confidence_score":     round(result.get("confidence_score", 0.0), 4),
        "pipeline_steps":       ", ".join(result.get("pipeline_steps", [])),
        "errors":               "; ".join(errors),
        "processing_time_s":    round(result.get("timings", {}).get("total", 0.0), 4),
        "status":               "ok_with_warnings" if errors else "ok",
    }


def _skipped_row(raw: Any, reason: str) -> Dict[str, Any]:
    return {
        "original_text": str(raw), "detected_language": "unknown",
        "detection_confidence": 0.0, "transliterated_text": "",
        "translated_text": "", "sentiment_label": "Neutral",
        "confidence_score": 0.0, "pipeline_steps": "",
        "errors": reason, "processing_time_s": 0.0, "status": "skipped",
    }


def _error_row(raw: Any, error: str) -> Dict[str, Any]:
    row = _skipped_row(raw, error)
    row["status"] = "error"
    return row


def _run_batch_with_progress(pipeline, texts: List[Any]) -> pd.DataFrame:
    """
    Process texts one at a time, updating st.progress() after each item.
    Returns a DataFrame of flattened results.
    """
    n = len(texts)
    progress_bar = st.progress(0.0, text=f"Starting — 0 / {n} reviews processed…")
    status_slot  = st.empty()

    rows: List[Dict[str, Any]] = []

    for i, raw in enumerate(texts):
        # ── Validate ────────────────────────────────────────────────────────
        if not isinstance(raw, str) or not str(raw).strip():
            rows.append(_skipped_row(raw, "Empty or non-string input — skipped"))
        else:
            try:
                result = pipeline.process(str(raw).strip())
                rows.append(_flatten_result(raw, result))
            except Exception as exc:
                logger.error("Batch item %d failed: %s", i, exc)
                rows.append(_error_row(raw, str(exc)))

        # ── Update progress ──────────────────────────────────────────────────
        done   = i + 1
        pct    = done / n
        last   = rows[-1]
        s_lbl  = last["sentiment_label"]
        s_conf = last["confidence_score"]
        s_stat = last["status"]

        progress_bar.progress(
            pct,
            text=f"Processing {done} / {n} reviews…",
        )
        if s_stat in ("ok", "ok_with_warnings"):
            cfg = _SENTIMENT_CONFIG.get(s_lbl, _SENTIMENT_CONFIG["Neutral"])
            status_slot.markdown(
                f'<div style="font-size:0.8rem;color:#6b7280;text-align:center;'
                f'padding:2px 0;">'
                f'Last: {cfg["emoji"]} <strong>{s_lbl}</strong> '
                f'({s_conf*100:.0f}%) · {_LANG_LABELS.get(last["detected_language"],"?")}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            status_slot.markdown(
                f'<div style="font-size:0.8rem;color:#92400e;text-align:center;">'
                f'⚠️ Item {done}: {s_stat}</div>',
                unsafe_allow_html=True,
            )

    progress_bar.progress(1.0, text=f"✅ Done — {n} reviews processed.")
    status_slot.empty()

    return pd.DataFrame(rows)


# =============================================================================
# ── Summary dashboard ─────────────────────────────────────────────────────────
# =============================================================================

def _render_summary_dashboard(df: pd.DataFrame) -> None:
    """Render key metrics + three interactive Plotly charts."""
    if not _PLOTLY_OK:
        st.warning(
            "Install plotly for interactive charts: `pip install plotly`",
            icon="📊",
        )
        return

    ok_mask   = df["status"].isin(["ok", "ok_with_warnings"])
    ok_df     = df[ok_mask]
    total     = len(df)
    processed = int(ok_mask.sum())
    errors    = total - processed
    avg_conf  = float(ok_df["confidence_score"].mean()) if not ok_df.empty else 0.0
    avg_time  = float(ok_df["processing_time_s"].mean()) if not ok_df.empty else 0.0

    # ── KPI row ──────────────────────────────────────────────────────────────
    st.markdown('<p class="dashboard-title">📊 Summary Dashboard</p>', unsafe_allow_html=True)

    pos_n = int((ok_df["sentiment_label"] == "Positive").sum())
    neg_n = int((ok_df["sentiment_label"] == "Negative").sum())
    neu_n = int((ok_df["sentiment_label"] == "Neutral").sum())

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    _kpi(k1, str(total),        "Total Reviews",  "")
    _kpi(k2, str(processed),    "Processed",       "")
    _kpi(k3, str(pos_n),        "Positive",        "stat-positive")
    _kpi(k4, str(neg_n),        "Negative",        "stat-negative")
    _kpi(k5, f"{avg_conf*100:.1f}%", "Avg Confidence", "")
    _kpi(k6, f"{avg_time*1000:.0f} ms", "Avg Time/Review", "")

    if ok_df.empty:
        st.info("No successfully processed rows to chart.")
        return

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Pie + Language bar ─────────────────────────────────────────────
    c_pie, c_lang = st.columns(2)

    with c_pie:
        sent_counts = (
            ok_df["sentiment_label"]
            .value_counts()
            .reset_index()
            .rename(columns={"sentiment_label": "Sentiment", "count": "Count"})
        )
        # pandas ≥ 2.0 renames the column; handle both
        if "index" in sent_counts.columns:
            sent_counts = sent_counts.rename(columns={"index": "Sentiment", "sentiment_label": "Count"})

        fig_pie = px.pie(
            sent_counts,
            values="Count",
            names="Sentiment",
            title="Sentiment Distribution",
            color="Sentiment",
            color_discrete_map=_SENTIMENT_COLORS,
            hole=0.45,
        )
        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label",
            textfont_size=13,
            pull=[0.04] * len(sent_counts),
        )
        fig_pie.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.25,
                xanchor="center", x=0.5,
            ),
            margin=dict(t=50, b=70, l=10, r=10),
            title_font_size=14,
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c_lang:
        lang_counts = (
            ok_df["detected_language"]
            .map(lambda x: _LANG_DISPLAY.get(x, x))
            .value_counts()
            .reset_index()
            .rename(columns={"detected_language": "Language", "count": "Count"})
        )
        if "index" in lang_counts.columns:
            lang_counts = lang_counts.rename(columns={"index": "Language", "detected_language": "Count"})

        fig_lang = px.bar(
            lang_counts,
            x="Count",
            y="Language",
            orientation="h",
            title="Language Distribution",
            color="Language",
            color_discrete_map=_LANG_COLORS,
            text="Count",
        )
        fig_lang.update_traces(textposition="outside", textfont_size=12)
        fig_lang.update_layout(
            showlegend=False,
            margin=dict(t=50, b=20, l=10, r=50),
            title_font_size=14,
            height=340,
            xaxis_title="Reviews",
            yaxis_title="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_lang.update_xaxes(gridcolor="#f0f0f0")
        st.plotly_chart(fig_lang, use_container_width=True)

    # ── Row 2: Confidence histogram ───────────────────────────────────────────
    fig_hist = px.histogram(
        ok_df,
        x="confidence_score",
        color="sentiment_label",
        title="Confidence Score Distribution by Sentiment",
        nbins=25,
        color_discrete_map=_SENTIMENT_COLORS,
        labels={
            "confidence_score": "Confidence Score",
            "sentiment_label":  "Sentiment",
        },
        barmode="overlay",
        opacity=0.78,
    )
    fig_hist.update_layout(
        height=300,
        title_font_size=14,
        margin=dict(t=50, b=30, l=10, r=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.35,
            xanchor="center", x=0.5, title="",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 1], gridcolor="#f0f0f0"),
        yaxis_title="Number of Reviews",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Row 3: Avg confidence per language (grouped bar) ─────────────────────
    if ok_df["detected_language"].nunique() > 1:
        conf_by_lang = (
            ok_df.groupby(["detected_language", "sentiment_label"])["confidence_score"]
            .mean()
            .reset_index()
        )
        conf_by_lang["language_label"] = conf_by_lang["detected_language"].map(
            lambda x: _LANG_DISPLAY.get(x, x)
        )
        fig_conf = px.bar(
            conf_by_lang,
            x="language_label",
            y="confidence_score",
            color="sentiment_label",
            title="Average Confidence by Language & Sentiment",
            barmode="group",
            color_discrete_map=_SENTIMENT_COLORS,
            labels={
                "language_label":    "Language",
                "confidence_score":  "Avg Confidence",
                "sentiment_label":   "Sentiment",
            },
            text_auto=".0%",
        )
        fig_conf.update_layout(
            height=320,
            title_font_size=14,
            margin=dict(t=50, b=60, l=10, r=10),
            yaxis=dict(tickformat=".0%", range=[0, 1.05], gridcolor="#f0f0f0"),
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.35,
                xanchor="center", x=0.5, title="",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_conf, use_container_width=True)


def _kpi(col, value: str, label: str, extra_css: str) -> None:
    """Render a single KPI card inside a column."""
    with col:
        st.markdown(
            f'<div class="stat-card {extra_css}">'
            f'<div class="stat-val">{value}</div>'
            f'<div class="stat-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# ── Tab renderers ─────────────────────────────────────────────────────────────
# =============================================================================

def _render_single_tab(backend: str) -> None:
    """Content for the Single Review tab."""
    # ── Sample quick-fill ─────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Try a sample review</p>', unsafe_allow_html=True)
    sample_cols = st.columns(len(SAMPLE_REVIEWS))
    for col, sample in zip(sample_cols, SAMPLE_REVIEWS):
        with col:
            if st.button(sample["label"], help=sample["hint"], use_container_width=True):
                st.session_state["review_text"] = sample["text"]
                st.session_state.pop("last_result", None)
                st.rerun()

    # ── Text area ─────────────────────────────────────────────────────────────
    st.markdown(
        '<p class="section-label" style="margin-top:0.75rem;">Your Review</p>',
        unsafe_allow_html=True,
    )
    review_text: str = st.text_area(
        label="Enter a product review in Kannada, Romanized Kannada, or English:",
        value=st.session_state.get("review_text", ""),
        height=140,
        max_chars=2000,
        key="review_text",
        placeholder=(
            "Type or paste a product review…\n\n"
            "Examples:\n"
            "  • ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ  (Kannada script)\n"
            "  • Tumba chennagide, olle product  (Romanized Kannada)\n"
            "  • This product is absolutely amazing!  (English)"
        ),
    )
    st.markdown(
        f'<div style="text-align:right;font-size:0.75rem;color:#9ca3af;'
        f'margin-top:-0.5rem;margin-bottom:0.75rem;">'
        f'{len(review_text)} / 2000 characters</div>',
        unsafe_allow_html=True,
    )

    # ── Buttons ───────────────────────────────────────────────────────────────
    left, _, right = st.columns([2, 3, 2])
    with left:
        analyze_clicked = st.button(
            "🔍 Analyze Sentiment",
            type="primary",
            use_container_width=True,
            disabled=not review_text.strip(),
        )
    with right:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state["review_text"] = ""
            st.session_state.pop("last_result", None)
            st.rerun()

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if analyze_clicked:
        text = review_text.strip()
        if not text:
            st.warning("Please enter a review before clicking Analyze.", icon="⚠️")
        else:
            try:
                with st.spinner("Loading models and analyzing your review…"):
                    pipeline = _load_pipeline(backend)
                    result   = pipeline.process(text)
                st.session_state["last_result"] = result
            except Exception as exc:
                st.error(
                    f"❌ Pipeline error: {exc}\n\n"
                    "Check that all dependencies are installed.",
                    icon="🚨",
                )
                logger.exception("Single-review pipeline crashed")

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown(
        "<hr style='border-color:#e5e7eb;margin:1.25rem 0;'>",
        unsafe_allow_html=True,
    )
    if "last_result" in st.session_state:
        _render_single_result(st.session_state["last_result"])
    else:
        st.markdown(
            """
            <div style="text-align:center;padding:3rem 1rem;color:#9ca3af;">
                <div style="font-size:4rem;margin-bottom:0.75rem;">🔍</div>
                <div style="font-size:1rem;font-weight:500;color:#6b7280;">
                    Results will appear here
                </div>
                <div style="font-size:0.84rem;margin-top:0.35rem;">
                    Enter a review above and click <strong>Analyze Sentiment</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_batch_tab(backend: str) -> None:
    """Content for the Batch Upload tab."""

    # ── Session-state initialisation ──────────────────────────────────────────
    if "batch_uploader_key" not in st.session_state:
        st.session_state["batch_uploader_key"] = 0

    # ── Demo Data button ──────────────────────────────────────────────────────
    col_demo, col_hint = st.columns([2, 5])
    with col_demo:
        if st.button(
            "🎯 Load Demo Data",
            use_container_width=True,
            help="Pre-fill with 20 sample reviews: Kannada script, Romanized Kannada, and English.",
        ):
            try:
                demo_df = pd.read_csv(_DEMO_CSV_PATH)
                st.session_state["batch_input_df"]   = demo_df
                st.session_state["batch_is_demo"]    = True
                st.session_state["batch_file_id"]    = None
                st.session_state["batch_uploader_key"] += 1   # reset the uploader widget
                st.session_state.pop("batch_results_df", None)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load demo data: {exc}", icon="🚨")

    with col_hint:
        st.markdown(
            '<div style="padding-top:0.55rem;font-size:0.82rem;color:#6b7280;">'
            'Try the demo data or upload your own CSV / Excel file below.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── File uploader ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-label" style="margin-top:0.5rem;">Upload File</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop a CSV or Excel file containing product reviews",
        type=["csv", "xlsx", "xls"],
        help="The file must have a column containing review text.",
        label_visibility="collapsed",
        key=f"batch_uploader_{st.session_state['batch_uploader_key']}",
    )

    # ── Handle new file upload ────────────────────────────────────────────────
    if uploaded is not None:
        file_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("batch_file_id") != file_id:
            # New file — parse, cache, and clear stale results
            try:
                df_parsed = _read_uploaded_file(uploaded)
                st.session_state["batch_input_df"] = df_parsed
                st.session_state["batch_is_demo"]  = False
                st.session_state["batch_file_id"]  = file_id
                st.session_state.pop("batch_results_df", None)
            except Exception as exc:
                st.error(f"Could not read file: {exc}", icon="🚨")
                return

    # ── Resolve active DataFrame ───────────────────────────────────────────────
    df_input: Optional[pd.DataFrame] = st.session_state.get("batch_input_df")
    is_demo: bool = st.session_state.get("batch_is_demo", False)

    if df_input is None or df_input.empty:
        st.markdown(
            """
            <div class="upload-zone">
                <div style="font-size:2.5rem;margin-bottom:0.5rem;">📂</div>
                <div style="font-size:0.95rem;font-weight:600;color:#374151;">
                    Drag &amp; drop your CSV or Excel file here, or click <em>Load Demo Data</em>
                </div>
                <div style="font-size:0.82rem;color:#9ca3af;margin-top:4px;">
                    Supports .csv · .xlsx · .xls
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    total_rows = len(df_input)

    # ── Source indicator ──────────────────────────────────────────────────────
    if is_demo:
        st.info(
            "**Demo mode** — showing 20 sample reviews (10 Kannada script · 5 Romanized · 5 English). "
            "Upload your own file above to replace.",
            icon="🎯",
        )
    else:
        file_label = uploaded.name if uploaded is not None else "cached file"
        st.success(
            f"Loaded **{file_label}** — {total_rows:,} rows × {len(df_input.columns)} columns",
            icon="📄",
        )

    # ── Data preview ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-label" style="margin-top:0.75rem;">Data Preview (first 5 rows)</p>', unsafe_allow_html=True)
    st.dataframe(
        df_input.head(5),
        use_container_width=True,
        hide_index=False,
    )

    # ── Column selector ───────────────────────────────────────────────────────
    st.markdown('<p class="section-label" style="margin-top:0.75rem;">Column Selection</p>', unsafe_allow_html=True)
    text_cols = df_input.columns.tolist()

    _guess_keywords = ("review", "text", "comment", "description", "feedback", "content")
    # Prefer an exact column-name match first (e.g. "review" before "review_id")
    default_idx = next(
        (i for i, c in enumerate(text_cols) if c.lower() in _guess_keywords),
        None,
    )
    # Fall back to substring match only when no exact match is found
    if default_idx is None:
        default_idx = next(
            (i for i, c in enumerate(text_cols) if any(k in c.lower() for k in _guess_keywords)),
            0,
        )
    selected_col = st.selectbox(
        "Select the column that contains the review text:",
        options=text_cols,
        index=default_idx,
        help="Choose the column whose values will be fed into the sentiment pipeline.",
    )

    reviews_series = df_input[selected_col]
    non_empty      = reviews_series.dropna().astype(str).str.strip().str.len().gt(0).sum()
    st.markdown(
        f'<div style="font-size:0.82rem;color:#6b7280;margin-top:-0.4rem;margin-bottom:0.5rem;">'
        f'Selected column has <strong>{non_empty:,}</strong> non-empty values '
        f'out of {total_rows:,} rows.</div>',
        unsafe_allow_html=True,
    )

    # ── Row limit warning ─────────────────────────────────────────────────────
    process_n = min(total_rows, _MAX_ROWS)
    if total_rows > _MAX_ROWS:
        st.markdown(
            f'<div class="row-limit-warn">ℹ️ Your file has <strong>{total_rows:,}</strong> rows. '
            f'Only the first <strong>{_MAX_ROWS:,}</strong> will be processed to prevent '
            f'excessive runtime. Export the full DataFrame programmatically for larger batches.</div>',
            unsafe_allow_html=True,
        )

    # ── Process All / Clear buttons ───────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, col_clear = st.columns([2, 5])
    with col_btn:
        process_clicked = st.button(
            f"⚡ Process All ({process_n:,} reviews)",
            type="primary",
            use_container_width=True,
        )
    with col_clear:
        if st.button("🗑 Clear Results", use_container_width=True):
            for _k in ("batch_results_df", "batch_input_df", "batch_is_demo", "batch_file_id"):
                st.session_state.pop(_k, None)
            st.session_state["batch_uploader_key"] += 1
            st.rerun()

    # ── Run batch ─────────────────────────────────────────────────────────────
    if process_clicked:
        texts = reviews_series.iloc[:process_n].tolist()
        try:
            pipeline = _load_pipeline(backend)
            with st.spinner("Initializing models…"):
                pass

            st.markdown(
                '<p class="section-label" style="margin-top:1rem;">Processing Progress</p>',
                unsafe_allow_html=True,
            )
            results_df = _run_batch_with_progress(pipeline, texts)
            st.session_state["batch_results_df"] = results_df
            st.success(
                f"✅ Processed {process_n:,} reviews successfully.",
                icon="🎉",
            )
        except Exception as exc:
            st.error(f"Batch processing failed: {exc}", icon="🚨")
            logger.exception("Batch pipeline crashed")

    # ── Show results ──────────────────────────────────────────────────────────
    if "batch_results_df" not in st.session_state:
        return

    results_df: pd.DataFrame = st.session_state["batch_results_df"]

    st.markdown(
        "<hr style='border-color:#e5e7eb;margin:1.5rem 0;'>",
        unsafe_allow_html=True,
    )

    # ── Summary dashboard ─────────────────────────────────────────────────────
    _render_summary_dashboard(results_df)

    # ── Results table ─────────────────────────────────────────────────────────
    st.markdown('<p class="dashboard-title">📋 Full Results Table</p>', unsafe_allow_html=True)

    display_cols = [
        "original_text", "sentiment_label", "confidence_score",
        "detected_language", "transliterated_text", "translated_text",
        "errors", "status", "processing_time_s",
    ]
    display_df = results_df[[c for c in display_cols if c in results_df.columns]]

    def _color_sentiment(val: str) -> str:
        return {
            "Positive": "color: #15803d; font-weight:600",
            "Negative": "color: #b91c1c; font-weight:600",
            "Neutral":  "color: #4b5563; font-weight:600",
        }.get(val, "")

    def _color_status(val: str) -> str:
        return {
            "ok":               "color:#15803d",
            "ok_with_warnings": "color:#a16207",
            "error":            "color:#b91c1c",
            "skipped":          "color:#6b7280",
        }.get(val, "")

    styled = (
        display_df.style
        .applymap(_color_sentiment, subset=["sentiment_label"])
        .applymap(_color_status,    subset=["status"])
        .format({"confidence_score": "{:.1%}", "processing_time_s": "{:.3f}s"})
    )
    st.dataframe(styled, use_container_width=True, height=380)

    # ── Download button ───────────────────────────────────────────────────────
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Full Results as CSV",
        data=csv_bytes,
        file_name="sentiment_results.csv",
        mime="text/csv",
        use_container_width=False,
    )


# =============================================================================
# ── Main ──────────────────────────────────────────────────────────────────────
# =============================================================================

def main() -> None:
    backend = _render_sidebar()

    # ── Persistent app header ─────────────────────────────────────────────────
    st.markdown(
        """
        <div class="app-header">
            <h1>🏷️ Kannada Product Review Sentiment Analyzer</h1>
            <p>Analyze reviews in any script — the pipeline handles language detection, transliteration, and translation automatically.</p>
            <div class="header-scripts">
                <span class="script-pill">ಕನ್ನಡ Script</span>
                <span class="script-pill">Romanized Kannada</span>
                <span class="script-pill">English</span>
                <span class="script-pill">Mixed Language</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_single, tab_batch = st.tabs(["📝  Single Review", "📊  Batch Upload"])

    with tab_single:
        _render_single_tab(backend)

    with tab_batch:
        _render_batch_tab(backend)


if __name__ == "__main__":
    main()
