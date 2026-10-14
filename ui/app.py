"""
Clinical AI Copilot & Decision Support
---------------------------------------
A multimodal Streamlit application pairing a pretrained multi-label chest
X-ray model (lung + heart findings) with a retrieval-augmented guideline
search tool, gated behind an explicit physician-approval workflow.

Dashboard supports both:
  - Chest X-ray analysis  (lung findings via TorchXRayVision DenseNet121)
  - ECG / cardiac rhythm  (heart-finding context & rate interpretation)

New features:
  - Login gate: physician credentials required before dashboard access
  - Patient Records: temporary session-based record store with table view
  - Doctor Feedback: per-finding confirmation / override panel
"""

import datetime
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as st_components

# Ensure the project root is importable regardless of the working directory
# Streamlit is launched from.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from tools.image_tool import classify_image  # noqa: E402
from tools.rag_tool import query_clinical_rag  # noqa: E402

TEMP_DIR = Path("./temp")
DEFAULT_XRAY_CANDIDATES = ("./results/xray.png", "./xray.png")

# --------------------------------------------------------------------------- #
# Auth — hardcoded physician credentials (replace with DB in production)
# --------------------------------------------------------------------------- #
_PHYSICIAN_USERS: dict[str, dict] = {
    "dr.sharma": {"password": "clinic@2025", "display": "Dr. Priya Sharma",  "dept": "Pulmonology"},
    "dr.khan":   {"password": "cardio#2025", "display": "Dr. Amir Khan",     "dept": "Cardiology"},
    "dr.jones":  {"password": "icu!2025",    "display": "Dr. Sarah Jones",   "dept": "ICU"},
    "admin":     {"password": "admin1234",   "display": "Admin User",         "dept": "Administration"},
}


# --------------------------------------------------------------------------- #
# Design tokens — clean light professional clinical palette
# --------------------------------------------------------------------------- #
COLORS = {
    # Backgrounds
    "bg_deep":      "#F1F5F9",
    "bg_card":      "#FFFFFF",
    "bg_glass":     "#FFFFFF",

    # Borders
    "border":       "#E2E8F0",
    "border_teal":  "#BFDBFE",

    # Text
    "text_primary": "#0F172A",
    "text_muted":   "#475569",
    "text_dim":     "#94A3B8",

    # Accents
    "teal":         "#2563EB",
    "teal_dark":    "#1D4ED8",
    "teal_glow":    "rgba(37,99,235,0.12)",
    "purple":       "#7C3AED",
    "purple_glow":  "rgba(124,58,237,0.12)",
    "amber":        "#D97706",

    # Status
    "success":      "#059669",
    "success_bg":   "rgba(5,150,105,0.08)",
    "danger":       "#DC2626",
    "danger_bg":    "rgba(220,38,38,0.06)",
    "info_bg":      "rgba(37,99,235,0.06)",
}


def inject_theme() -> None:
    """Apply the clean light professional clinical design system."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: {COLORS['text_primary']};
        }}

        .stApp {{
            background: {COLORS['bg_deep']};
        }}

        header[data-testid="stHeader"] {{
            background: #FFFFFF !important;
            border-bottom: 1px solid {COLORS['border']} !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        }}
        header[data-testid="stHeader"] * {{ color: {COLORS['text_muted']} !important; }}

        section[data-testid="stSidebar"] {{
            background: #FFFFFF !important;
            border-right: 1px solid {COLORS['border']};
            box-shadow: 2px 0 8px rgba(0,0,0,0.04);
        }}
        section[data-testid="stSidebar"] h4 {{
            color: {COLORS['text_primary']};
            font-weight: 700;
        }}

        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label {{
            color: {COLORS['text_muted']} !important;
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div,
        div[data-baseweb="select"] > div,
        .stNumberInput > div > div {{
            background: #FFFFFF !important;
            border: 1.5px solid {COLORS['border']} !important;
            border-radius: 8px !important;
            color: {COLORS['text_primary']} !important;
        }}
        input, textarea {{ color: {COLORS['text_primary']} !important; background: #FFFFFF !important; }}
        div[data-baseweb="select"] span {{ color: {COLORS['text_primary']} !important; }}
        .stNumberInput button {{
            background: {COLORS['bg_deep']} !important;
            border-color: {COLORS['border']} !important;
            color: {COLORS['text_primary']} !important;
        }}
        div[data-baseweb="input"]:focus-within > div,
        div[data-baseweb="textarea"]:focus-within > div,
        div[data-baseweb="select"]:focus-within > div {{
            border-color: {COLORS['teal']} !important;
            box-shadow: 0 0 0 3px {COLORS['teal_glow']} !important;
        }}

        [data-testid="stFileUploaderDropzone"] {{
            background: {COLORS['info_bg']} !important;
            border: 1.5px dashed {COLORS['border_teal']} !important;
            border-radius: 12px !important;
        }}
        [data-testid="stFileUploaderDropzone"] * {{ color: {COLORS['text_muted']} !important; }}
        [data-testid="stFileUploaderDropzone"] button {{
            background: #EFF6FF !important;
            border: 1px solid #BFDBFE !important;
            color: {COLORS['teal']} !important;
        }}

        .stCheckbox p {{ color: {COLORS['text_primary']} !important; font-weight: 500; }}
        [data-testid="stCaptionContainer"] {{ color: {COLORS['text_dim']} !important; }}
        h1, h2, h3, h4, h5, h6 {{ color: {COLORS['text_primary']}; }}

        .stButton > button {{
            background: {COLORS['teal']} !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            border: none !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            box-shadow: 0 2px 8px rgba(37,99,235,0.25) !important;
            transition: all 0.18s ease !important;
        }}
        .stButton > button:hover {{
            background: {COLORS['teal_dark']} !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 16px rgba(37,99,235,0.35) !important;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background: {COLORS['bg_deep']};
            padding: 5px;
            border-radius: 10px;
            border: 1px solid {COLORS['border']};
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 7px;
            padding: 9px 18px;
            color: {COLORS['text_muted']};
            font-weight: 500;
        }}
        .stTabs [data-baseweb="tab"] p {{ color: inherit !important; font-weight: inherit !important; }}
        .stTabs [aria-selected="true"] {{
            background: #FFFFFF !important;
            color: {COLORS['teal']} !important;
            font-weight: 700;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }}
        .stTabs [data-baseweb="tab-highlight"] {{ background-color: {COLORS['teal']} !important; }}

        /* ===== MASTHEAD ===== */
        .masthead {{
            background: linear-gradient(135deg, #1E40AF 0%, #1D4ED8 60%, #2563EB 100%);
            border-radius: 14px;
            padding: 30px 36px;
            margin-bottom: 22px;
            box-shadow: 0 4px 24px rgba(37,99,235,0.20);
        }}
        .masthead-eyebrow {{
            font-size: 0.68rem; font-weight: 700; letter-spacing: 0.15em;
            text-transform: uppercase; color: #93C5FD; margin-bottom: 6px;
        }}
        .masthead-title {{
            font-size: 1.95rem; font-weight: 800; margin: 0; letter-spacing: -0.03em;
            color: #FFFFFF;
        }}
        .masthead-subtitle {{ color: #BFDBFE; font-size: 0.92rem; margin-top: 4px; }}
        .masthead-badges {{ display: flex; gap: 10px; margin-top: 18px; flex-wrap: wrap; }}
        .masthead-badge {{
            display: flex; align-items: center; gap: 6px;
            background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.20);
            border-radius: 20px; padding: 5px 12px; font-size: 0.72rem;
            font-weight: 600; color: #DBEAFE; letter-spacing: 0.03em;
        }}
        .masthead-badge.live {{ border-color: rgba(52,211,153,0.6); color: #6EE7B7; }}
        .pulse-dot {{
            width: 7px; height: 7px; border-radius: 50%;
            background: #34D399; box-shadow: 0 0 0 3px rgba(52,211,153,0.3);
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ box-shadow: 0 0 0 3px rgba(52,211,153,0.3); }}
            50% {{ box-shadow: 0 0 0 7px rgba(52,211,153,0.06); }}
        }}

        /* ===== METRIC STRIP ===== */
        .metric-strip {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 22px; }}
        .metric-card {{
            background: {COLORS['bg_card']}; border: 1px solid {COLORS['border']};
            border-radius: 12px; padding: 16px 18px; position: relative; overflow: hidden;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        }}
        .metric-card::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }}
        .metric-card.teal::before {{ background: {COLORS['teal']}; }}
        .metric-card.purple::before {{ background: {COLORS['purple']}; }}
        .metric-card.amber::before {{ background: {COLORS['amber']}; }}
        .metric-card.green::before {{ background: {COLORS['success']}; }}
        .metric-label {{ font-size: 0.67rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: {COLORS['text_dim']}; margin-bottom: 6px; }}
        .metric-value {{ font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; font-weight: 700; color: {COLORS['text_primary']}; }}
        .metric-sub {{ font-size: 0.68rem; color: {COLORS['text_dim']}; margin-top: 3px; }}

        /* ===== VITALS RAIL ===== */
        .vitals-rail {{ border: 1px solid {COLORS['border']}; border-radius: 10px; overflow: hidden; margin-bottom: 14px; background: {COLORS['bg_deep']}; }}
        .vitals-row {{ display: flex; justify-content: space-between; align-items: center; padding: 9px 14px; border-bottom: 1px solid {COLORS['border']}; background:#fff; }}
        .vitals-row:last-child {{ border-bottom: none; }}
        .vitals-label {{ font-size: 0.70rem; text-transform: uppercase; letter-spacing: 0.06em; color: {COLORS['text_dim']}; font-weight: 600; }}
        .vitals-value {{ font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.82rem; color: {COLORS['teal']}; }}

        /* ===== SCAN TAG ===== */
        .scan-tag {{ display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: 20px; font-size: 0.70rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }}
        .scan-tag-xray {{ background: #EFF6FF; border: 1px solid #BFDBFE; color: {COLORS['teal']}; }}
        .scan-tag-ecg {{ background: #F5F3FF; border: 1px solid #DDD6FE; color: {COLORS['purple']}; }}

        /* ===== FINDINGS PANEL ===== */
        .findings-panel {{ background: {COLORS['bg_card']}; border: 1px solid {COLORS['border']}; border-radius: 12px; padding: 20px 22px; margin-top: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }}
        .findings-header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px; padding-bottom: 14px; border-bottom: 1px solid {COLORS['border']}; }}
        .findings-title {{ font-weight: 700; font-size: 0.95rem; color: {COLORS['text_primary']}; }}

        .finding-row {{ margin-bottom: 14px; }}
        .finding-meta {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }}
        .finding-name {{ font-size: 0.88rem; font-weight: 600; color: {COLORS['text_primary']}; }}
        .finding-region {{ font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-left: 8px; }}
        .region-heart {{ color: {COLORS['danger']}; }}
        .region-lung {{ color: {COLORS['teal']}; }}
        .finding-pct {{ font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; font-weight: 700; }}
        .pct-flagged {{ color: {COLORS['danger']}; }}
        .pct-normal {{ color: {COLORS['teal']}; }}
        .progress-track {{ height: 5px; border-radius: 3px; background: {COLORS['bg_deep']}; overflow: hidden; }}
        .progress-fill {{ height: 100%; border-radius: 3px; }}

        /* ===== BADGE ===== */
        .badge {{ display: inline-flex; align-items: center; gap: 5px; padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 0.78rem; letter-spacing: 0.02em; font-family: 'JetBrains Mono', monospace; }}
        .badge-normal {{ background: {COLORS['success_bg']}; color: {COLORS['success']}; border: 1px solid rgba(5,150,105,0.25); }}
        .badge-abnormal {{ background: {COLORS['danger_bg']}; color: {COLORS['danger']}; border: 1px solid rgba(220,38,38,0.20); }}

        /* ===== CONTEXT STRIP ===== */
        .context-strip {{
            background: #EFF6FF; border: 1px solid #BFDBFE;
            border-left: 4px solid {COLORS['teal']};
            border-radius: 10px; padding: 16px 20px; margin-bottom: 18px;
        }}
        .context-strip h4 {{
            margin: 0 0 4px 0; color: {COLORS['teal']}; font-size: 0.85rem;
            font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;
        }}
        .context-strip p {{ margin: 0; color: {COLORS['text_muted']}; font-size: 0.88rem; line-height: 1.6; }}

        /* ===== CITATION CARD ===== */
        .citation-card {{ background: {COLORS['bg_card']}; border: 1px solid {COLORS['border']}; border-left: 3px solid {COLORS['teal']}; border-radius: 10px; padding: 16px 20px; margin-bottom: 12px; }}
        .citation-card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .citation-label {{ font-size: 0.70rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: {COLORS['text_dim']}; }}
        .citation-match {{ font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 600; color: {COLORS['teal']}; }}
        .citation-body {{ font-size: 0.90rem; color: {COLORS['text_muted']}; line-height: 1.65; }}

        /* ===== ECG METRICS ===== */
        .ecg-metric-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 14px; }}
        .ecg-metric {{ background: {COLORS['bg_card']}; border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 12px 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
        .ecg-metric-label {{ font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: {COLORS['text_dim']}; margin-bottom: 5px; }}
        .ecg-metric-value {{ font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: {COLORS['success']}; }}
        .ecg-metric-unit {{ font-size: 0.72rem; color: {COLORS['text_dim']}; margin-left: 4px; }}

        /* ===== SECTION HEAD ===== */
        .section-head {{ display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }}
        .section-icon {{ width: 36px; height: 36px; border-radius: 9px; display: flex; align-items: center; justify-content: center; font-size: 1rem; }}
        .section-icon-teal {{ background: #EFF6FF; border: 1px solid #BFDBFE; }}
        .section-icon-purple {{ background: #F5F3FF; border: 1px solid #DDD6FE; }}
        .section-label {{ font-size: 1.05rem; font-weight: 700; color: {COLORS['text_primary']}; }}
        .section-sub {{ font-size: 0.78rem; color: {COLORS['text_dim']}; margin-top: 2px; }}

        /* ===== FOOTNOTE ===== */
        .footnote {{ font-size: 0.70rem; color: {COLORS['text_dim']}; margin-top: 8px; }}

        /* ===== SIDEBAR SECTION DIVIDER ===== */
        .sidebar-section {{
            font-size: 0.63rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.10em; color: {COLORS['text_dim']};
            margin: 14px 0 6px; padding-top: 12px;
            border-top: 1px solid {COLORS['border']};
        }}
        /* ===== LOGIN PAGE ===== */
        .login-wrap {{
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 50%, #1E40AF 100%);
        }}
        .login-card {{
            background: #FFFFFF;
            border-radius: 20px;
            padding: 48px 44px;
            width: 100%;
            max-width: 420px;
            box-shadow: 0 24px 64px rgba(0,0,0,0.30);
        }}
        .login-logo {{
            text-align: center;
            margin-bottom: 28px;
        }}
        .login-logo-icon {{
            font-size: 2.8rem;
            display: block;
            margin-bottom: 10px;
        }}
        .login-logo-title {{
            font-size: 1.5rem;
            font-weight: 800;
            color: {COLORS['text_primary']};
            letter-spacing: -0.03em;
        }}
        .login-logo-sub {{
            font-size: 0.80rem;
            color: {COLORS['text_muted']};
            margin-top: 4px;
        }}
        .login-error {{
            background: {COLORS['danger_bg']};
            border: 1px solid rgba(220,38,38,0.25);
            border-radius: 8px;
            padding: 10px 14px;
            color: {COLORS['danger']};
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 14px;
        }}

        /* ===== RECORDS TABLE ===== */
        .records-table {{
            width: 100%; border-collapse: collapse;
            font-family: 'Inter', sans-serif; font-size: 0.82rem;
        }}
        .records-table th {{
            background: {COLORS['bg_deep']};
            color: {COLORS['text_dim']};
            font-size: 0.65rem; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.08em;
            padding: 9px 14px; border-bottom: 1px solid {COLORS['border']};
            text-align: left;
        }}
        .records-table td {{
            padding: 10px 14px;
            border-bottom: 1px solid {COLORS['border']};
            color: {COLORS['text_primary']};
            vertical-align: middle;
        }}
        .records-table tr:hover td {{ background: {COLORS['bg_deep']}; }}
        .rec-badge-flagged {{
            background: {COLORS['danger_bg']}; color: {COLORS['danger']};
            border: 1px solid rgba(220,38,38,0.20);
            border-radius: 20px; padding: 2px 9px;
            font-size: 0.68rem; font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}
        .rec-badge-ok {{
            background: {COLORS['success_bg']}; color: {COLORS['success']};
            border: 1px solid rgba(5,150,105,0.20);
            border-radius: 20px; padding: 2px 9px;
            font-size: 0.68rem; font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}

        /* ===== FEEDBACK PANEL ===== */
        .feedback-card {{
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['border']};
            border-left: 4px solid {COLORS['purple']};
            border-radius: 12px;
            padding: 18px 22px;
            margin-bottom: 12px;
        }}
        .feedback-finding-name {{
            font-size: 0.90rem; font-weight: 700;
            color: {COLORS['text_primary']}; margin-bottom: 8px;
        }}
        .feedback-submitted {{
            background: {COLORS['success_bg']};
            border: 1px solid rgba(5,150,105,0.25);
            border-radius: 10px; padding: 12px 18px;
            color: {COLORS['success']}; font-weight: 600;
            font-size: 0.88rem; margin-top: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class PatientContext:
    # Identity
    name: str
    patient_id: str
    dob: str
    age: int
    gender: str
    blood_type: str
    # Clinical
    symptoms: str
    # Admission
    physician: str
    department: str
    ward: str


@dataclass
class PatientRecord:
    """A snapshot of a patient visit stored in session memory."""
    timestamp: str
    name: str
    patient_id: str
    age: int
    gender: str
    blood_type: str
    symptoms: str
    physician: str
    department: str
    ward: str
    xray_top_finding: str = "—"
    xray_flagged: int = 0
    ecg_top_finding: str = "—"
    ecg_flagged: int = 0
    feedback: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
# Scan-mode subtitle map — keyed by active scan state
_SCAN_SUBTITLES: dict[str, tuple[str, str]] = {
    "xray":    ("&#127807;",  "Chest X-Ray active &mdash; analyzing lung &amp; cardiac pathologies via DenseNet121 with evidence-based guideline retrieval"),
    "ecg":     ("&#9829;",    "ECG Cardiac active &mdash; heart rate &amp; rhythm assessment with clinical guideline context"),
    "both":    ("&#128202;",  "Multimodal session &mdash; chest X-ray &amp; ECG analyses complete; combined diagnostic view available"),
    "default": ("&#127973;",  "Multimodal diagnostics &mdash; chest X-ray imaging analysis &amp; ECG cardiac assessment with evidence-based guideline retrieval"),
}


def render_masthead(scan_mode: str = "default") -> None:
    """Render the top masthead with a dynamic subtitle that reflects the active scan type."""
    icon, subtitle = _SCAN_SUBTITLES.get(scan_mode, _SCAN_SUBTITLES["default"])
    st.markdown(
        f"""
        <div class="masthead">
            <div class="masthead-eyebrow">&#9679; Clinical Decision Support Platform</div>
            <p class="masthead-title">Clinical AI Copilot</p>
            <p class="masthead-subtitle">
                {icon}&nbsp; {subtitle}
            </p>
            <div class="masthead-badges">
                <div class="masthead-badge live">
                    <div class="pulse-dot"></div>
                    SYSTEM ONLINE
                </div>
                <div class="masthead-badge">&#128274; SESSION ENCRYPTED</div>
                <div class="masthead-badge">&#10003; HIPAA COMPLIANT</div>
                <div class="masthead-badge">&#129657; PHYSICIAN-SUPERVISED</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_strip() -> None:
    st.markdown(
        """
        <div class="metric-strip">
            <div class="metric-card teal">
                <div class="metric-label">Imaging Model</div>
                <div class="metric-value">DenseNet121</div>
                <div class="metric-sub">TorchXRayVision &middot; 18 pathologies</div>
            </div>
            <div class="metric-card purple">
                <div class="metric-label">Scan Coverage</div>
                <div class="metric-value">X-Ray + ECG</div>
                <div class="metric-sub">Heart &amp; Lung findings</div>
            </div>
            <div class="metric-card amber">
                <div class="metric-label">Guideline Index</div>
                <div class="metric-value">Vector RAG</div>
                <div class="metric-sub">Live &middot; evidence-based</div>
            </div>
            <div class="metric-card green">
                <div class="metric-label">Agent Mode</div>
                <div class="metric-value">Supervised</div>
                <div class="metric-sub">Physician approval required</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> PatientContext:
    with st.sidebar:
        # ---- Header ----
        st.markdown(
            f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;">
                <div style="width:6px;height:24px;border-radius:3px;
                            background:linear-gradient(180deg,{COLORS['teal']},{COLORS['purple']});"></div>
                <span style="font-size:0.78rem;font-weight:700;text-transform:uppercase;
                             letter-spacing:0.1em;color:{COLORS['teal']};">Patient Record</span>
            </div>""",
            unsafe_allow_html=True,
        )

        # ---- Section: Demographics ----
        st.markdown(
            f"""<div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                           letter-spacing:0.1em;color:{COLORS['text_dim']};margin:10px 0 6px;">Demographics</div>""",
            unsafe_allow_html=True,
        )
        name = st.text_input("Full Name", "John Doe", key="pt_name")
        col_id, col_dob = st.columns(2)
        with col_id:
            patient_id = st.text_input("MRN / Patient ID", "MRN-1042", key="pt_id")
        with col_dob:
            dob = st.text_input("Date of Birth", "1979-03-15", key="pt_dob")
        col_age, col_gender, col_blood = st.columns(3)
        with col_age:
            age = st.number_input("Age", min_value=0, max_value=120, value=45, key="pt_age")
        with col_gender:
            gender = st.selectbox("Sex", ["M", "F", "Other"], key="pt_gender")
        with col_blood:
            blood_type = st.selectbox("Blood", ["—", "A+", "A−", "B+", "B−", "AB+", "AB−", "O+", "O−"], key="pt_blood")

        # ---- Section: Clinical Presentation ----
        st.markdown(
            f"""<div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                           letter-spacing:0.1em;color:{COLORS['text_dim']};margin:14px 0 6px;">Clinical Presentation</div>""",
            unsafe_allow_html=True,
        )
        symptoms = st.text_area(
            "Chief Complaint / History",
            "Fever, persistent cough, dyspnea on exertion for 3 days.",
            height=90,
            key="pt_symptoms",
        )

        # ---- Section: Admission ----
        st.markdown(
            f"""<div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                           letter-spacing:0.1em;color:{COLORS['text_dim']};margin:14px 0 6px;">Admission</div>""",
            unsafe_allow_html=True,
        )
        _default_physician = st.session_state.get("logged_in_user", "Dr. Priya Sharma")
        physician = st.text_input("Attending Physician", _default_physician, key="pt_physician")
        col_dept, col_ward = st.columns(2)
        with col_dept:
            department = st.selectbox(
                "Department",
                ["Pulmonology", "Cardiology", "Emergency", "ICU", "General Medicine", "Radiology"],
                key="pt_dept",
            )
        with col_ward:
            ward = st.text_input("Ward / Room", "4B-12", key="pt_ward")

        # ---- Live Summary Card ----
        st.markdown(
            f"""
            <div class="vitals-rail" style="margin-top:16px;">
                <div class="vitals-row">
                    <span class="vitals-label">Patient</span>
                    <span class="vitals-value" style="font-size:0.80rem;">{name}</span>
                </div>
                <div class="vitals-row">
                    <span class="vitals-label">MRN</span>
                    <span class="vitals-value">{patient_id}</span>
                </div>
                <div class="vitals-row">
                    <span class="vitals-label">Age / Sex / Blood</span>
                    <span class="vitals-value">{age} / {gender[0]} / {blood_type}</span>
                </div>
                <div class="vitals-row">
                    <span class="vitals-label">Physician</span>
                    <span class="vitals-value" style="font-size:0.75rem;">{physician}</span>
                </div>
                <div class="vitals-row">
                    <span class="vitals-label">Dept &middot; Ward</span>
                    <span class="vitals-value">{department} &middot; {ward}</span>
                </div>
                <div class="vitals-row">
                    <span class="vitals-label">Active Scans</span>
                    <span class="vitals-value">X-Ray &middot; ECG</span>
                </div>
            </div>
            <div class="footnote" style="margin-top:6px;">
                &#128274; Data stays local to this session
            </div>
            """,
            unsafe_allow_html=True,
        )

    return PatientContext(
        name=name, patient_id=patient_id, dob=dob, age=age, gender=gender,
        blood_type=blood_type, symptoms=symptoms,
        physician=physician, department=department, ward=ward,
    )


def _findings_html(findings: list, title: str, accent: str = "#2563EB") -> str:
    """Build a fully self-contained HTML block for a findings list.
    Uses only inline styles so it renders correctly inside st.components.v1.html()
    (which runs in a sandboxed iframe — external CSS classes are not available)."""
    if not findings:
        return "<p style='color:#64748B;font-family:Inter,sans-serif;'>No findings returned.</p>"

    top_flagged = any(f["flagged"] for f in findings)
    badge_bg   = "rgba(220,38,38,0.08)"   if top_flagged else "rgba(5,150,105,0.08)"
    badge_clr  = "#DC2626"                 if top_flagged else "#059669"
    badge_bdr  = "rgba(220,38,38,0.25)"   if top_flagged else "rgba(5,150,105,0.25)"
    badge_icon = "&#9888;" if top_flagged else "&#10003;"
    badge_text = findings[0]["label"] if top_flagged else "No Significant Findings"

    rows_html = ""
    for f in findings[:8]:
        is_heart = f["region"] == "heart"
        region_icon  = "&#9829;" if is_heart else "&#127129;"
        region_label = "Heart"   if is_heart else "Lung"
        region_clr   = "#DC2626" if is_heart else "#2563EB"
        bar_clr      = "#DC2626" if f["flagged"] else accent
        pct_clr      = "#DC2626" if f["flagged"] else "#2563EB"
        pct          = f["probability"]
        rows_html += f"""
        <div style="margin-bottom:14px;">
          <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;">
            <span style="font-family:Inter,sans-serif;">
              <span style="font-size:0.88rem;font-weight:600;color:#0F172A;">{f['label']}</span>
              <span style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                           letter-spacing:0.06em;color:{region_clr};margin-left:8px;">
                {region_icon} {region_label}
              </span>
            </span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;
                          font-weight:700;color:{pct_clr};">{pct:.1f}%</span>
          </div>
          <div style="height:5px;border-radius:3px;background:#F1F5F9;overflow:hidden;">
            <div style="height:100%;width:{min(pct,100):.1f}%;background:{bar_clr};border-radius:3px;"></div>
          </div>
        </div>
        """

    return f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
      body {{ margin:0; padding:0; background:transparent; }}
    </style>
    <div style="background:#FFFFFF;border:1px solid #E2E8F0;
                border-radius:12px;padding:20px 22px;
                font-family:Inter,sans-serif;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);">
      <div style="display:flex;align-items:center;justify-content:space-between;
                  margin-bottom:18px;padding-bottom:14px;
                  border-bottom:1px solid #E2E8F0;">
        <span style="font-weight:700;font-size:0.95rem;color:#0F172A;">
          &#128203; {title}
        </span>
        <span style="display:inline-flex;align-items:center;gap:5px;padding:4px 12px;
                      border-radius:20px;font-weight:700;font-size:0.78rem;
                      font-family:'JetBrains Mono',monospace;
                      background:{badge_bg};color:{badge_clr};
                      border:1px solid {badge_bdr};">
          {badge_icon} {badge_text}
        </span>
      </div>
      {rows_html}
      <div style="font-size:0.70rem;color:#94A3B8;margin-top:14px;
                  border-top:1px solid #F1F5F9;padding-top:12px;">
        &#9888; Model output is decision support only and does not constitute a clinical diagnosis.
      </div>
    </div>
    """


def render_findings_panel(result: dict, title: str = "Clinical Findings &mdash; Chest X-Ray",
                          accent: str = "#00D2BE") -> None:
    """Render a findings panel using st.components.v1.html() to bypass Streamlit's
    HTML sanitizer, which strips complex nested divs from st.markdown()."""
    findings = result.get("findings", [])
    html = _findings_html(findings, title, accent)
    # Height: ~90px header + ~60px per row + 60px footer
    height = 140 + min(len(findings), 8) * 62
    st_components.html(html, height=height, scrolling=False)


def save_uploaded_file(uploaded_file: Any) -> Path:
    """Persist an uploaded file to disk and return its path."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    destination = TEMP_DIR / uploaded_file.name
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def find_default_xray() -> str | None:
    for candidate in DEFAULT_XRAY_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None



# --------------------------------------------------------------------------- #
# Dynamic description helpers — derive per-scan context from actual findings
# --------------------------------------------------------------------------- #
def _generate_scan_description(result: dict, scan_type: str = "X-Ray") -> str:
    """Build a concise clinical summary sentence from classify_image() output.
    The text changes with every new image because it reflects the actual
    flagged findings and confidence scores from *that* specific scan."""
    findings = result.get("findings", [])
    flagged  = [f for f in findings if f["flagged"]]
    top      = result.get("top_finding", "Unknown")
    top_conf = result.get("top_confidence", 0.0)

    if not flagged:
        return (
            f"{scan_type} analysis complete. No significant pathologies flagged above "
            f"decision threshold. Top model output: {top} ({top_conf:.1f}% confidence). "
            f"Recommend clinical correlation with patient presentation."
        )

    # Group by region
    heart_flags = [f for f in flagged if f["region"] == "heart"]
    lung_flags  = [f for f in flagged if f["region"] == "lung"]

    parts = []
    if lung_flags:
        names = ", ".join(f"{f['label']} ({f['probability']:.1f}%)" for f in lung_flags[:3])
        parts.append(f"Pulmonary findings: {names}")
    if heart_flags:
        names = ", ".join(f"{f['label']} ({f['probability']:.1f}%)" for f in heart_flags[:2])
        parts.append(f"Cardiac findings: {names}")

    summary = "; ".join(parts)
    return (
        f"{scan_type} analysis flagged {len(flagged)} finding(s) above threshold. "
        f"{summary}. Primary concern: {top} at {top_conf:.1f}% confidence. "
        f"Immediate physician review recommended."
    )


def _generate_guideline_query(result: dict, patient_symptoms: str = "") -> str:
    """Generate a targeted clinical guideline search query from scan findings.
    Incorporates the top flagged pathologies so every image produces a
    distinct, relevant query instead of the generic default."""
    findings = result.get("findings", [])
    flagged  = [f for f in findings if f["flagged"]]
    top      = result.get("top_finding", "")

    if not flagged:
        base = f"Clinical management guidelines for {top}" if top else "General chest imaging guidelines"
        return base + (f" in patient with {patient_symptoms}" if patient_symptoms else "")

    # Use top-3 flagged findings as query anchors
    top_flagged_names = ", ".join(f["label"] for f in flagged[:3])
    query = f"Treatment and management guidelines for {top_flagged_names}"
    if patient_symptoms:
        query += f" in patient presenting with {patient_symptoms}"
    return query


# --------------------------------------------------------------------------- #
# File-change callbacks — Streamlit's on_change fires before the next render,
# reliably clearing stale results regardless of filename or file size.
# --------------------------------------------------------------------------- #
def _on_xray_change() -> None:
    """Called by Streamlit when the X-ray uploader widget changes."""
    st.session_state.pop("xray_result", None)
    st.session_state.pop("xray_image_path", None)


def _on_ecg_change() -> None:
    """Called by Streamlit when the ECG uploader widget changes."""
    st.session_state.pop("ecg_result", None)
    st.session_state.pop("ecg_image_path", None)


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def render_radiology_tab() -> None:
    st.markdown(
        """
        <div class="section-head">
            <div class="section-icon section-icon-teal">&#128247;</div>
            <div>
                <div class="section-label">Chest Radiograph Diagnostics</div>
                <div class="section-sub">Upload a chest X-ray for AI-powered lung &amp; heart finding analysis</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a chest X-ray (PNG / JPG)",
        type=["png", "jpg", "jpeg"],
        key="xray_upload",
        on_change=_on_xray_change,   # fires on every file swap — clears stale result
    )

    if uploaded_file is None:
        st.markdown(
            f"""
            <div style="background:{COLORS['info_bg']}; border:1px dashed {COLORS['border_teal']};
                        border-radius:12px; padding:32px; text-align:center; margin-top:8px;">
                <div style="font-size:2rem; margin-bottom:8px;">&#127807;</div>
                <div style="color:{COLORS['text_muted']}; font-size:0.9rem; font-weight:500;">
                    Upload a PNG or JPG radiograph to begin analysis
                </div>
                <div style="color:{COLORS['text_dim']}; font-size:0.78rem; margin-top:4px;">
                    Supports: Chest PA / Lateral views
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state.pop("xray_result", None)
        st.session_state.pop("xray_image_path", None)
        return

    image_path = save_uploaded_file(uploaded_file)
    # Persist path for the Agent Workflow tab
    st.session_state["xray_image_path"] = str(image_path)
    col_original, col_heatmap = st.columns(2)

    with col_original:
        st.markdown(
            """<div class="scan-tag scan-tag-xray" style="margin-bottom:8px;">
               &#127807; Chest X-Ray &middot; Original</div>""",
            unsafe_allow_html=True,
        )
        st.image(str(image_path), use_container_width=True)

    if st.button("&#128269;  Run Diagnostic Analysis", key="run_xray"):
        with st.spinner("Analyzing radiograph for lung and heart findings..."):
            try:
                result = classify_image(str(image_path))
                if "error" not in result:
                    st.session_state["xray_result"] = result
                else:
                    st.error(result["error"])
                    st.session_state.pop("xray_result", None)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.session_state.pop("xray_result", None)

    # Render from session_state — persists across widget interactions, clears on new file
    if "xray_result" in st.session_state:
        result = st.session_state["xray_result"]
        with col_heatmap:
            st.markdown(
                f"""<div class="scan-tag scan-tag-xray" style="margin-bottom:8px;">
                    &#128293; Grad-CAM &middot; {result['top_finding']}</div>""",
                unsafe_allow_html=True,
            )
            st.image(result["heatmap_path"], use_container_width=True)
        render_findings_panel(result, title="Clinical Findings &mdash; Chest X-Ray", accent="#2563EB")

        # --- Dynamic per-scan description (changes with every image) ---
        scan_desc = _generate_scan_description(result, scan_type="Chest X-Ray")
        flagged_count = sum(1 for f in result.get("findings", []) if f["flagged"])
        desc_color = COLORS["danger"] if flagged_count > 0 else COLORS["success"]
        desc_icon  = "&#9888;" if flagged_count > 0 else "&#10003;"
        st.markdown(
            f"""
            <div class="context-strip" style="margin-top:16px;">
                <h4 style="color:{desc_color};">{desc_icon}&nbsp; AI Interpretation &mdash; Chest X-Ray</h4>
                <p>{scan_desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_ecg_tab() -> None:
    st.markdown(
        """
        <div class="section-head">
            <div class="section-icon section-icon-purple">&#9829;</div>
            <div>
                <div class="section-label">ECG / Cardiac Rhythm Analysis</div>
                <div class="section-sub">Upload an ECG strip for heart rate &amp; rhythm assessment</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_ecg = st.file_uploader(
        "Upload an ECG strip image (PNG / JPG)",
        type=["png", "jpg", "jpeg"],
        key="ecg_upload",
        on_change=_on_ecg_change,    # fires on every file swap — clears stale result
    )

    if uploaded_ecg is None:
        st.markdown(
            f"""
            <div style="background:#F5F3FF; border:1px dashed #DDD6FE;
                        border-radius:12px; padding:32px; text-align:center; margin-top:8px;">
                <div style="font-size:2rem; margin-bottom:8px;">&#9829;</div>
                <div style="color:{COLORS['text_muted']}; font-size:0.9rem; font-weight:500;">
                    Upload an ECG strip to begin cardiac analysis
                </div>
                <div style="color:{COLORS['text_dim']}; font-size:0.78rem; margin-top:4px;">
                    Supports: 12-lead ECG, single-lead rhythm strips
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state.pop("ecg_result", None)
        return

    ecg_path = save_uploaded_file(uploaded_ecg)
    # Persist path so the Agent Workflow tab can access it
    st.session_state["ecg_image_path"] = str(ecg_path)

    col_ecg, col_metrics = st.columns([3, 2])
    with col_ecg:
        st.markdown(
            """<div class="scan-tag scan-tag-ecg" style="margin-bottom:8px;">
               &#9829; ECG Strip &middot; Uploaded</div>""",
            unsafe_allow_html=True,
        )
        st.image(str(ecg_path), use_container_width=True)

    if st.button("&#128209;  Analyze ECG", key="run_ecg"):
        with st.spinner("Processing ECG for cardiac findings..."):
            try:
                result = classify_image(str(ecg_path))
                if "error" not in result:
                    st.session_state["ecg_result"] = result
                else:
                    st.error(result["error"])
                    st.session_state.pop("ecg_result", None)
            except Exception as exc:
                st.error(f"ECG analysis failed: {exc}")
                st.session_state.pop("ecg_result", None)

    if "ecg_result" in st.session_state:
        result = st.session_state["ecg_result"]
        findings = result.get("findings", [])
        heart_findings = [f for f in findings if f["region"] == "heart"]

        top_flagged = any(f["flagged"] for f in heart_findings) if heart_findings else False
        badge_class = "badge-abnormal" if top_flagged else "badge-normal"
        badge_text = "Cardiac Findings Detected" if top_flagged else "No Significant Cardiac Findings"
        badge_icon = "&#9888;" if top_flagged else "&#10003;"

        with col_metrics:
            flagged_count = sum(1 for f in findings if f["flagged"])
            heart_flagged = sum(1 for f in heart_findings if f["flagged"])
            top_conf = result.get("top_confidence", 0)
            top_finding = result.get("top_finding", "N/A")
            metric_html = f"""
            <style>
              @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
              body {{ margin:0;padding:0;background:transparent; }}
            </style>
            <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;font-family:Inter,sans-serif;">
              <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:12px 14px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#94A3B8;margin-bottom:5px;">Top Confidence</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;font-weight:700;color:#059669;">{top_conf:.1f}<span style="font-size:0.72rem;color:#94A3B8;margin-left:4px;">%</span></div>
              </div>
              <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:12px 14px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#94A3B8;margin-bottom:5px;">Findings Flagged</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;font-weight:700;color:#059669;">{flagged_count}<span style="font-size:0.72rem;color:#94A3B8;margin-left:4px;">total</span></div>
              </div>
              <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:12px 14px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#94A3B8;margin-bottom:5px;">Cardiac Flags</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;font-weight:700;color:#059669;">{heart_flagged}<span style="font-size:0.72rem;color:#94A3B8;margin-left:4px;">findings</span></div>
              </div>
              <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:12px 14px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#94A3B8;margin-bottom:5px;">Top Finding</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;font-weight:700;color:#059669;">{top_finding}</div>
              </div>
            </div>
            """
            st_components.html(metric_html, height=160, scrolling=False)

        # Reuse _findings_html with purple accent for cardiac theme
        ecg_html = _findings_html(findings, "Cardiac &amp; Chest Findings &mdash; ECG Analysis", accent="#7C5CFC")
        height = 140 + min(len(findings), 8) * 62
        st_components.html(ecg_html, height=height, scrolling=False)

        # --- Dynamic per-scan description (changes with every ECG image) ---
        ecg_desc  = _generate_scan_description(result, scan_type="ECG / Cardiac")
        heart_flagged = sum(1 for f in heart_findings if f["flagged"])
        desc_color = COLORS["danger"] if heart_flagged > 0 else COLORS["success"]
        desc_icon  = "&#9888;" if heart_flagged > 0 else "&#10003;"
        st.markdown(
            f"""
            <div class="context-strip" style="margin-top:16px;
                border-left-color:{COLORS['purple']};background:#F5F3FF;">
                <h4 style="color:{desc_color};">{desc_icon}&nbsp; AI Interpretation &mdash; ECG / Cardiac</h4>
                <p>{ecg_desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Updated pattern: capture the full bracket content (source + score) as one group
_CITATION_PATTERN = re.compile(
    r"--- Citation (\d+) \[(.*?), ([\d.]+)% match\] ---\n(.*?)(?=\n\n--- Citation|\Z)",
    re.DOTALL,
)


def _relevance_label(match_pct_str: str) -> tuple[str, str, str]:
    """Convert a raw Chroma relevance % (which can be 0 when distance > 1.0)
    into a human-readable tier with colour coding.
    Chroma cosine distance → relevance = (1 - distance) * 100.
    Distances > 1 produce 0%; distance 0 = 100%.  Bucket into tiers.
    """
    try:
        pct = float(match_pct_str)
    except ValueError:
        return "Relevant", "#00D2BE", "rgba(0,210,190,0.15)"
    if pct >= 60:
        return f"{pct:.0f}% match — High",  "#10E898", "rgba(16,232,152,0.12)"
    if pct >= 30:
        return f"{pct:.0f}% match — Medium", "#FFB547", "rgba(255,181,71,0.12)"
    if pct > 0:
        return f"{pct:.0f}% match — Low",    "#8BA3C7", "rgba(139,163,199,0.10)"
    # 0% means Chroma distance was ≥ 1.0 — still a retrieved passage, just distant
    return "Retrieved — Low similarity", "#8BA3C7", "rgba(139,163,199,0.10)"


def render_citations_panel(raw_text: str) -> None:
    """Parses rag_tool.py's plain-text citation format and renders styled cards
    via st.components.v1.html() to avoid Streamlit's HTML sanitizer."""
    matches = _CITATION_PATTERN.findall(raw_text)
    if not matches:
        # Plain error/no-results message
        st_components.html(
            f"""<p style='color:#8BA3C7;font-family:Inter,sans-serif;
                         font-size:0.9rem;padding:8px 0;'>{raw_text}</p>""",
            height=60,
        )
        return

    cards_html = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
      body {{ margin:0;padding:0;background:transparent; }}
    </style>
    <div style="font-size:0.80rem;color:#94A3B8;margin-bottom:14px;font-weight:600;
                font-family:Inter,sans-serif;">
      &#128269; {len(matches)} relevant passage(s) retrieved from guideline index
    </div>
    """

    for idx, source, match_pct, content in matches:
        label, clr, bg = _relevance_label(match_pct)
        # Shorten long source paths to just the filename
        src_display = Path(source).name if source else "Unknown"
        cards_html += f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;
                    border-left:3px solid #2563EB;border-radius:10px;
                    padding:16px 20px;margin-bottom:12px;font-family:Inter,sans-serif;
                    box-shadow:0 1px 4px rgba(0,0,0,0.05);">
          <div style="display:flex;justify-content:space-between;align-items:center;
                      margin-bottom:10px;">
            <span style="font-size:0.70rem;font-weight:700;text-transform:uppercase;
                          letter-spacing:0.06em;color:#94A3B8;">
              Citation {idx} &middot; {src_display}
            </span>
            <span style="font-size:0.72rem;font-weight:600;font-family:'JetBrains Mono',monospace;
                          background:{bg};color:{clr};padding:2px 10px;
                          border-radius:20px;border:1px solid {clr + '44'};">
              {label}
            </span>
          </div>
          <div style="font-size:0.90rem;color:#475569;line-height:1.65;">
            {content.strip()}
          </div>
        </div>
        """

    total_height = 60 + len(matches) * 160
    st_components.html(cards_html, height=total_height, scrolling=False)


def render_guidelines_tab() -> None:
    st.markdown(
        """
        <div class="section-head">
            <div class="section-icon section-icon-teal">&#128218;</div>
            <div>
                <div class="section-label">Evidence-Based Literature Search</div>
                <div class="section-sub">Query clinical guidelines &amp; treatment protocols via vector RAG</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Auto-generate a scan-specific query whenever a new result is available
    _xray_result = st.session_state.get("xray_result")
    _ecg_result  = st.session_state.get("ecg_result")
    _latest_result = _xray_result or _ecg_result  # prefer X-ray; fall back to ECG

    _default_query = "Empiric outpatient treatment guidelines for pneumonia"
    if _latest_result:
        # Key the auto-query on the top finding so it only changes when results change
        _auto_query = _generate_guideline_query(_latest_result)
        _query_key  = f"guideline_query_{_latest_result.get('top_finding', '')}"
    else:
        _auto_query = _default_query
        _query_key  = "guideline_query_default"

    # Show a banner when the query was auto-populated from scan findings
    if _latest_result:
        flagged = [f for f in _latest_result.get("findings", []) if f["flagged"]]
        top_names = ", ".join(f["label"] for f in flagged[:3]) if flagged else _latest_result.get("top_finding", "")
        st.markdown(
            f"""
            <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-left:4px solid {COLORS['teal']};
                        border-radius:8px;padding:10px 16px;margin-bottom:12px;font-size:0.82rem;
                        color:{COLORS['text_muted']};font-family:Inter,sans-serif;">
                <strong style="color:{COLORS['teal']};">⚡ Auto-populated from latest scan findings:</strong>
                &nbsp;{top_names}
            </div>
            """,
            unsafe_allow_html=True,
        )

    query = st.text_input(
        "Search treatment protocols & guidelines",
        value=_auto_query,
        key=_query_key,
    )

    if st.button("&#128269;  Search Knowledge Base", key="search_guidelines"):
        with st.spinner("Querying the clinical guideline index..."):
            try:
                response = query_clinical_rag(query)
                st.session_state["guideline_result"] = response
            except Exception as exc:
                st.error(f"Search failed: {exc}")

    if "guideline_result" in st.session_state:
        render_citations_panel(st.session_state["guideline_result"])


def render_agent_tab(patient: PatientContext) -> None:
    st.markdown(
        """
        <div class="section-head">
            <div class="section-icon section-icon-teal">&#129302;</div>
            <div>
                <div class="section-label">Physician Control &amp; Copilot Execution</div>
                <div class="section-sub">Autonomous diagnostic pipeline &mdash; requires explicit physician authorization</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Build a dynamic context strip that reflects actual scan findings
    _xray_res = st.session_state.get("xray_result")
    _ecg_res  = st.session_state.get("ecg_result")

    # Generate scan-specific descriptions for agent context
    _xray_desc = _generate_scan_description(_xray_res, "Chest X-Ray") if _xray_res else None
    _ecg_desc  = _generate_scan_description(_ecg_res,  "ECG / Cardiac") if _ecg_res else None

    _scan_summary_html = ""
    if _xray_desc:
        _xf = sum(1 for f in _xray_res.get("findings", []) if f["flagged"])
        _ic = "&#9888;" if _xf > 0 else "&#10003;"
        _cl = COLORS["danger"] if _xf > 0 else COLORS["success"]
        _scan_summary_html += (
            f"<div style='margin-top:10px;padding:10px 12px;border-radius:8px;"
            f"background:rgba(37,99,235,0.06);border:1px solid #BFDBFE;'>"
            f"<span style='font-size:0.72rem;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:0.06em;color:{COLORS['teal']};'>&#127807; X-Ray</span>"
            f"<br><span style='font-size:0.85rem;color:{_cl};font-weight:600;'>{_ic}&nbsp;</span>"
            f"<span style='font-size:0.85rem;color:{COLORS['text_muted']};'>{_xray_desc}</span></div>"
        )
    if _ecg_desc:
        _hf = sum(1 for f in _ecg_res.get("findings", []) if f["flagged"] and f["region"] == "heart")
        _ic = "&#9888;" if _hf > 0 else "&#10003;"
        _cl = COLORS["danger"] if _hf > 0 else COLORS["success"]
        _scan_summary_html += (
            f"<div style='margin-top:10px;padding:10px 12px;border-radius:8px;"
            f"background:rgba(124,58,237,0.06);border:1px solid #DDD6FE;'>"
            f"<span style='font-size:0.72rem;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:0.06em;color:{COLORS['purple']};'>&#9829; ECG / Cardiac</span>"
            f"<br><span style='font-size:0.85rem;color:{_cl};font-weight:600;'>{_ic}&nbsp;</span>"
            f"<span style='font-size:0.85rem;color:{COLORS['text_muted']};'>{_ecg_desc}</span></div>"
        )

    if not _scan_summary_html:
        _scan_summary_html = (
            f"<div style='margin-top:10px;color:{COLORS['text_dim']};font-size:0.85rem;font-style:italic;'>"
            f"No scans completed yet. Run analysis in the X-Ray or ECG tabs to populate AI findings here."
            f"</div>"
        )

    st.markdown(
        f"""
        <div class="context-strip">
            <h4>Patient Context &middot; {patient.patient_id}</h4>
            <p>
                <strong style="color:#1E40AF;">{patient.name}</strong>
                &nbsp;&middot;&nbsp; DOB: {patient.dob}
                &nbsp;&middot;&nbsp; Age {patient.age} &middot; {patient.gender[0]} &middot; {patient.blood_type}
                <br>
                &#9673; {patient.physician} &nbsp;&middot;&nbsp; {patient.department} &nbsp;&middot;&nbsp; Ward {patient.ward}
                <br><br>
                <em>{patient.symptoms}</em>
            </p>
            {_scan_summary_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="background:{COLORS['danger_bg']}; border:1px solid rgba(255,77,106,0.25);
                    border-radius:10px; padding:14px 18px; margin-bottom:16px;">
            <span style="font-size:0.78rem; font-weight:700; text-transform:uppercase;
                          letter-spacing:0.06em; color:{COLORS['danger']};">
                &#9888; Safety Checkpoint
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    approved = st.checkbox(
        "I authorize the AI copilot to run the autonomous diagnostic pipeline."
    )

    if not st.button("&#9654;  Execute Copilot Agent", key="execute_agent"):
        return

    if not approved:
        st.error("&#9888;  Execution halted: physician authorization is required.")
        return

    st.success("&#10003;  Physician approval granted. Processing full diagnostic pipeline...")

    # Collect whichever images the physician uploaded
    agent_xray  = st.session_state.get("xray_image_path") or find_default_xray()
    agent_ecg   = st.session_state.get("ecg_image_path")

    with st.spinner("Running full diagnostic pipeline (X-Ray + ECG + Guidelines)..."):
        # --- X-Ray analysis ---
        try:
            xray_result = (
                classify_image(agent_xray)
                if agent_xray
                else {"error": "No X-ray uploaded. Go to the X-Ray Analysis tab and upload an image first."}
            )
        except Exception as exc:
            xray_result = {"error": str(exc)}

        # --- ECG analysis ---
        try:
            ecg_result = (
                classify_image(agent_ecg)
                if agent_ecg
                else {"error": "No ECG uploaded. Go to the ECG Cardiac tab and upload an image first."}
            )
        except Exception as exc:
            ecg_result = {"error": str(exc)}

        # --- Build guideline query from combined flagged findings of both scans ---
        _combined_findings: list = []
        if "error" not in xray_result:
            _combined_findings += [f for f in xray_result.get("findings", []) if f["flagged"]]
        if "error" not in ecg_result:
            _combined_findings += [f for f in ecg_result.get("findings", []) if f["flagged"]]

        if _combined_findings:
            _top_names = ", ".join(f["label"] for f in _combined_findings[:4])
            _guideline_query = (
                f"Treatment and management guidelines for {_top_names}"
                + (f" in patient presenting with {patient.symptoms}" if patient.symptoms else "")
            )
        else:
            _guideline_query = patient.symptoms or "General clinical management guidelines"

        try:
            guideline_result = query_clinical_rag(_guideline_query)
        except Exception as exc:
            guideline_result = f"Guideline lookup failed: {exc}"

    # ---- Display: X-Ray | ECG | Guidelines ----
    col_xray, col_ecg, col_guideline = st.columns(3)

    with col_xray:
        st.markdown(
            """<div class="scan-tag scan-tag-xray" style="margin-bottom:8px;">
               &#127807; X-Ray Findings</div>""",
            unsafe_allow_html=True,
        )
        if "error" in xray_result:
            st.warning(xray_result["error"])
        else:
            render_findings_panel(
                xray_result,
                title="X-Ray &mdash; Agent Pipeline",
                accent="#2563EB",
            )
            # Dynamic X-ray description
            _xd = _generate_scan_description(xray_result, "Chest X-Ray")
            _xf = sum(1 for f in xray_result.get("findings", []) if f["flagged"])
            _xc = COLORS["danger"] if _xf > 0 else COLORS["success"]
            st.markdown(
                f"""
                <div class="context-strip" style="margin-top:12px;font-size:0.82rem;">
                    <h4 style="color:{_xc};">{'&#9888;' if _xf > 0 else '&#10003;'}&nbsp; X-Ray Interpretation</h4>
                    <p style="font-size:0.82rem;">{_xd}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_ecg:
        st.markdown(
            f"""<div class="scan-tag scan-tag-ecg" style="margin-bottom:8px;">
               &#9829; ECG Findings</div>""",
            unsafe_allow_html=True,
        )
        if "error" in ecg_result:
            st.warning(ecg_result["error"])
        else:
            render_findings_panel(
                ecg_result,
                title="ECG &mdash; Agent Pipeline",
                accent="#7C3AED",
            )
            # Dynamic ECG description
            _ed = _generate_scan_description(ecg_result, "ECG / Cardiac")
            _hf = sum(1 for f in ecg_result.get("findings", [])
                      if f["flagged"] and f["region"] == "heart")
            _ec = COLORS["danger"] if _hf > 0 else COLORS["success"]
            st.markdown(
                f"""
                <div class="context-strip" style="margin-top:12px;font-size:0.82rem;
                    border-left-color:{COLORS['purple']};background:#F5F3FF;">
                    <h4 style="color:{_ec};">{'&#9888;' if _hf > 0 else '&#10003;'}&nbsp; ECG Interpretation</h4>
                    <p style="font-size:0.82rem;">{_ed}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_guideline:
        st.markdown(
            """<div class="scan-tag scan-tag-xray" style="margin-bottom:8px;">
               &#128218; Evidence-Based Recommendation</div>""",
            unsafe_allow_html=True,
        )
        st.caption(f"Query: *{_guideline_query[:120]}{'...' if len(_guideline_query) > 120 else ''}*")
        render_citations_panel(guideline_result)


# --------------------------------------------------------------------------- #
def render_combined_tab() -> None:
    """Side-by-side view of X-Ray and ECG results when both are available."""
    st.markdown(
        """
        <div class="section-head">
            <div class="section-icon section-icon-teal">&#128202;</div>
            <div>
                <div class="section-label">Combined Diagnostic Results</div>
                <div class="section-sub">X-Ray and ECG findings displayed together for comparison</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    has_xray = "xray_result" in st.session_state
    has_ecg  = "ecg_result"  in st.session_state

    if not has_xray and not has_ecg:
        st.markdown(
            f"""
            <div style="background:{COLORS['info_bg']};border:1px solid {COLORS['border_teal']};
                        border-radius:12px;padding:36px;text-align:center;">
                <div style="font-size:2rem;margin-bottom:12px;">&#128202;</div>
                <div style="font-size:1rem;font-weight:600;color:{COLORS['text_primary']};margin-bottom:6px;">
                    No analyses completed yet
                </div>
                <div style="font-size:0.88rem;color:{COLORS['text_muted']};">
                    Go to the <strong>X-Ray Analysis</strong> and <strong>ECG Cardiac</strong> tabs,
                    upload images, and run analysis to see combined results here.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    col_xray, col_divider, col_ecg = st.columns([10, 1, 10])

    # ---- X-Ray column ----
    with col_xray:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;
                        padding-bottom:10px;border-bottom:2px solid {COLORS['teal']};">
              <span style="font-size:0.70rem;font-weight:700;text-transform:uppercase;
                            letter-spacing:0.10em;color:{COLORS['teal']};">
                &#127807; Chest X-Ray Findings
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if has_xray:
            render_findings_panel(
                st.session_state["xray_result"],
                title="X-Ray Analysis",
                accent="#2563EB",
            )
        else:
            st.markdown(
                f"""<div style="background:{COLORS['bg_deep']};border:1px dashed {COLORS['border']};
                             border-radius:10px;padding:28px;text-align:center;color:{COLORS['text_dim']};font-size:0.88rem;">
                    Upload an X-ray and run analysis to see findings here.
                </div>""",
                unsafe_allow_html=True,
            )

    # ---- Thin divider ----
    with col_divider:
        st.markdown(
            """<div style="width:1px;background:#E2E8F0;min-height:400px;margin:0 auto;"></div>""",
            unsafe_allow_html=True,
        )

    # ---- ECG column ----
    with col_ecg:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;
                        padding-bottom:10px;border-bottom:2px solid {COLORS['purple']};">
              <span style="font-size:0.70rem;font-weight:700;text-transform:uppercase;
                            letter-spacing:0.10em;color:{COLORS['purple']};">
                &#9829; ECG Cardiac Findings
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if has_ecg:
            render_findings_panel(
                st.session_state["ecg_result"],
                title="ECG Analysis",
                accent="#7C3AED",
            )
        else:
            st.markdown(
                f"""<div style="background:{COLORS['bg_deep']};border:1px dashed {COLORS['border']};
                             border-radius:10px;padding:28px;text-align:center;color:{COLORS['text_dim']};font-size:0.88rem;">
                    Upload an ECG strip and run analysis to see findings here.
                </div>""",
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------- #
# Login page
# --------------------------------------------------------------------------- #
def render_login_page() -> None:
    """Full-screen login gate — CSS overrides Streamlit's layout completely."""

    # --- Full-page style override ---
    st.markdown(
        """
        <style>
        /* Hide Streamlit chrome on login page */
        header[data-testid="stHeader"]          { display: none !important; }
        section[data-testid="stSidebar"]        { display: none !important; }
        #MainMenu, footer, .stDeployButton      { display: none !important; }
        [data-testid="stToolbar"]               { display: none !important; }

        /* Full-screen gradient background */
        .stApp {
            background: linear-gradient(135deg, #0A0F1E 0%, #0F2044 40%, #1a3566 70%, #1E40AF 100%) !important;
            min-height: 100vh;
        }

        /* Remove default block padding so we can centre freely */
        .block-container {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            max-width: 100% !important;
        }

        /* The middle column becomes our card */
        div[data-testid="column"]:nth-child(2) > div:first-child {
            background: #FFFFFF;
            border-radius: 20px;
            padding: 44px 40px 36px !important;
            box-shadow: 0 32px 80px rgba(0,0,0,0.45);
            margin-top: 10vh;
        }

        /* Card heading — white because it sits on the dark bg above the card */
        div[data-testid="column"]:nth-child(2) h3 {
            color: #FFFFFF !important;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            margin: 0 0 20px !important;
            text-align: center;
            letter-spacing: -0.01em;
        }

        /* Override label colours inside the card */
        div[data-testid="column"]:nth-child(2) [data-testid="stWidgetLabel"] p,
        div[data-testid="column"]:nth-child(2) [data-testid="stWidgetLabel"] label {
            color: #64748B !important;
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.07em;
        }

        /* Sign In button — make it pop */
        div[data-testid="column"]:nth-child(2) .stButton > button {
            background: linear-gradient(135deg, #1E40AF, #2563EB) !important;
            color: #FFFFFF !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            padding: 14px 0 !important;
            border-radius: 10px !important;
            border: none !important;
            box-shadow: 0 6px 20px rgba(37,99,235,0.40) !important;
            margin-top: 6px !important;
            letter-spacing: 0.02em;
        }
        div[data-testid="column"]:nth-child(2) .stButton > button:hover {
            background: linear-gradient(135deg, #1D4ED8, #3B82F6) !important;
            box-shadow: 0 8px 28px rgba(37,99,235,0.55) !important;
            transform: translateY(-1px) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Spacer row to push card down
    st.markdown("<div style='height:6vh'></div>", unsafe_allow_html=True)

    # Centre column — this becomes the visual card via CSS above
    _, col_mid, _ = st.columns([1, 1.1, 1])
    with col_mid:
        # Logo + heading block (all on dark bg — white text throughout)
        st.markdown(
            """
            <div style="text-align:center;margin-bottom:30px;">
                <div style="font-size:3rem;margin-bottom:10px;">🩺</div>
                <div style="font-size:1.55rem;font-weight:800;color:#FFFFFF;letter-spacing:-0.03em;">
                    Clinical AI Copilot
                </div>
                <div style="font-size:0.82rem;color:#93C5FD;margin-top:5px;font-weight:500;">
                    Physician Decision Support Platform
                </div>
                <div style="width:48px;height:3px;background:linear-gradient(90deg,#60A5FA,#2563EB);
                            border-radius:2px;margin:14px auto 16px;"></div>
                <div style="font-size:1.10rem;font-weight:700;color:#FFFFFF;letter-spacing:-0.01em;">
                    Sign in to your account
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        username = st.text_input("USERNAME", placeholder="e.g. dr.sharma", key="login_user", label_visibility="visible")
        password = st.text_input("PASSWORD", type="password", placeholder="Enter your password", key="login_pass", label_visibility="visible")

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        if st.button("🔐  Sign In", key="login_btn", use_container_width=True):
            user = _PHYSICIAN_USERS.get(username.strip().lower())
            if user and user["password"] == password:
                st.session_state["authenticated"] = True
                st.session_state["logged_in_user"] = user["display"]
                st.session_state["logged_in_dept"] = user["dept"]
                st.rerun()
            else:
                st.error("❌  Invalid username or password.")

        st.markdown(
            """
            <div style="margin-top:20px;font-size:0.72rem;color:#94A3B8;text-align:center;line-height:1.7;">
                🔒 Access restricted to authorized clinical staff only<br>
                <span style="font-size:0.68rem;color:#CBD5E1;">
                    Demo: <strong style="color:#94A3B8;">dr.sharma</strong> / clinic@2025
                    &nbsp;·&nbsp;
                    <strong style="color:#94A3B8;">dr.khan</strong> / cardio#2025
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Patient Records tab
# --------------------------------------------------------------------------- #
def render_records_tab() -> None:
    """Session-scoped patient record store — saved per physician session."""
    st.markdown(
        """
        <div class="section-head">
            <div class="section-icon section-icon-teal">📋</div>
            <div>
                <div class="section-label">Patient Visit Records</div>
                <div class="section-sub">Temporary session store — records are cleared when the app restarts</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Init store
    if "patient_records" not in st.session_state:
        st.session_state["patient_records"] = []

    records: list[PatientRecord] = st.session_state["patient_records"]

    # ---- Save current patient button ----
    col_save, col_clear = st.columns([2, 1])
    with col_save:
        if st.button("💾  Save Current Patient to Records", key="save_record"):
            # Pull current patient from session_state widget values
            xray_res = st.session_state.get("xray_result", {})
            ecg_res  = st.session_state.get("ecg_result",  {})
            rec = PatientRecord(
                timestamp    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                name         = st.session_state.get("pt_name", "—"),
                patient_id   = st.session_state.get("pt_id",   "—"),
                age          = int(st.session_state.get("pt_age", 0)),
                gender       = st.session_state.get("pt_gender", "—"),
                blood_type   = st.session_state.get("pt_blood",  "—"),
                symptoms     = st.session_state.get("pt_symptoms", "—"),
                physician    = st.session_state.get("pt_physician", "—"),
                department   = st.session_state.get("pt_dept", "—"),
                ward         = st.session_state.get("pt_ward", "—"),
                xray_top_finding = xray_res.get("top_finding", "—") if xray_res else "—",
                xray_flagged     = sum(1 for f in xray_res.get("findings", []) if f["flagged"]) if xray_res else 0,
                ecg_top_finding  = ecg_res.get("top_finding",  "—") if ecg_res else "—",
                ecg_flagged      = sum(1 for f in ecg_res.get("findings",  []) if f["flagged"]) if ecg_res else 0,
                feedback         = list(st.session_state.get("doctor_feedback", [])),
            )
            records.append(rec)
            st.session_state["patient_records"] = records
            st.success(f"✅ Record saved for **{rec.name}** ({rec.patient_id}) at {rec.timestamp}")

    with col_clear:
        if st.button("🗑️  Clear All Records", key="clear_records"):
            st.session_state["patient_records"] = []
            st.rerun()

    if not records:
        st.markdown(
            f"""
            <div style="background:{COLORS['info_bg']};border:1px dashed {COLORS['border_teal']};
                        border-radius:12px;padding:36px;text-align:center;margin-top:12px;">
                <div style="font-size:2rem;margin-bottom:10px;">📋</div>
                <div style="font-size:0.95rem;font-weight:600;color:{COLORS['text_primary']};margin-bottom:4px;">
                    No records saved yet
                </div>
                <div style="font-size:0.85rem;color:{COLORS['text_muted']};">
                    Fill in a patient's details, run analyses, then click
                    <strong>Save Current Patient to Records</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ---- Search/filter ----
    search = st.text_input("🔍 Search records by name or MRN", key="rec_search", placeholder="Type to filter...")
    filtered = [
        r for r in records
        if search.lower() in r.name.lower() or search.lower() in r.patient_id.lower()
    ] if search else records

    st.caption(f"Showing **{len(filtered)}** of **{len(records)}** record(s)")

    # ---- Table ----
    rows_html = ""
    for i, r in enumerate(reversed(filtered)):
        xray_badge = (
            f'<span class="rec-badge-flagged">⚠ {r.xray_flagged} flags — {r.xray_top_finding}</span>'
            if r.xray_flagged > 0
            else f'<span class="rec-badge-ok">✓ {r.xray_top_finding}</span>'
        )
        ecg_badge = (
            f'<span class="rec-badge-flagged">⚠ {r.ecg_flagged} flags — {r.ecg_top_finding}</span>'
            if r.ecg_flagged > 0
            else f'<span class="rec-badge-ok">✓ {r.ecg_top_finding}</span>'
        )
        fb_count = len(r.feedback)
        fb_text  = f"{fb_count} note(s)" if fb_count else "—"
        rows_html += f"""
        <tr>
            <td><strong>{r.name}</strong><br>
                <span style="font-size:0.70rem;color:#94A3B8;">{r.patient_id}</span></td>
            <td>{r.age} / {r.gender} / {r.blood_type}</td>
            <td>{r.physician}<br>
                <span style="font-size:0.70rem;color:#94A3B8;">{r.department} · {r.ward}</span></td>
            <td>{xray_badge}</td>
            <td>{ecg_badge}</td>
            <td style="font-size:0.78rem;color:#475569;max-width:180px;">{r.symptoms[:80]}{'…' if len(r.symptoms)>80 else ''}</td>
            <td style="font-size:0.75rem;color:#7C3AED;">{fb_text}</td>
            <td style="font-size:0.72rem;color:#94A3B8;">{r.timestamp}</td>
        </tr>
        """

    table_html = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
      body {{ margin:0;padding:0;background:transparent; }}
    </style>
    <div style="overflow-x:auto;">
    <table class="records-table">
      <thead><tr>
        <th>Patient</th><th>Age/Sex/Blood</th><th>Physician</th>
        <th>X-Ray</th><th>ECG</th><th>Symptoms</th><th>Feedback</th><th>Saved At</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    """
    height = 80 + len(filtered) * 72
    st_components.html(table_html, height=min(height, 600), scrolling=True)


# --------------------------------------------------------------------------- #
# Doctor Feedback panel (injected into Agent Workflow tab)
# --------------------------------------------------------------------------- #
def render_feedback_panel(patient: PatientContext) -> None:
    """Allow the physician to confirm or override individual AI findings."""
    st.markdown(
        f"""
        <div class="section-head" style="margin-top:28px;">
            <div class="section-icon section-icon-purple">💬</div>
            <div>
                <div class="section-label">Physician Feedback &amp; Finding Confirmation</div>
                <div class="section-sub">Confirm AI findings you are certain about, or flag disagreements for the record</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    xray_res = st.session_state.get("xray_result", {})
    ecg_res  = st.session_state.get("ecg_result",  {})

    all_findings: list[dict] = []
    if xray_res:
        for f in xray_res.get("findings", [])[:6]:
            all_findings.append({**f, "source": "X-Ray"})
    if ecg_res:
        for f in ecg_res.get("findings", [])[:6]:
            all_findings.append({**f, "source": "ECG"})

    if not all_findings:
        st.markdown(
            f"""
            <div style="background:{COLORS['info_bg']};border:1px dashed {COLORS['border_teal']};
                        border-radius:10px;padding:24px;text-align:center;color:{COLORS['text_muted']};font-size:0.88rem;">
                Run an X-Ray or ECG analysis first to enable physician feedback.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if "doctor_feedback" not in st.session_state:
        st.session_state["doctor_feedback"] = []

    st.markdown(
        f"<div style='font-size:0.80rem;color:{COLORS['text_muted']};margin-bottom:14px;'>"
        f"Review each AI finding below. Select your clinical assessment and add notes where needed."
        f"</div>",
        unsafe_allow_html=True,
    )

    feedback_items = []
    for idx, finding in enumerate(all_findings):
        src_color = COLORS["teal"] if finding["source"] == "X-Ray" else COLORS["purple"]
        src_icon  = "🫁" if finding["source"] == "X-Ray" else "❤️"
        flag_icon = "⚠" if finding["flagged"] else "✓"
        flag_clr  = COLORS["danger"] if finding["flagged"] else COLORS["success"]

        st.markdown(
            f"""
            <div class="feedback-card">
              <div class="feedback-finding-name">
                {src_icon} <span style="color:{src_color};font-size:0.72rem;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.06em;">{finding['source']}</span>
                &nbsp;·&nbsp;
                {finding['label']}
                &nbsp;
                <span style="font-size:0.78rem;color:{flag_clr};">{flag_icon} {finding['probability']:.1f}%</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns([1, 2])
        with col_a:
            assessment = st.selectbox(
                "Clinical Assessment",
                ["— Select —", "✅ Confirmed — I agree with AI", "❌ Overridden — AI is incorrect",
                 "⚠️ Uncertain — Needs further workup", "➕ Additional finding noted"],
                key=f"fb_assess_{idx}",
            )
        with col_b:
            note = st.text_input(
                "Physician note (optional)",
                placeholder="e.g. Correlates with clinical presentation...",
                key=f"fb_note_{idx}",
            )

        if assessment != "— Select —":
            feedback_items.append({
                "finding":    finding["label"],
                "source":     finding["source"],
                "probability": finding["probability"],
                "assessment": assessment,
                "note":       note,
                "physician":  st.session_state.get("logged_in_user", patient.physician),
                "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            })

    if st.button("💾  Submit Physician Feedback", key="submit_feedback"):
        if feedback_items:
            st.session_state["doctor_feedback"] = feedback_items
            st.markdown(
                f"""
                <div class="feedback-submitted">
                    ✅ Feedback recorded for {len(feedback_items)} finding(s) by
                    <strong>{st.session_state.get('logged_in_user', patient.physician)}</strong>.
                    Save the patient record to include this feedback.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("Please select a clinical assessment for at least one finding before submitting.")

    # Show previously submitted feedback
    existing = st.session_state.get("doctor_feedback", [])
    if existing:
        with st.expander(f"📝 View submitted feedback ({len(existing)} item(s))", expanded=False):
            for fb in existing:
                st.markdown(
                    f"**{fb['finding']}** ({fb['source']}) — {fb['assessment']}"
                    + (f" | *{fb['note']}*" if fb['note'] else "")
                    + f" — `{fb['physician']}` @ {fb['timestamp']}"
                )


# --------------------------------------------------------------------------- #
# App entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="Clinical AI Copilot",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()

    # ---- Authentication gate ----
    if not st.session_state.get("authenticated", False):
        render_login_page()
        return

    # ---- Logged-in header badge (sidebar top) ----
    with st.sidebar:
        logged_user = st.session_state.get("logged_in_user", "Unknown")
        logged_dept = st.session_state.get("logged_in_dept", "")
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#1E40AF,#2563EB);border-radius:10px;
                        padding:12px 14px;margin-bottom:10px;">
              <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                          letter-spacing:0.1em;color:#93C5FD;margin-bottom:3px;">Signed in as</div>
              <div style="font-size:0.90rem;font-weight:700;color:#FFFFFF;">{logged_user}</div>
              <div style="font-size:0.72rem;color:#BFDBFE;">{logged_dept}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🚪 Sign Out", key="signout_btn", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # Determine active scan mode from session state for dynamic masthead subtitle
    has_xray = "xray_result" in st.session_state
    has_ecg  = "ecg_result"  in st.session_state
    if has_xray and has_ecg:
        _scan_mode = "both"
    elif has_xray:
        _scan_mode = "xray"
    elif has_ecg:
        _scan_mode = "ecg"
    else:
        _scan_mode = "default"

    render_masthead(scan_mode=_scan_mode)
    render_metric_strip()

    patient = render_sidebar()

    tab_radiology, tab_ecg, tab_combined, tab_guidelines, tab_agent, tab_records = st.tabs(
        ["🫁 X-Ray Analysis", "❤️ ECG Cardiac", "📊 Combined Results",
         "📚 Clinical Guidelines", "🤖 Agent Workflow", "📋 Patient Records"]
    )
    with tab_radiology:
        render_radiology_tab()
    with tab_ecg:
        render_ecg_tab()
    with tab_combined:
        render_combined_tab()
    with tab_guidelines:
        render_guidelines_tab()
    with tab_agent:
        render_agent_tab(patient)
        st.markdown("<hr style='border:none;border-top:1px solid #E2E8F0;margin:28px 0;'>", unsafe_allow_html=True)
        render_feedback_panel(patient)
    with tab_records:
        render_records_tab()


if __name__ == "__main__":
    main()