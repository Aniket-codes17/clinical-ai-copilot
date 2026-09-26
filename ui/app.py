"""
Clinical AI Copilot & Decision Support
---------------------------------------
A multimodal Streamlit application pairing a pretrained multi-label chest
X-ray model (lung + heart findings) with a retrieval-augmented guideline
search tool, gated behind an explicit physician-approval workflow.
"""

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

# Ensure the project root is importable regardless of the working directory
# Streamlit is launched from.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from tools.image_tool import classify_image  # noqa: E402
from tools.rag_tool import query_clinical_rag  # noqa: E402

TEMP_DIR = Path("./temp")
DEFAULT_XRAY_CANDIDATES = ("./results/xray.png", "./xray.png")


# --------------------------------------------------------------------------- #
# Design tokens
# --------------------------------------------------------------------------- #
# A small, deliberate palette rather than default Streamlit theming.
# "Ink" = primary text/headers, "Teal" = clinical accent, "Mist" = page bg.
COLORS = {
    "ink": "#0B2545",
    "ink_soft": "#3C5578",
    "teal": "#0F766E",
    "teal_dark": "#0B5A54",
    "mist": "#F3F6FA",
    "card": "#FFFFFF",
    "border": "#DCE4EE",
    "slate": "#64748B",
    "confirm": "#047857",
    "confirm_bg": "#E3F5EE",
    "alert": "#B91C1C",
    "alert_bg": "#FCEAEA",
    "amber": "#B45309",
    "amber_bg": "#FDF3E1",
}


def inject_theme() -> None:
    """Apply the clinical design system (fonts, layout chrome, components)."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        .stApp {{
            background-color: {COLORS['mist']};
            color: {COLORS['ink']};
        }}

        /* Streamlit's own top toolbar defaults to dark chrome regardless of
           theme; pin it to the ink navy so it reads as part of the design
           instead of a clashing strip. */
        header[data-testid="stHeader"] {{
            background-color: {COLORS['ink']};
        }}
        header[data-testid="stHeader"] * {{
            color: #E7ECF5 !important;
        }}

        section[data-testid="stSidebar"] {{
            background-color: {COLORS['card']};
            border-right: 1px solid {COLORS['border']};
        }}
        section[data-testid="stSidebar"] h4 {{
            color: {COLORS['ink']};
            font-weight: 700;
            letter-spacing: -0.01em;
        }}

        /* ---------- Reskin native form widgets (fixes dark-theme bleed) ---------- */
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label {{
            color: {COLORS['ink_soft']} !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}

        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div,
        div[data-baseweb="select"] > div,
        .stNumberInput > div > div {{
            background-color: {COLORS['mist']} !important;
            border: 1px solid {COLORS['border']} !important;
            border-radius: 7px !important;
            color: {COLORS['ink']} !important;
        }}
        input, textarea {{
            color: {COLORS['ink']} !important;
            background-color: transparent !important;
        }}
        div[data-baseweb="select"] span {{
            color: {COLORS['ink']} !important;
        }}
        .stNumberInput button {{
            background-color: {COLORS['card']} !important;
            border-color: {COLORS['border']} !important;
            color: {COLORS['ink']} !important;
        }}
        div[data-baseweb="input"]:focus-within > div,
        div[data-baseweb="textarea"]:focus-within > div,
        div[data-baseweb="select"]:focus-within > div {{
            border-color: {COLORS['teal']} !important;
            box-shadow: 0 0 0 1px {COLORS['teal']} !important;
        }}

        [data-testid="stFileUploaderDropzone"] {{
            background-color: {COLORS['mist']} !important;
            border: 1.5px dashed {COLORS['border']} !important;
            border-radius: 8px !important;
        }}
        [data-testid="stFileUploaderDropzone"] * {{
            color: {COLORS['ink_soft']} !important;
        }}
        [data-testid="stFileUploaderDropzone"] button {{
            background-color: {COLORS['card']} !important;
            border: 1px solid {COLORS['border']} !important;
            color: {COLORS['ink']} !important;
        }}

        .stCheckbox p {{
            color: {COLORS['ink']} !important;
            font-weight: 500;
        }}

        [data-testid="stCaptionContainer"] {{
            color: {COLORS['slate']} !important;
        }}

        h1, h2, h3, h4, h5, h6 {{ color: {COLORS['ink']}; }}

        /* ---------- System status strip (signature element) ---------- */
        .status-strip {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1px;
            background: {COLORS['border']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 22px;
        }}
        .status-cell {{
            background: {COLORS['card']};
            padding: 10px 16px;
            display: flex;
            justify-content: space-between;
            align-items: baseline;
        }}
        .status-cell-label {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: {COLORS['slate']};
        }}
        .status-cell-value {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            font-weight: 600;
            color: {COLORS['ink']};
        }}

        /* ---------- Masthead ---------- */
        .masthead {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: {COLORS['ink']};
            border-radius: 10px;
            padding: 22px 28px;
            margin-bottom: 20px;
        }}
        .masthead-title {{
            color: #FFFFFF;
            font-size: 1.55rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.01em;
        }}
        .masthead-subtitle {{
            color: #B9C8E0;
            font-size: 0.9rem;
            margin-top: 2px;
        }}
        .masthead-status {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            color: #B9C8E0;
            border: 1px solid #2A4066;
            border-radius: 20px;
            padding: 6px 14px;
        }}
        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #34D399;
            box-shadow: 0 0 0 3px rgba(52, 211, 153, 0.25);
        }}

        /* ---------- Sidebar vitals rail ---------- */
        .vitals-rail {{
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 14px;
        }}
        .vitals-row {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            padding: 9px 12px;
            border-bottom: 1px solid {COLORS['border']};
            background: {COLORS['card']};
        }}
        .vitals-row:last-child {{ border-bottom: none; }}
        .vitals-label {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: {COLORS['slate']};
        }}
        .vitals-value {{
            font-family: 'IBM Plex Mono', monospace;
            font-weight: 600;
            font-size: 0.85rem;
            color: {COLORS['ink']};
        }}

        /* ---------- Cards & panels ---------- */
        .panel {{
            background: {COLORS['card']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            padding: 18px 20px;
        }}
        .panel-title {{
            font-weight: 600;
            font-size: 0.95rem;
            color: {COLORS['ink']};
            margin-bottom: 10px;
        }}
        .context-strip {{
            background: {COLORS['card']};
            border-left: 3px solid {COLORS['teal']};
            border-radius: 6px;
            padding: 14px 18px;
            margin-bottom: 18px;
        }}
        .context-strip h4 {{
            margin: 0 0 4px 0;
            color: {COLORS['ink']};
            font-size: 0.95rem;
        }}
        .context-strip p {{
            margin: 0;
            color: {COLORS['ink_soft']};
            font-size: 0.85rem;
            line-height: 1.5;
        }}

        /* ---------- Result badges ---------- */
        .badge {{
            display: inline-block;
            padding: 5px 14px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
            font-family: 'IBM Plex Mono', monospace;
        }}
        .badge-normal {{ background: {COLORS['confirm_bg']}; color: {COLORS['confirm']}; }}
        .badge-abnormal {{ background: {COLORS['alert_bg']}; color: {COLORS['alert']}; }}

        .confidence-track {{
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: {COLORS['border']};
            margin-top: 10px;
            overflow: hidden;
        }}
        .confidence-fill {{
            height: 100%;
            background: {COLORS['teal']};
            border-radius: 3px;
        }}

        /* ---------- Buttons ---------- */
        .stButton>button {{
            background-color: {COLORS['teal']} !important;
            color: #FFFFFF !important;
            border-radius: 7px !important;
            border: none !important;
            padding: 8px 20px !important;
            font-weight: 600 !important;
            transition: background-color 0.15s ease;
        }}
        .stButton>button:hover {{
            background-color: {COLORS['teal_dark']} !important;
        }}

        /* ---------- Tabs ---------- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
            background-color: {COLORS['card']};
            padding: 6px;
            border-radius: 8px;
            border: 1px solid {COLORS['border']};
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 6px;
            padding: 8px 16px;
            color: {COLORS['slate']};
            font-weight: 500;
        }}
        .stTabs [data-baseweb="tab"] p {{
            color: inherit !important;
            font-weight: inherit !important;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {COLORS['mist']} !important;
            color: {COLORS['teal']} !important;
            font-weight: 600;
        }}
        .stTabs [data-baseweb="tab-highlight"] {{
            background-color: {COLORS['teal']} !important;
        }}

        .footnote {{
            font-size: 0.72rem;
            color: {COLORS['slate']};
            margin-top: 4px;
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
    patient_id: str
    age: int
    gender: str
    symptoms: str


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
def render_masthead() -> None:
    st.markdown(
        """
        <div class="masthead">
            <div>
                <p class="masthead-title">Clinical AI Copilot</p>
                <p class="masthead-subtitle">Multimodal decision support &mdash; imaging analysis &amp; evidence-based guidelines</p>
            </div>
            <div class="masthead-status">
                <span class="status-dot"></span> SESSION ENCRYPTED &middot; HIPAA COMPLIANT
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_strip() -> None:
    """A bedside-monitor-style readout of system state — the page's
    signature element, standing in for a generic hero banner."""
    st.markdown(
        """
        <div class="status-strip">
            <div class="status-cell">
                <span class="status-cell-label">Imaging model</span>
                <span class="status-cell-value">TorchXRayVision &middot; DenseNet121</span>
            </div>
            <div class="status-cell">
                <span class="status-cell-label">Guideline index</span>
                <span class="status-cell-value">Vector RAG &middot; live</span>
            </div>
            <div class="status-cell">
                <span class="status-cell-label">Agent mode</span>
                <span class="status-cell-value">Physician-supervised</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> PatientContext:
    with st.sidebar:
        st.markdown("#### Patient Record")
        patient_id = st.text_input("Patient ID", "PATIENT-1042")
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        symptoms = st.text_area(
            "Clinical Presentation",
            "Fever, persistent cough, dyspnea on exertion for 3 days.",
            height=100,
        )

        st.markdown(
            f"""
            <div class="vitals-rail">
                <div class="vitals-row">
                    <span class="vitals-label">Record</span>
                    <span class="vitals-value">{patient_id}</span>
                </div>
                <div class="vitals-row">
                    <span class="vitals-label">Age / Sex</span>
                    <span class="vitals-value">{age} / {gender[0]}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Data stays local to this session and is not persisted.")

    return PatientContext(patient_id=patient_id, age=age, gender=gender, symptoms=symptoms)


def render_findings_panel(result: dict) -> str:
    """Renders every scored finding, not just one -- a chest X-ray can
    show more than one abnormality (or none) at a time, and this project
    is explicitly scoped to both heart and lung findings."""
    findings = result.get("findings", [])
    if not findings:
        return "<div class='panel'>No findings returned.</div>"

    top = findings[0]
    top_flagged = any(f["flagged"] for f in findings)
    headline_class = "badge-abnormal" if top_flagged else "badge-normal"
    headline_text = result.get("prediction", top["label"])

    rows = []
    for finding in findings[:6]:  # top 6 keeps the panel scannable
        region_tag = "&hearts; Heart" if finding["region"] == "heart" else "&#129502; Lung"
        bar_color = COLORS["alert"] if finding["flagged"] else COLORS["teal"]
        rows.append(
            f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:4px;">
                    <span style="color:{COLORS['ink']}; font-weight:500;">
                        {finding['label']}
                        <span style="color:{COLORS['slate']}; font-size:0.72rem; margin-left:6px;">{region_tag}</span>
                    </span>
                    <span style="font-family:'IBM Plex Mono', monospace; color:{COLORS['slate']};">
                        {finding['probability']:.1f}%
                    </span>
                </div>
                <div class="confidence-track">
                    <div class="confidence-fill" style="width:{finding['probability']}%; background:{bar_color};"></div>
                </div>
            </div>
            """
        )

    return f"""
        <div class="panel">
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:14px;">
                <span style="font-weight:600;">Top Finding</span>
                <span class="badge {headline_class}">{headline_text}</span>
            </div>
            {''.join(rows)}
        </div>
    """


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
# Tabs
# --------------------------------------------------------------------------- #
def render_radiology_tab() -> None:
    st.subheader("Chest Radiograph Diagnostics")
    uploaded_file = st.file_uploader(
        "Upload a chest X-ray", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is None:
        st.info("Upload a PNG or JPG radiograph to begin analysis.")
        return

    image_path = save_uploaded_file(uploaded_file)
    col_original, col_heatmap = st.columns(2)

    with col_original:
        st.markdown("**Original radiograph**")
        st.image(str(image_path), use_container_width=True)

    if not st.button("Run Diagnostic Analysis"):
        return

    with st.spinner("Analyzing radiograph for lung and heart findings..."):
        try:
            result = classify_image(str(image_path))
        except Exception as exc:  # surface model/runtime failures cleanly
            st.error(f"Analysis failed: {exc}")
            return

    if "error" in result:
        st.error(result["error"])
        return

    with col_heatmap:
        st.markdown(f"**Grad-CAM &middot; {result['top_finding']}**", unsafe_allow_html=True)
        st.image(result["heatmap_path"], use_container_width=True)

    st.markdown(render_findings_panel(result), unsafe_allow_html=True)
    st.markdown(
        "<p class='footnote'>Model output is decision support only and does "
        "not constitute a clinical diagnosis.</p>",
        unsafe_allow_html=True,
    )


_CITATION_PATTERN = re.compile(
    r"--- Citation (\d+) \[(.*?), (\d+)% match\] ---\n(.*?)(?=\n\n--- Citation|\Z)",
    re.DOTALL,
)


def render_citations_panel(raw_text: str) -> str:
    """Parses rag_tool.py's plain-text citation format into styled cards.

    query_clinical_rag() returns plain text on purpose -- that's also what
    gets fed into the LLM tool-calling context (tools/agent_tools.py), and
    markup there would just waste tokens. So the HTML formatting lives
    here, in the UI layer, rather than in the tool itself.
    """
    matches = _CITATION_PATTERN.findall(raw_text)
    if not matches:
        # Not in the citation format -- an error or "no results" message.
        return f"<div class='panel'>{raw_text}</div>"

    cards = [
        f"""
        <div style="font-size:0.82rem; color:{COLORS['slate']}; margin-bottom:10px;">
            {len(matches)} relevant passage(s) found
        </div>
        """
    ]
    for idx, source, match_pct, content in matches:
        cards.append(
            f"""
            <div style="background:{COLORS['card']}; border:1px solid {COLORS['border']};
                        border-left:3px solid {COLORS['teal']}; border-radius:6px;
                        padding:14px 18px; margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.04em;
                                 color:{COLORS['slate']}; font-weight:600;">
                        Citation {idx} &middot; {source}
                    </span>
                    <span style="font-family:'IBM Plex Mono', monospace; font-size:0.75rem;
                                 color:{COLORS['teal']}; font-weight:600;">
                        {match_pct}% match
                    </span>
                </div>
                <p style="margin:0; font-size:0.9rem; color:{COLORS['ink']}; line-height:1.6;">
                    {content.strip()}
                </p>
            </div>
            """
        )
    return "".join(cards)


def render_guidelines_tab() -> None:
    st.subheader("Evidence-Based Literature Search")
    query = st.text_input(
        "Search treatment protocols & guidelines",
        "Empiric outpatient treatment guidelines for pneumonia",
    )

    if not st.button("Search Knowledge Base"):
        return

    with st.spinner("Querying the clinical guideline index..."):
        try:
            response = query_clinical_rag(query)
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            return

    st.markdown(render_citations_panel(response), unsafe_allow_html=True)


def render_agent_tab(patient: PatientContext) -> None:
    st.subheader("Physician Control & Copilot Execution")

    st.markdown(
        f"""
        <div class="context-strip">
            <h4>Patient Context &middot; {patient.patient_id}</h4>
            <p>Age {patient.age} &middot; {patient.gender}<br>{patient.symptoms}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Safety Checkpoint")
    approved = st.checkbox(
        "Authorize the AI copilot to run the autonomous diagnostic pipeline."
    )

    if not st.button("Execute Copilot Agent"):
        return

    if not approved:
        st.error("Execution halted: physician authorization is required.")
        return

    st.success("Physician approval granted. Processing...")

    default_image = find_default_xray()
    with st.spinner("Running the imaging and guideline pipeline..."):
        try:
            imaging_result = (
                classify_image(default_image)
                if default_image
                else {"error": "No default radiograph found for this patient."}
            )
        except Exception as exc:
            imaging_result = {"error": str(exc)}

        try:
            guideline_result = query_clinical_rag(patient.symptoms)
        except Exception as exc:
            guideline_result = f"Guideline lookup failed: {exc}"

    col_imaging, col_guideline = st.columns(2)
    with col_imaging:
        st.markdown("**Imaging findings**")
        if "error" in imaging_result:
            st.warning(imaging_result["error"])
        else:
            flagged = [f for f in imaging_result.get("findings", []) if f["flagged"]]
            st.info(
                f"Top finding: {imaging_result.get('top_finding', 'N/A')} "
                f"({imaging_result.get('top_confidence', 0)}%) · "
                f"{len(flagged)} finding(s) above threshold"
            )
    with col_guideline:
        st.markdown("**Evidence-based recommendation**")
        st.markdown(guideline_result)


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
    render_masthead()
    render_status_strip()

    patient = render_sidebar()

    tab_radiology, tab_guidelines, tab_agent = st.tabs(
        ["Radiology Analysis", "Clinical Guidelines", "Agent Workflow"]
    )
    with tab_radiology:
        render_radiology_tab()
    with tab_guidelines:
        render_guidelines_tab()
    with tab_agent:
        render_agent_tab(patient)


if __name__ == "__main__":
    main()