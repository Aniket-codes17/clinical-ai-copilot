"""
Clinical AI Copilot & Decision Support
---------------------------------------
A multimodal Streamlit application that pairs a radiograph classification
model with a retrieval-augmented guideline search tool, gated behind an
explicit physician-approval workflow for autonomous execution.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Ensure the project root is importable
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from tools.image_tool import classify_image  # noqa: E402
from tools.rag_tool import query_clinical_rag  # noqa: E402

TEMP_DIR = Path("./temp")


# --------------------------------------------------------------------------- #
# Comprehensive Database Auto-Discovery Loader (Clean Caching)
# --------------------------------------------------------------------------- #
@st.cache_data
def load_patient_database() -> dict[str, dict[str, Any]]:
    """Recursively scans the project workspace for CSV datasets and images to load ALL patients."""
    database = {}

    # Build an index of every image on disk ONCE (filename -> full path)
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.dicom"]
    image_index: dict[str, str] = {}
    for ext in image_extensions:
        for img_path in Path(".").rglob(ext):
            image_index[img_path.name] = str(img_path)  # last match wins on name collisions

    csv_files = list(Path(".").rglob("*.csv"))

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            cols = [c.lower() for c in df.columns]
            if any(
                k in cols
                for k in ["patient id", "patient_id", "image index", "image_index", "filename"]
            ):
                for idx, row in df.iterrows():
                    p_id_raw = row.get(
                        "Patient ID", row.get("patient_id", row.get("PatientID", idx))
                    )
                    patient_id = f"PATIENT-{p_id_raw}"

                    img_name = str(
                        row.get(
                            "Image Index",
                            row.get("image_index", row.get("filename", f"{idx}.png")),
                        )
                    )

                    # O(1) lookup instead of a full filesystem walk per row
                    img_path = image_index.get(img_name, "./results/xray.png")

                    symptoms_raw = str(
                        row.get(
                            "Finding Labels",
                            row.get("findings", row.get("Diagnosis", "Respiratory Symptoms")),
                        )
                    )

                    database[patient_id] = {
                        "age": int(row.get("Patient Age", row.get("age", 45))),
                        "gender": str(row.get("Patient Gender", row.get("gender", "Male"))),
                        "symptoms": f"Presenting findings: {symptoms_raw}",
                        "default_query": f"Clinical guidelines for {symptoms_raw.split('|')[0]}",
                        "image_path": img_path,
                    }
                if database:
                    return database
        except Exception:
            continue

    # ... rest of fallback logic unchanged

    # 2. Fallback: Automatically discover ALL images in the project directory
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.dicom"]
    all_images = []
    for ext in image_extensions:
        all_images.extend(list(Path(".").rglob(ext)))

    valid_images = [
        img
        for img in all_images
        if "results" not in img.parts and "temp" not in img.parts
    ]

    if valid_images:
        for idx, img_file in enumerate(valid_images, start=1000):
            patient_id = f"PATIENT-{idx}"
            database[patient_id] = {
                "age": 20 + (idx * 7 % 60),
                "gender": "Female" if idx % 2 == 0 else "Male",
                "symptoms": "Fever, cough, and shortness of breath.",
                "default_query": "Empiric outpatient treatment guidelines for pneumonia",
                "image_path": str(img_file),
            }
    else:
        # Emergency fallback if no dataset is found on disk
        database["PATIENT-1042"] = {
            "age": 45,
            "gender": "Male",
            "symptoms": "Fever, persistent cough, dyspnea on exertion for 3 days.",
            "default_query": "Empiric outpatient treatment guidelines for pneumonia",
            "image_path": "./results/xray.png",
        }

    return database


PATIENT_DATABASE = load_patient_database()


# --------------------------------------------------------------------------- #
# Design Tokens & Theme Styling
# --------------------------------------------------------------------------- #
COLORS = {
    "ink": "#0B2545",
    "ink_soft": "#3C5578",
    "teal": "#0F766E",
    "teal_dark": "#0B5A54",
    "mist": "#F3F6FA",
    "card": "#FFFFFF",
    "border": "#DCE4EE",
    "slate": "#64748B",
}


def inject_theme() -> None:
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

        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label {{
            color: {COLORS['ink_soft']} !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
        }}

        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div,
        div[data-baseweb="select"] > div {{
            background-color: {COLORS['mist']} !important;
            border: 1px solid {COLORS['border']} !important;
            border-radius: 7px !important;
            color: {COLORS['ink']} !important;
        }}

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
        }}
        .masthead-subtitle {{
            color: #B9C8E0;
            font-size: 0.9rem;
            margin-top: 2px;
        }}

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

        .panel {{
            background: {COLORS['card']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            padding: 18px 20px;
        }}

        .stButton>button {{
            background-color: {COLORS['teal']} !important;
            color: #FFFFFF !important;
            border-radius: 7px !important;
            border: none !important;
            padding: 8px 20px !important;
            font-weight: 600 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@dataclass
class PatientContext:
    patient_id: str
    age: int
    gender: str
    symptoms: str
    default_query: str
    image_path: str


def render_masthead() -> None:
    st.markdown(
        """
        <div class="masthead">
            <div>
                <p class="masthead-title">Clinical AI Copilot</p>
                <p class="masthead-subtitle">Multimodal decision support &mdash; imaging analysis &amp; evidence-based guidelines</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def on_patient_change():
    selected_id = st.session_state["selected_patient_id"]
    if selected_id in PATIENT_DATABASE:
        data = PATIENT_DATABASE[selected_id]
        st.session_state["patient_age"] = data["age"]
        st.session_state["patient_gender"] = data["gender"]
        st.session_state["patient_symptoms"] = data["symptoms"]


def render_sidebar() -> PatientContext:
    with st.sidebar:
        st.markdown("#### Patient Registry Search")

        patient_options = list(PATIENT_DATABASE.keys())
        if (
            "selected_patient_id" not in st.session_state
            or st.session_state["selected_patient_id"] not in PATIENT_DATABASE
        ):
            st.session_state["selected_patient_id"] = patient_options[0]
            on_patient_change()

        selected_id = st.selectbox(
            "Select or Search Patient Record",
            options=patient_options,
            key="selected_patient_id",
            on_change=on_patient_change,
        )

        age = st.number_input(
            "Age", min_value=0, max_value=120, key="patient_age"
        )
        gender = st.selectbox(
            "Gender", options=["Male", "Female", "Other"], key="patient_gender"
        )
        symptoms = st.text_area(
            "Clinical Presentation", key="patient_symptoms", height=100
        )

        p_data = PATIENT_DATABASE.get(selected_id, {})
        default_query = p_data.get(
            "default_query", "Empiric treatment guidelines for pneumonia"
        )
        image_path = p_data.get("image_path", "./results/xray.png")

        st.markdown(
            f"""
            <div class="vitals-rail">
                <div class="vitals-row">
                    <span class="vitals-label">Selected ID</span>
                    <span class="vitals-value">{selected_id}</span>
                </div>
                <div class="vitals-row">
                    <span class="vitals-label">Age / Sex</span>
                    <span class="vitals-value">{age} / {str(gender)[0] if gender else 'U'}</span>
                </div>
                <div class="vitals-row">
                    <span class="vitals-label">Database Size</span>
                    <span class="vitals-value">{len(patient_options)} Patients</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return PatientContext(
        patient_id=selected_id,
        age=age,
        gender=gender,
        symptoms=symptoms,
        default_query=default_query,
        image_path=image_path,
    )


def save_uploaded_file(uploaded_file: Any) -> Path:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    destination = TEMP_DIR / uploaded_file.name
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


# --------------------------------------------------------------------------- #
# Application Tabs
# --------------------------------------------------------------------------- #
def render_radiology_tab(patient: PatientContext) -> None:
    st.subheader(f"Chest Radiograph Diagnostics — {patient.patient_id}")

    uploaded_file = st.file_uploader(
        "Upload a custom chest X-ray (Optional)", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        active_image_path = str(save_uploaded_file(uploaded_file))
    elif os.path.exists(patient.image_path):
        active_image_path = patient.image_path
    else:
        active_image_path = "./results/xray.png"

    col_original, col_heatmap = st.columns(2)

    with col_original:
        st.markdown(f"**Patient Radiograph ({patient.patient_id})**")
        if os.path.exists(active_image_path):
            st.image(active_image_path, use_container_width=True)
        else:
            st.warning("No radiograph image found on disk for this patient.")
            return

    if not st.button("Run Diagnostic Analysis"):
        return

    with st.spinner("Analyzing radiograph with ResNet classifier..."):
        try:
            result = classify_image(active_image_path)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return

    if "error" in result:
        st.error(result["error"])
        return

    with col_heatmap:
        st.markdown("**Grad-CAM Explainability Heatmap**")
        st.image(result["heatmap_path"], use_container_width=True)

    st.success(
        f"**Diagnostic Prediction:** {result['prediction']} | **Confidence:** {result['confidence']}%"
    )


def render_guidelines_tab(patient: PatientContext) -> None:
    st.subheader("Evidence-Based Literature Search")
    query = st.text_input(
        "Search treatment protocols & guidelines", patient.default_query
    )

    if not st.button("Search Knowledge Base"):
        return

    with st.spinner("Querying clinical guideline index..."):
        try:
            response = query_clinical_rag(query)
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            return

    st.markdown(f"<div class='panel'>{response}</div>", unsafe_allow_html=True)


def render_agent_tab(patient: PatientContext) -> None:
    st.subheader("Physician Control & Copilot Execution")

    st.markdown(
        f"""
        <div class="panel">
            <h4>Patient Context &middot; {patient.patient_id}</h4>
            <p>Age {patient.age} &middot; {patient.gender}<br>{patient.symptoms}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Safety Checkpoint")
    approved = st.checkbox(
        f"Authorize AI Copilot to run autonomous diagnostic pipeline for {patient.patient_id}."
    )

    if not st.button("Execute Copilot Agent"):
        return

    if not approved:
        st.error("Execution halted: physician authorization required.")
        return

    st.success("Physician approval granted. Processing...")

    active_image = (
        patient.image_path
        if os.path.exists(patient.image_path)
        else "./results/xray.png"
    )

    with st.spinner("Running imaging and guideline pipeline..."):
        try:
            imaging_result = classify_image(active_image)
        except Exception as exc:
            imaging_result = {"error": str(exc)}

        try:
            guideline_result = query_clinical_rag(patient.symptoms)
        except Exception as exc:
            guideline_result = f"Guideline lookup failed: {exc}"

    col_imaging, col_guideline = st.columns(2)
    with col_imaging:
        st.markdown("**Imaging Findings**")
        if "error" in imaging_result:
            st.warning(imaging_result["error"])
        else:
            st.info(
                f"Prediction: {imaging_result.get('prediction', 'N/A')} "
                f"({imaging_result.get('confidence', 0)}%)"
            )
    with col_guideline:
        st.markdown("**Evidence-Based Recommendation**")
        st.markdown(guideline_result)


def main() -> None:
    st.set_page_config(
        page_title="Clinical AI Copilot",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()
    render_masthead()

    patient = render_sidebar()

    tab_radiology, tab_guidelines, tab_agent = st.tabs(
        ["Radiology Analysis", "Clinical Guidelines", "Agent Workflow"]
    )
    with tab_radiology:
        render_radiology_tab(patient)
    with tab_guidelines:
        render_guidelines_tab(patient)
    with tab_agent:
        render_agent_tab(patient)


if __name__ == "__main__":
    main()