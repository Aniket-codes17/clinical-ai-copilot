import os
import sys

# Ensure root folder is on Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image
import streamlit as st
from tools.image_tool import classify_image
from tools.rag_tool import query_clinical_rag

st.set_page_config(
    page_title="Clinical AI Copilot", page_icon="🩺", layout="wide"
)

st.title("🩺 Clinical AI Copilot & Decision Support System")
st.caption("Multimodal Medical AI Agent — Vision | RAG | Human-in-the-Loop")

st.sidebar.header("📋 Patient Information")
patient_id = st.sidebar.text_input("Patient ID", "PATIENT-1042")
patient_age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=45)
symptoms = st.sidebar.text_area(
    "Clinical Presentation",
    "Fever, persistent cough, dyspnea on exertion for 3 days.",
)

tab1, tab2, tab3 = st.tabs(
    ["📷 Chest X-Ray Analysis", "📚 Guidelines & RAG Search", "🤖 Copilot Agent"]
)

with tab1:
    st.header("Chest X-Ray Diagnostic Model")
    uploaded_file = st.file_uploader(
        "Upload Chest Radiograph (PNG / JPG)", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        os.makedirs("./temp", exist_ok=True)
        temp_path = os.path.join("./temp", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Image")
            st.image(temp_path, use_container_width=True)

        if st.button("Run AI Image Diagnostic"):
            with st.spinner("Analyzing radiograph with fine-tuned CNN..."):
                res = classify_image(temp_path)

            if "error" in res:
                st.error(res["error"])
            else:
                with col2:
                    st.subheader("Grad-CAM Explainability Heatmap")
                    st.image(res["heatmap_path"], use_container_width=True)

                st.success(
                    f"**Predicted Finding:** {res['prediction']} | **Confidence:** {res['confidence']}%"
                )


with tab2:
    st.header("Evidence-Based Clinical Knowledge Retrieval")
    rag_query = st.text_input(
        "Clinical Inquiry",
        "Empiric outpatient treatment guidelines for pneumonia",
    )

    if st.button("Search Medical Knowledge Base"):
        with st.spinner("Searching Chroma vector store..."):
            retrieval_res = query_clinical_rag(rag_query)
        st.markdown(retrieval_res)

with tab3:
    st.header("Agent Execution Checkpoint")
    st.info(
        f"**Patient Record:** {patient_id} ({patient_age} Y/O)\n\n**Symptoms:** {symptoms}"
    )

    st.warning(
        "⚠️ **Human-in-the-Loop Checkpoint**: Doctor authorization required before AI agent execution."
    )

    approve = st.checkbox("I authorize the AI Copilot to run full multi-modal analysis.")

    if st.button("Execute Clinical Copilot"):
        if approve:
            st.success("Authorization granted. Executing workflow...")
            with st.spinner("Running vision and RAG pipeline..."):
                default_img = (
                    "./results/xray.png"
                    if os.path.exists("./results/xray.png")
                    else "xray.png"
                )
                img_eval = classify_image(default_img)
                rag_eval = query_clinical_rag(symptoms)

            st.markdown("### Combined Analysis Summary")
            st.markdown(
                f"**Radiology Prediction:** {img_eval.get('prediction', 'N/A')} ({img_eval.get('confidence', 0)}%)"
            )
            st.markdown("### Evidence Base")
            st.markdown(rag_eval)
        else:
            st.error("Execution blocked: Physician authorization required.")