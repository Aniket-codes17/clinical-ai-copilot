from langchain_core.tools import tool
from tools.image_tool import classify_image
from tools.rag_tool import query_clinical_rag


@tool
def analyze_chest_xray_tool(image_path: str) -> str:
    """Analyzes a chest X-ray image using a fine-tuned CNN model to detect conditions like Pneumonia and returns a Grad-CAM heatmap path."""
    result = classify_image(image_path)
    if "error" in result:
        return f"Error analyzing image: {result['error']}"

    return (
        f"X-Ray Diagnostic Prediction: {result['prediction']}\n"
        f"Confidence Score: {result['confidence']}%\n"
        f"Grad-CAM Heatmap Saved To: {result['heatmap_path']}"
    )


@tool
def search_clinical_guidelines_tool(query: str) -> str:
    """Searches clinical practice guidelines and evidence-based medical documentation in the vector database."""
    return query_clinical_rag(query)