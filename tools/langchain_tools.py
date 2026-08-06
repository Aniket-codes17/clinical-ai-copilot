"""
tools/agent_tools.py
---------------------
LangChain @tool wrappers exposing the imaging and RAG capabilities to an
LLM agent (e.g. for binding into clinical_agent_graph.py).

Renamed from "lang_chain.py" -- that name is one character away from the
`langchain` package itself, which invites confusion, and it sits better
alongside the other tool modules (image_tool.py, rag_tool.py, llm_tool.py).

Note on formatting: `search_clinical_guidelines_tool` calls
`query_clinical_rag_text`, not `query_clinical_rag`. The latter returns
HTML formatted for the Streamlit panel; feeding HTML markup into an LLM's
tool-call context wastes tokens and risks confusing the model with markup
it has no use for. The plain-text variant is the one meant for this.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from tools.image_tool import classify_image
from tools.rag_tool import query_clinical_rag

logger = logging.getLogger("clinical_agent.agent_tools")


@tool
def analyze_chest_xray_tool(image_path: str) -> str:
    """Analyzes a chest X-ray image using a fine-tuned CNN model to detect
    conditions like Pneumonia and returns a Grad-CAM heatmap path."""
    try:
        result = classify_image(image_path)
    except Exception as exc:  # noqa: BLE001 -- a tool call should never crash the agent
        logger.exception("analyze_chest_xray_tool failed unexpectedly.")
        return f"Error analyzing image: {exc}"

    if "error" in result:
        return f"Error analyzing image: {result['error']}"

    return (
        f"X-Ray Diagnostic Prediction: {result['prediction']}\n"
        f"Confidence Score: {result['confidence']}%\n"
        f"Grad-CAM Heatmap Saved To: {result['heatmap_path']}"
    )


@tool
def search_clinical_guidelines_tool(query: str) -> str:
    """Searches clinical practice guidelines and evidence-based medical
    documentation in the vector database."""
    try:
        return query_clinical_rag(query)
    except Exception as exc:  # noqa: BLE001
        logger.exception("search_clinical_guidelines_tool failed unexpectedly.")
        return f"Error searching guidelines: {exc}"


# Convenience export so an agent (e.g. clinical_agent_graph.py, or
# create_react_agent / llm.bind_tools) can pull in every clinical tool at
# once without importing each function individually.
CLINICAL_TOOLS = [analyze_chest_xray_tool, search_clinical_guidelines_tool]