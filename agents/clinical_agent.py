import os
import sys

# Ensure project root directory is in Python's module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from tools.langchain_tools import (
    analyze_chest_xray_tool,
    search_clinical_guidelines_tool,
)


class CopilotState(TypedDict):
    messages: Annotated[list, add_messages]
    doctor_approved: bool


# 1. Initialize LLM and bind tools
llm = ChatOllama(model="llama3.2:1b", temperature=0.1)
tools_map = {
    "analyze_chest_xray": analyze_chest_xray_tool,
    "search_clinical_guidelines": search_clinical_guidelines_tool,
}


def agent_node(state: CopilotState):
    """Primary reasoning node for the clinical assistant."""
    system_prompt = SystemMessage(
        content=(
            "You are an advanced Clinical AI Copilot assisting a medical doctor. "
            "You have access to tools for analyzing chest X-rays and searching medical guidelines. "
            "Formulate concise, evidence-based clinical insights."
        )
    )
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def doctor_checkpoint_node(state: CopilotState):
    """Human-in-the-loop checkpoint requiring doctor authorization."""
    print("\n--- [HUMAN-IN-THE-LOOP CHECKPOINT] ---")
    user_input = (
        input("Doctor, do you authorize proceeding with AI analysis? (y/n): ")
        .strip()
        .lower()
    )
    approved = user_input == "y"
    return {"doctor_approved": approved}


def execute_tool_node(state: CopilotState):
    """Executes clinical tools based on state context."""
    if not state.get("doctor_approved"):
        return {
            "messages": [
                HumanMessage(
                    content="System Notice: Execution halted by reviewing physician."
                )
            ]
        }

    # Run automated clinical evaluation pipeline
    xray_res = analyze_chest_xray_tool.invoke({"image_path": "./results/xray.png"})
    rag_res = search_clinical_guidelines_tool.invoke(
        {"query": "Empiric treatment guidelines for outpatient pneumonia"}
    )

    combined_output = (
        f"--- Clinical Image Analysis ---\n{xray_res}\n\n"
        f"--- Evidence-Based Guidance ---\n{rag_res}"
    )

    return {"messages": [HumanMessage(content=combined_output)]}



builder = StateGraph(CopilotState)
builder.add_node("agent", agent_node)
builder.add_node("doctor_checkpoint", doctor_checkpoint_node)
builder.add_node("execute_tools", execute_tool_node)

builder.add_edge(START, "agent")
builder.add_edge("agent", "doctor_checkpoint")
builder.add_edge("doctor_checkpoint", "execute_tools")
builder.add_edge("execute_tools", END)

clinical_copilot_app = builder.compile()


if __name__ == "__main__":
    print("\n=== Launching Clinical AI Copilot Workflow ===")
    initial_input = {
        "messages": [
            HumanMessage(
                content="Evaluate patient presentation and provide initial differential diagnosis along with treatment protocol."
            )
        ]
    }
    final_state = clinical_copilot_app.invoke(initial_input)

    print("\n=== Workflow Execution Summary ===")
    for msg in final_state["messages"]:
        sender = msg.__class__.__name__
        print(f"\n[{sender}]: {msg.content}")