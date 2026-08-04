import os
from typing import Annotated, TypedDict
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages


# 1. Define the shared state object passed through nodes
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    doctor_approval: bool


# 2. Define workflow nodes
def supervisor_node(state: AgentState):
    print("\n[Supervisor Agent]: Processing clinical input...")
    return {
        "messages": [
            "Supervisor: Recommended action -> Query RAG knowledge base for treatment guidelines."
        ]
    }


def human_in_the_loop_node(state: AgentState):
    print(
        "\n[Human-in-the-loop Checkpoint]: Supervisor proposes searching evidence database."
    )
    user_input = (
        input("Doctor, do you approve executing this tool action? (y/n): ")
        .strip()
        .lower()
    )
    approved = user_input == "y"
    return {"doctor_approval": approved}


def execute_tool_node(state: AgentState):
    if state.get("doctor_approval"):
        print(
            "\n[Tool Execution]: Doctor approved. Fetching evidence from RAG store..."
        )
        return {
            "messages": [
                "Tool Output: Empiric treatment for healthy adults is Amoxicillin 1g TID."
            ]
        }
    else:
        print("\n[Tool Execution]: Doctor declined action. Stopping tool execution.")
        return {
            "messages": [
                "Tool Output: Action cancelled by reviewing physician."
            ]
        }


# 3. Construct the LangGraph StateGraph
builder = StateGraph(AgentState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("human_approval", human_in_the_loop_node)
builder.add_node("execute_tool", execute_tool_node)

builder.add_edge(START, "supervisor")
builder.add_edge("supervisor", "human_approval")
builder.add_edge("human_approval", "execute_tool")
builder.add_edge("execute_tool", END)

graph = builder.compile()

# 4. Test the agent workflow
if __name__ == "__main__":
    initial_state = {
        "messages": [
            "Patient is a 45-year-old presenting with symptoms of acute pneumonia."
        ]
    }
    output = graph.invoke(initial_state)

    print("\n--- Final Workflow Messages ---")
    for msg in output["messages"]:
        print(msg)