"""
Clinical Decision-Support Agent Graph
--------------------------------------
A LangGraph workflow that proposes an evidence-lookup action and pauses for
mandatory physician sign-off before any tool executes.

Design decisions that differ from a "toy" version of this graph:

1. Human approval is implemented with LangGraph's native `interrupt()` /
   `Command(resume=...)` pattern instead of a blocking `input()` call inside
   a node. A blocking call only works in a single synchronous CLI process;
   `interrupt()` actually suspends the graph and persists its state via a
   checkpointer, so the approval can come from a web request, a queue
   message, or a different process entirely -- which is how a real
   physician-approval step would need to work.
2. Approval/decline are separate graph branches (conditional edges), not an
   if/else buried inside the tool node. The tool node should only ever be
   reached when approval is true.
3. Every step writes a timestamped entry to an `audit_log` field on the
   state. For a clinical workflow, "what happened and who approved it" is
   not optional -- it's the artifact a compliance review would ask for.
4. Logging goes through the standard `logging` module instead of `print`,
   so the graph behaves the same whether it's run from a terminal or
   embedded in a service.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("clinical_agent")


# --------------------------------------------------------------------------- #
# Domain types
# --------------------------------------------------------------------------- #
class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    DECLINED = "declined"


@dataclass
class AuditEntry:
    timestamp: str
    actor: str
    event: str

    @classmethod
    def now(cls, actor: str, event: str) -> "AuditEntry":
        return cls(timestamp=datetime.now(timezone.utc).isoformat(), actor=actor, event=event)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    proposed_action: str
    approval: str | None          # ApprovalDecision.value, or None until decided
    approver_id: str | None
    audit_log: Annotated[list[dict], lambda a, b: a + b]


# --------------------------------------------------------------------------- #
# Tool stub -- swap this out for the real RAG lookup used in the rest of
# the app (tools/rag_tool.py's query_clinical_rag).
# --------------------------------------------------------------------------- #
def fetch_treatment_guideline(query: str) -> str:
    """Placeholder for the real evidence-base lookup."""
    return "Empiric treatment for a healthy adult with CAP: amoxicillin 1g TID for 5-7 days."


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def supervisor_node(state: AgentState) -> dict:
    """Reviews the clinical input and proposes a next action for sign-off."""
    logger.info("Supervisor reviewing clinical input.")
    proposed_action = "Query the clinical guideline RAG store for a treatment recommendation."
    return {
        "messages": [f"Supervisor: proposed action -> {proposed_action}"],
        "proposed_action": proposed_action,
        "audit_log": [AuditEntry.now("supervisor", f"Proposed: {proposed_action}").__dict__],
    }


def request_approval_node(state: AgentState) -> dict:
    """Suspends the graph and waits for an external physician decision.

    Calling `interrupt()` here pauses execution and returns control to the
    caller with the payload below. The graph resumes -- from this exact
    point, with full state intact -- only when the caller invokes it again
    with `Command(resume=<decision>)`. Nothing downstream runs until that
    happens, and the pause survives process restarts as long as the same
    checkpointer/thread is used.
    """
    decision = interrupt(
        {
            "reason": "Physician approval required before tool execution.",
            "proposed_action": state["proposed_action"],
        }
    )
    approved = decision.get("approved", False)
    approver_id = decision.get("approver_id", "unknown")

    return {
        "approval": ApprovalDecision.APPROVED.value if approved else ApprovalDecision.DECLINED.value,
        "approver_id": approver_id,
        "audit_log": [
            AuditEntry.now(
                approver_id,
                f"Decision: {'approved' if approved else 'declined'}",
            ).__dict__
        ],
    }


def execute_tool_node(state: AgentState) -> dict:
    """Runs only on the approved branch -- never reachable otherwise."""
    logger.info("Approval confirmed. Executing guideline lookup.")
    result = fetch_treatment_guideline(state["proposed_action"])
    return {
        "messages": [f"Tool output: {result}"],
        "audit_log": [AuditEntry.now("system", "Tool executed successfully.").__dict__],
    }


def cancelled_node(state: AgentState) -> dict:
    """Runs only on the declined branch."""
    logger.info("Action declined by reviewing physician. No tool was called.")
    return {
        "messages": ["Tool output: action cancelled by reviewing physician."],
        "audit_log": [AuditEntry.now("system", "Workflow halted: no approval given.").__dict__],
    }


def route_on_approval(state: AgentState) -> str:
    """Conditional edge: send the graph to the tool node only if approved."""
    return "execute_tool" if state.get("approval") == ApprovalDecision.APPROVED.value else "cancelled"


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("request_approval", request_approval_node)
    builder.add_node("execute_tool", execute_tool_node)
    builder.add_node("cancelled", cancelled_node)

    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "request_approval")
    builder.add_conditional_edges(
        "request_approval",
        route_on_approval,
        {"execute_tool": "execute_tool", "cancelled": "cancelled"},
    )
    builder.add_edge("execute_tool", END)
    builder.add_edge("cancelled", END)

    # A checkpointer is required for interrupt()/resume to work at all --
    # it's what lets the graph "remember" where it paused.
    return builder.compile(checkpointer=MemorySaver())


# --------------------------------------------------------------------------- #
# CLI demo entry point
# --------------------------------------------------------------------------- #
def run_cli_demo() -> None:
    graph = build_graph()
    thread_config = {"configurable": {"thread_id": "demo-patient-1042"}}

    initial_state: AgentState = {
        "messages": ["Patient is a 45-year-old presenting with symptoms of acute pneumonia."],
        "proposed_action": "",
        "approval": None,
        "approver_id": None,
        "audit_log": [],
    }

    result = graph.invoke(initial_state, config=thread_config)

    # The graph is now paused at the interrupt. `result["__interrupt__"]`
    # holds the payload the approval node passed to interrupt(...).
    pending = result["__interrupt__"][0].value
    print(f"\n[Approval requested]: {pending['reason']}")
    print(f"Proposed action: {pending['proposed_action']}")

    raw = input("Doctor, do you approve this action? (y/n): ").strip().lower()
    approver_id = input("Approver ID: ").strip() or "unspecified"

    final_state = graph.invoke(
        Command(resume={"approved": raw == "y", "approver_id": approver_id}),
        config=thread_config,
    )

    print("\n--- Final Workflow Messages ---")
    for msg in final_state["messages"]:
        text = msg.content if hasattr(msg, "content") else msg
        print(text)

    print("\n--- Audit Trail ---")
    for entry in final_state["audit_log"]:
        print(f"{entry['timestamp']}  [{entry['actor']}]  {entry['event']}")


if __name__ == "__main__":
    run_cli_demo()