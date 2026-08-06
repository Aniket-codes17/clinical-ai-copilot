"""
tools/llm_tool.py
------------------
Thin, reusable wrapper around a local Ollama model for clinical Q&A.

The original script only ever ran one hardcoded query and had no error
handling at all -- if Ollama wasn't running, the model hadn't been pulled
(`ollama pull llama3.2:1b`), or the connection simply timed out on a cold
start, it would crash with a raw exception. Since a local model server is
one of the more fragile parts of a demo to keep running, this version
assumes that failure mode is the common case, not the exception.

It's also written so it can optionally take retrieved context (e.g. from
`tools/rag_tool.py`) and synthesize it into a direct answer, rather than
only ever answering from the model's own training data.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

logger = logging.getLogger("clinical_agent.llm_tool")

# --------------------------------------------------------------------------- #
# Config -- overridable via environment variables so the model/host can
# change per environment without touching code.
# --------------------------------------------------------------------------- #
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "30"))

SYSTEM_PROMPT = (
    "You are an AI clinical decision-support assistant aiding a physician. "
    "Provide concise, professional, evidence-based information. "
    "State clearly when evidence is limited or when the question requires "
    "clinical judgment you cannot provide. You are a decision-support tool, "
    "not a diagnostic authority -- the physician makes the final call."
)

CONTEXT_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    "\n\nUse the provided guideline excerpts as your primary source. "
    "If they don't answer the question, say so explicitly rather than "
    "filling the gap from general knowledge without flagging it."
)


class LLMUnavailableError(RuntimeError):
    """Raised when the local model can't be reached or fails to respond."""


# --------------------------------------------------------------------------- #
# Model initialization (cached -- one client per process)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _get_llm() -> ChatOllama:
    logger.info(
        "Connecting to Ollama model '%s' at %s...", OLLAMA_MODEL, OLLAMA_BASE_URL
    )
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=OLLAMA_TEMPERATURE,
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )


def _build_chain(system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{question}")]
    )
    return prompt | _get_llm()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def ask_clinical_llm(question: str, context: str | None = None) -> str:
    """Ask the local clinical LLM a question, optionally grounded in
    retrieved guideline text (e.g. from `query_clinical_rag`).

    Returns the model's answer as plain text. Never raises -- connection
    and runtime failures are caught and returned as a readable message,
    since this is meant to be called directly from UI code.
    """
    if not question or not question.strip():
        return "Please provide a question."

    try:
        if context and context.strip():
            chain = _build_chain(CONTEXT_SYSTEM_PROMPT)
            payload = {"question": f"Guideline excerpts:\n{context}\n\nQuestion: {question}"}
        else:
            chain = _build_chain(SYSTEM_PROMPT)
            payload = {"question": question}

        response = chain.invoke(payload)
        return response.content

    except Exception as exc:  # noqa: BLE001 -- connection refused, timeout, model not pulled, etc.
        logger.exception("LLM call failed.")
        return (
            "The clinical AI assistant is currently unavailable "
            f"({exc.__class__.__name__}). Confirm Ollama is running "
            f"(`ollama serve`) and that '{OLLAMA_MODEL}' has been pulled "
            f"(`ollama pull {OLLAMA_MODEL}`)."
        )


def check_llm_health() -> tuple[bool, str]:
    """Lightweight connectivity check, useful for a startup banner or a
    'system status' indicator in the app rather than failing mid-query."""
    try:
        _get_llm().invoke("ping")
        return True, f"Connected to '{OLLAMA_MODEL}' at {OLLAMA_BASE_URL}."
    except Exception as exc:  # noqa: BLE001
        return False, f"Cannot reach '{OLLAMA_MODEL}' at {OLLAMA_BASE_URL}: {exc}"


# --------------------------------------------------------------------------- #
# CLI demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

    healthy, status = check_llm_health()
    print(f"[{'OK' if healthy else 'UNAVAILABLE'}] {status}\n")

    if healthy:
        query = "What are the primary clinical indicators of acute pneumonia on a chest X-ray?"
        print(f"Sending query: {query}\n")
        print("--- Clinical AI Response ---")
        print(ask_clinical_llm(query))