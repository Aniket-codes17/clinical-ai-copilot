"""
tools/rag_tool.py
------------------
Retrieval over clinical guideline documents, backed by a persisted Chroma
vector store.

Split into two distinct operations, which the original script conflated:

1. `build_index()` -- expensive, run once (or whenever source documents
   change). Loads documents, chunks them, embeds them, and persists the
   result to disk.
2. `query_clinical_rag()` -- cheap, called on every user question from the
   Streamlit app. Opens the *already persisted* index and retrieves.

Re-embedding the entire document set on every question -- which is what
the original script did, since it ran the whole pipeline top to bottom
every time -- would make each search in the app take as long as a full
reindex. This version only pays that cost once.
"""

from __future__ import annotations

import argparse
import logging
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("clinical_agent.rag_tool")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DOCS_DIR = Path("./sample-docs")
PERSIST_DIR = Path("./chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 300/30 chars is quite small for guideline prose and risks splitting a
# recommendation mid-sentence. 800/120 keeps most clinical recommendations
# intact in a single chunk while still being small enough for precise
# retrieval. Tune per corpus if needed.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 3


# --------------------------------------------------------------------------- #
# Index building (offline / one-time step)
# --------------------------------------------------------------------------- #
def build_index(
    docs_path: Path = DOCS_DIR,
    persist_directory: Path = PERSIST_DIR,
) -> int:
    """Load, chunk, embed, and persist the document set. Returns the
    number of chunks indexed. Run this via the CLI (see bottom of file)
    whenever source documents change -- not from the app on every query.
    """
    if not docs_path.exists():
        raise FileNotFoundError(f"Document directory not found at '{docs_path}'.")

    loader = DirectoryLoader(
        str(docs_path), glob="**/*.txt", loader_cls=TextLoader, recursive=True
    )
    documents = loader.load()
    if not documents:
        raise ValueError(f"No .txt documents found under '{docs_path}'.")
    logger.info("Loaded %d document(s) from %s.", len(documents), docs_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunk(s).", len(chunks))

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(chunks, embeddings, persist_directory=str(persist_directory))
    logger.info("Persisted vector store to %s.", persist_directory)

    return len(chunks)


# --------------------------------------------------------------------------- #
# Query path (used by the app, cheap after the first call)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _get_vectorstore() -> Chroma:
    if not PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"No vector store found at '{PERSIST_DIR}'. Build it first with: "
            f"python -m tools.rag_tool build"
        )
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embeddings)


def _run_similarity_search(query: str, k: int) -> tuple[list[tuple], str | None]:
    """Shared retrieval step used by both the HTML (UI) and plain-text
    (LLM tool) formatters below, so there's one place that talks to the
    vector store. Returns (results, error_message)."""
    if not query or not query.strip():
        return [], "Enter a question to search the guideline index."

    try:
        vectorstore = _get_vectorstore()
    except FileNotFoundError as exc:
        logger.warning(str(exc))
        return [], str(exc)

    try:
        results = vectorstore.similarity_search_with_score(query, k=k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG query failed.")
        return [], f"Guideline search failed: {exc}"

    logger.info("Query '%s' returned %d result(s).", query, len(results))
    return results, None


def _format_results_html(results: list[tuple]) -> str:
    """Render retrieved chunks as source-cited HTML cards, matching the
    app's existing `.panel` / border-left card styling. For display in
    the Streamlit UI only -- see `_format_results_text` for LLM/tool use."""
    if not results:
        return (
            "<p>No relevant guideline passages were found for this query. "
            "Try rephrasing, or confirm the guideline index has been built.</p>"
        )

    cards = []
    for idx, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "Unknown source")
        relevance = max(0.0, 1 - score) * 100  # Chroma returns a distance; smaller is closer
        cards.append(
            f"""
            <div style="border-left:3px solid #0F766E; padding:10px 14px; margin-bottom:10px;">
                <div style="font-size:0.78rem; color:#64748B; margin-bottom:4px;">
                    Source {idx} &middot; {source} &middot; {relevance:.0f}% match
                </div>
                <div style="font-size:0.9rem; color:#0B2545; line-height:1.5;">
                    {doc.page_content}
                </div>
            </div>
            """
        )
    return "".join(cards)


def _format_results_text(results: list[tuple]) -> str:
    """Render retrieved chunks as plain text, suitable for putting into an
    LLM's context window (e.g. as a LangChain tool result) -- no HTML, no
    inline styling, just source-attributed passages."""
    if not results:
        return "No relevant guideline passages were found for this query."

    lines = []
    for idx, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "Unknown source")
        relevance = max(0.0, 1 - score) * 100
        lines.append(f"[{idx}] Source: {source} ({relevance:.0f}% match)\n{doc.page_content}")
    return "\n\n".join(lines)


def query_clinical_rag(query: str, k: int = DEFAULT_TOP_K) -> str:
    """Retrieve the top-k most relevant guideline chunks for `query` and
    return them as HTML formatted for display in the Streamlit app, each
    with its source and a relevance score. Never raises. For LLM/agent
    tool use, call `query_clinical_rag_text` instead -- HTML markup in an
    LLM's context wastes tokens and can confuse the model.
    """
    results, error = _run_similarity_search(query, k)
    if error:
        return f"<p>{error}</p>"
    return _format_results_html(results)


def query_clinical_rag_text(query: str, k: int = DEFAULT_TOP_K) -> str:
    """Same retrieval as `query_clinical_rag`, but returns plain text with
    no markup -- this is the one LangChain tool wrappers should call."""
    results, error = _run_similarity_search(query, k)
    if error:
        return error
    return _format_results_text(results)


# --------------------------------------------------------------------------- #
# CLI: build the index, or run a one-off query from the terminal
# --------------------------------------------------------------------------- #
def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

    parser = argparse.ArgumentParser(description="Build or query the clinical guideline index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="(Re)build the vector index from ./sample-docs.")
    build_parser.add_argument("--docs-path", type=Path, default=DOCS_DIR)
    build_parser.add_argument("--persist-dir", type=Path, default=PERSIST_DIR)

    query_parser = subparsers.add_parser("query", help="Run a one-off query against the index.")
    query_parser.add_argument("text", type=str)
    query_parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)

    args = parser.parse_args()

    if args.command == "build":
        n_chunks = build_index(docs_path=args.docs_path, persist_directory=args.persist_dir)
        print(f"Indexed {n_chunks} chunks into {args.persist_dir}.")
    elif args.command == "query":
        print(query_clinical_rag(args.text, k=args.k))


if __name__ == "__main__":
    _main()