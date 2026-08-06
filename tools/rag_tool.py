"""
tools/rag_tool.py
------------------
Retrieval over clinical guideline documents, backed by a persisted Chroma
vector store. Returns plain-text, source-cited results -- suitable both
for direct display and for passing into an LLM's context (e.g. via
tools/agent_tools.py), since there's no markup to strip out either way.
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

DOCS_DIR = Path("./sample-docs")
DB_DIR = Path("./chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
DEFAULT_K = 2


# --------------------------------------------------------------------------- #
# Index building -- run this once (or whenever ./sample-docs changes), not
# from the query path. If you already have a separate ingestion script,
# this is redundant and you can drop it; kept here so this module is
# self-contained.
# --------------------------------------------------------------------------- #
def build_index(docs_path: Path = DOCS_DIR, db_dir: Path = DB_DIR) -> int:
    """Load, chunk, embed, and persist the document set. Returns the
    number of chunks indexed."""
    if not docs_path.exists():
        raise FileNotFoundError(f"Document directory not found at '{docs_path}'.")

    loader = DirectoryLoader(str(docs_path), glob="**/*.txt", loader_cls=TextLoader, recursive=True)
    documents = loader.load()
    if not documents:
        raise ValueError(f"No .txt documents found under '{docs_path}'.")
    logger.info("Loaded %d document(s) from %s.", len(documents), docs_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunk(s).", len(chunks))

    embeddings = _load_embeddings()
    Chroma.from_documents(chunks, embeddings, persist_directory=str(db_dir))
    logger.info("Persisted vector store to %s.", db_dir)

    return len(chunks)


# --------------------------------------------------------------------------- #
# Cached loaders -- the embedding model and vector store are loaded once
# per process and reused, instead of being reconstructed on every query.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _load_embeddings() -> HuggingFaceEmbeddings:
    logger.info("Loading embedding model '%s'...", EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _load_vectorstore(db_dir: str) -> Chroma:
    logger.info("Opening vector store at %s...", db_dir)
    return Chroma(persist_directory=db_dir, embedding_function=_load_embeddings())


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def query_clinical_rag(
    query_text: str,
    db_dir: str = str(DB_DIR),
    k: int = DEFAULT_K,
) -> str:
    """Queries the Chroma vector store for relevant clinical guidelines and
    returns structured, source-cited text. Never raises -- any failure
    comes back as a readable "Error: ..." string instead, since this is
    called directly from both the UI and LLM tool layers.
    """
    if not query_text or not query_text.strip():
        return "Error: no query text provided."

    if not Path(db_dir).exists():
        return f"Error: clinical knowledge base not found at '{db_dir}'. Build it first with build_index()."

    try:
        vectorstore = _load_vectorstore(db_dir)
        results = vectorstore.similarity_search_with_score(query_text, k=k)
    except Exception as exc:  # noqa: BLE001 -- surface to the caller rather than crash it
        logger.exception("RAG query failed for '%s'.", query_text)
        return f"Error: guideline search failed ({exc})."

    if not results:
        return "No relevant clinical evidence found in knowledge base."

    citations = []
    for idx, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "Unknown Document")
        relevance = max(0.0, 1 - score) * 100  # Chroma returns a distance; smaller is closer
        citations.append(
            f"--- Citation {idx} [{source}, {relevance:.0f}% match] ---\n{doc.page_content}"
        )

    logger.info("Query '%s' returned %d result(s).", query_text, len(results))
    return "\n\n".join(citations)


# --------------------------------------------------------------------------- #
# CLI: build the index, or run a one-off query from the terminal
# --------------------------------------------------------------------------- #
def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

    parser = argparse.ArgumentParser(description="Build or query the clinical guideline index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="(Re)build the vector index from ./sample-docs.")
    build_parser.add_argument("--docs-path", type=Path, default=DOCS_DIR)
    build_parser.add_argument("--db-dir", type=Path, default=DB_DIR)

    query_parser = subparsers.add_parser("query", help="Run a one-off query against the index.")
    query_parser.add_argument("text", type=str)
    query_parser.add_argument("--k", type=int, default=DEFAULT_K)

    args = parser.parse_args()

    if args.command == "build":
        n_chunks = build_index(docs_path=args.docs_path, db_dir=args.db_dir)
        print(f"Indexed {n_chunks} chunks into {args.db_dir}.")
    elif args.command == "query":
        print("\n--- Testing RAG Search Tool ---")
        print(query_clinical_rag(args.text, k=args.k))


if __name__ == "__main__":
    _main()