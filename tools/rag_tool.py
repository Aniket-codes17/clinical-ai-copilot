import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def query_clinical_rag(
    query_text: str,
    db_dir: str = "./chroma_db",
    k: int = 2,
) -> str:
    """Queries the Chroma vector store for relevant clinical guidelines and returns structured citations."""
    if not os.path.exists(db_dir):
        return "Error: Clinical knowledge base (vector DB) does not exist."

    # Load embedding model and vector database
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=db_dir, embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query_text)

    if not docs:
        return "No relevant clinical evidence found in knowledge base."

    results = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown Document")
        results.append(
            f"--- Citation {idx} [{source}] ---\n{doc.page_content}"
        )

    return "\n\n".join(results)


if __name__ == "__main__":
    sample_query = "What is the empiric treatment for outpatient pneumonia?"
    print("\n--- Testing RAG Search Tool ---")
    print(query_clinical_rag(sample_query))