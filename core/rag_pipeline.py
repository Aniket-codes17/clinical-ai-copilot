import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load clinical documents
docs_path = "./sample-docs"
loader = DirectoryLoader(docs_path, glob="*.txt", loader_cls=TextLoader)
documents = loader.load()

# 2. Chunk documents into searchable pieces
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=30
)
chunks = text_splitter.split_documents(documents)

# 3. Create vector embeddings & store in Chroma DB
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(
    chunks, embeddings, persist_directory="./chroma_db"
)

# 4. Search relevant context with source citations
query = "What is the empiric outpatient treatment for healthy adults with pneumonia?"
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
relevant_docs = retriever.invoke(query)

print(f"Query: {query}\n")
for idx, doc in enumerate(relevant_docs):
    source = doc.metadata.get("source", "Unknown")
    print(f"--- Source {idx+1}: {source} ---")
    print(f"{doc.page_content}\n")