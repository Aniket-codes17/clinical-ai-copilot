import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Load environment variables
load_dotenv()

# 1. Initialize API client pointing to local Ollama instance
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.2,
)

# 2. Define a structured clinical prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert AI clinical decision-support assistant aiding a doctor. "
            "Provide concise, professional, and evidence-based information.",
        ),
        ("human", "{question}"),
    ]
)

# 3. Combine into a LangChain execution chain
chain = prompt | llm

# 4. Run sample query
if __name__ == "__main__":
    query = "What are the primary clinical indicators of acute pneumonia on a chest X-ray?"
    print(f"Sending Query: {query}\n")

    response = chain.invoke({"question": query})

    print("--- Clinical AI Response ---")
    print(response.content)