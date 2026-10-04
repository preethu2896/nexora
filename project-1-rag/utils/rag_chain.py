import os
from pathlib import Path
from typing import List, Tuple, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from config import config
from utils.logger import get_logger

logger = get_logger(__name__, config.log_level)

# Strict anti-hallucination prompt template
SYSTEM_PROMPT = """You are an accurate, truthful AI assistant specialized in answering questions from technical documents.

Answer the user's question ONLY using the provided document context below.

CRITICAL INSTRUCTIONS:
1. If the information required to answer the question is NOT present in the context, explicitly state:
   "The requested information is unavailable in the provided document(s)."
2. Do NOT attempt to answer using pre-trained external knowledge or make assumptions beyond the text.
3. Keep the answer clear, precise, and directly relevant to the question.

---
DOCUMENT CONTEXT:
{context}
---
QUESTION:
{question}

ANSWER:"""


def get_llm() -> Any:
    """Initializes and returns Google Generative AI Chat Model.

    Returns:
        Configured ChatGoogleGenerativeAI instance.
    """
    config.validate_api_key()
    from langchain_google_genai import ChatGoogleGenerativeAI

    model_name = config.llm_model_name
    if model_name.startswith("models/"):
        model_name = model_name[len("models/"):]

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=config.api_key,
        temperature=0.0,
    )


def format_docs(docs: List[Document]) -> str:
    """Formats retrieved Document objects into structured text block for LLM prompt context.

    Args:
        docs: List of retrieved Documents.

    Returns:
        Formatted string containing content and source metadata.
    """
    if not docs:
        return "No relevant document context found."

    formatted_blocks = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown file")
        page = doc.metadata.get("page", 0) + 1  # 1-indexed page display
        block = f"[Doc {idx} | Source: {source} | Page: {page}]\n{doc.page_content.strip()}"
        formatted_blocks.append(block)

    return "\n\n".join(formatted_blocks)


def extract_citations(docs: List[Document]) -> List[Tuple[str, int]]:
    """Extracts unique source file names and page numbers from document metadata.

    Args:
        docs: List of retrieved Document objects.

    Returns:
        List of tuples (file_name, page_number).
    """
    citations = []
    seen = set()

    for doc in docs:
        source_path = doc.metadata.get("source", "Unknown file")
        file_name = Path(source_path).name if source_path != "Unknown file" else "Unknown file"
        page = doc.metadata.get("page", 0) + 1  # 1-indexed

        key = (file_name, page)
        if key not in seen:
            seen.add(key)
            citations.append(key)

    return citations


def build_rag_chain(retriever, llm: Any):
    """Assembles LCEL RAG chain (Context Retrieval -> Anti-hallucination Prompt -> LLM -> String Output).

    Args:
        retriever: Chroma vector store retriever object.
        llm: Initialized ChatOpenAI instance.

    Returns:
        LangChain Runnable chain.
    """
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
