"""Query Engine: Retrieves relevant document chunks and generates grounded answers."""

import sys
from typing import Dict, Any, List
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from config import config
from utils.logger import get_logger
from ingest import get_embedding_function
from utils.rag_chain import get_llm, build_rag_chain, extract_citations

logger = get_logger("query", config.log_level)


def load_vector_store() -> Chroma:
    """Loads the persisted ChromaDB vector database.

    Returns:
        Chroma vectorstore instance.
    """
    if not config.chroma_db_dir.exists():
        logger.error(f"ChromaDB directory not found at {config.chroma_db_dir}")
        raise FileNotFoundError(
            f"Vector store database directory missing at {config.chroma_db_dir}. "
            "Please run `python ingest.py` first."
        )

    logger.info(f"Loading ChromaDB vector database from: {config.chroma_db_dir}")
    embeddings = get_embedding_function()

    try:
        vector_store = Chroma(
            collection_name="pdf_rag_collection",
            embedding_function=embeddings,
            persist_directory=str(config.chroma_db_dir),
        )
        return vector_store
    except Exception as e:
        logger.error(f"Error loading ChromaDB vector database: {e}")
        raise RuntimeError(
            f"Failed to load vector store. If you switched API keys or embedding models, "
            f"please re-ingest your documents by running `python ingest.py --reset`. Error: {e}"
        ) from e


def query_rag(question: str, top_k: int = config.top_k) -> Dict[str, Any]:
    """Executes RAG pipeline for a user question.

    Args:
        question: User query string.
        top_k: Number of relevant document chunks to retrieve.

    Returns:
        Dictionary containing question, generated answer, and source citations.
    """
    logger.info(f"Processing query: '{question}' (top_k={top_k})")

    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    # Retrieve source docs directly for citation display
    retrieved_docs = retriever.invoke(question)
    citations = extract_citations(retrieved_docs)

    logger.info(f"Retrieved {len(retrieved_docs)} relevant context chunk(s).")

    llm = get_llm()
    rag_chain = build_rag_chain(retriever, llm)

    answer = rag_chain.invoke(question)

    return {
        "question": question,
        "answer": answer.strip(),
        "sources": citations,
        "raw_docs": retrieved_docs,
    }


import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def print_query_result(result: Dict[str, Any], show_sources: bool = True) -> None:
    """Prints formatted query result and citations to CLI console.

    Args:
        result: Result dictionary from query_rag().
        show_sources: Whether to display source metadata.
    """
    print("\n" + "=" * 65)
    print(f"QUESTION: {result['question']}")
    print("=" * 65)
    print(f"ANSWER:\n{result['answer']}")

    if show_sources and result.get("sources"):
        print("\n" + "-" * 65)
        print("CITATIONS & SOURCES:")
        for idx, (file_name, page) in enumerate(result["sources"], start=1):
            print(f"   [{idx}] Document: '{file_name}' - Page {page}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query the PDF RAG Vector Store.")
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        required=True,
        help="The question to ask about the ingested PDF documents.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.top_k,
        help="Number of context chunks to retrieve (default: top_k in .env)",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Hide source citation details in output",
    )

    args = parser.parse_args()

    try:
        res = query_rag(question=args.question, top_k=args.top_k)
        print_query_result(res, show_sources=not args.no_sources)
    except Exception as err:
        logger.critical(f"Query execution failed: {err}", exc_info=True)
        sys.exit(1)
