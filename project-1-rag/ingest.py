"""Ingestion pipeline: Load PDFs, chunk text, generate embeddings, and persist to ChromaDB."""

import sys
import os
from pathlib import Path
from typing import Optional, Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from config import config
from utils.logger import get_logger
from utils.pdf_processor import load_pdf, load_all_pdfs, split_documents

logger = get_logger("ingest", config.log_level)


def get_embedding_function() -> Any:
    """Initializes and returns Google Generative AI embedding model.

    Returns:
        Configured GoogleGenerativeAIEmbeddings instance.
    """
    config.validate_api_key()
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    model_name = config.embedding_model_name
    if model_name.startswith("models/"):
        model_name = model_name[len("models/"):]
    if not ("gemini" in model_name.lower() or "001" in model_name.lower()):
        model_name = "gemini-embedding-001"

    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=config.api_key,
    )


def ingest_documents(source_path: Optional[Path] = None, clear_existing: bool = False) -> int:
    """Runs the document ingestion process.

    Args:
        source_path: Optional path to a specific PDF file or folder. Defaults to config.data_dir.
        clear_existing: If True, resets/overwrites existing Chroma vector store.

    Returns:
        Number of chunked documents stored in ChromaDB.
    """
    target = source_path or config.data_dir

    logger.info("=" * 60)
    logger.info(f"Starting Document Ingestion Pipeline target: {target}")
    logger.info("=" * 60)

    raw_documents = []
    if target.is_file():
        raw_documents = load_pdf(target)
    elif target.is_dir():
        raw_documents = load_all_pdfs(target)

    if not raw_documents:
        print("\n" + "=" * 65)
        print(" [!] No PDF documents found in default data folder.")
        print(" Please provide the path to the PDF file or directory you want to ingest.")
        print("=" * 65)
        try:
            user_path_str = input("\nEnter path to your PDF file (or folder): ").strip().strip('"').strip("'")
            if user_path_str:
                user_p = Path(user_path_str).resolve()
                if user_p.is_file():
                    raw_documents = load_pdf(user_p)
                elif user_p.is_dir():
                    raw_documents = load_all_pdfs(user_p)
                else:
                    logger.error(f"Provided path does not exist: {user_p}")
        except (KeyboardInterrupt, EOFError):
            print("\nIngestion cancelled by user.")

    if not raw_documents:
        logger.warning("No document content found to ingest. Aborting ingestion.")
        return 0

    # Split documents into chunks
    chunks = split_documents(
        raw_documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    if not chunks:
        logger.warning("Chunking resulted in 0 text segments. Aborting.")
        return 0

    logger.info(f"Initializing Embedding Model ({config.embedding_model_name})...")
    embeddings = get_embedding_function()

    logger.info(f"Connecting to ChromaDB persistent directory: {config.chroma_db_dir}")

    if clear_existing:
        logger.info("Clearing existing vector store collection...")
        try:
            vector_store = Chroma(
                collection_name="pdf_rag_collection",
                embedding_function=embeddings,
                persist_directory=str(config.chroma_db_dir),
            )
            vector_store.delete_collection()
        except Exception as e:
            logger.info(f"Note on clearing collection: {e}")

    logger.info(f"Generating embeddings and indexing {len(chunks)} chunks into ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="pdf_rag_collection",
        persist_directory=str(config.chroma_db_dir),
    )

    logger.info("=" * 60)
    logger.info(f"Successfully ingested and persisted {len(chunks)} chunks to ChromaDB!")
    logger.info(f"Vector Database Location: {config.chroma_db_dir.resolve()}")
    logger.info("=" * 60)

    return len(chunks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PDF documents into ChromaDB Vector Store.")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to a PDF file or directory containing PDFs (default: data_dir in .env)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset existing vector collection before indexing",
    )

    args = parser.parse_args()
    source_p = Path(args.source) if args.source else None

    try:
        count = ingest_documents(source_path=source_p, clear_existing=args.reset)
        if count == 0:
            sys.exit(1)
    except Exception as err:
        logger.critical(f"Ingestion failed: {err}", exc_info=True)
        sys.exit(1)
