"""PDF Loading and Text Chunking Utilities using PyPDFLoader and RecursiveCharacterTextSplitter."""

from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import get_logger
from config import config

logger = get_logger(__name__, config.log_level)


def load_pdf(file_path: Path) -> List[Document]:
    """Loads a single PDF document using PyPDFLoader.

    Args:
        file_path: Path object pointing to the PDF file.

    Returns:
        List of loaded LangChain Document objects.
    """
    if not file_path.exists():
        logger.error(f"PDF file not found: {file_path}")
        raise FileNotFoundError(f"PDF file does not exist at {file_path}")

    if file_path.suffix.lower() != ".pdf":
        logger.error(f"File is not a PDF: {file_path}")
        raise ValueError(f"Expected a .pdf file, but got {file_path.suffix}")

    try:
        logger.info(f"Loading PDF document: {file_path.name}")
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} pages from {file_path.name}")
        return documents
    except Exception as e:
        logger.error(f"Failed to load PDF {file_path}: {e}")
        raise RuntimeError(f"Error loading PDF {file_path}: {e}") from e


def load_all_pdfs(directory_path: Path) -> List[Document]:
    """Scans a directory for all .pdf files and loads them.

    Args:
        directory_path: Path to the directory containing PDFs.

    Returns:
        Aggregated list of Document objects from all valid PDFs.
    """
    if not directory_path.exists() or not directory_path.is_dir():
        logger.error(f"Directory not found: {directory_path}")
        raise FileNotFoundError(f"Data directory does not exist: {directory_path}")

    raw_files = list(directory_path.glob("*.pdf")) + list(directory_path.glob("*.PDF"))
    seen_paths = set()
    pdf_files = []
    for f in raw_files:
        resolved = f.resolve()
        if resolved not in seen_paths:
            seen_paths.add(resolved)
            pdf_files.append(f)

    if not pdf_files:
        logger.warning(f"No PDF files found in directory: {directory_path}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF file(s) in {directory_path}")

    all_docs: List[Document] = []
    for pdf in pdf_files:
        try:
            docs = load_pdf(pdf)
            all_docs.extend(docs)
        except Exception as e:
            logger.warning(f"Skipping corrupt or unreadable PDF {pdf.name}: {e}")

    logger.info(f"Total loaded pages across all PDFs: {len(all_docs)}")
    return all_docs


def split_documents(
    documents: List[Document],
    chunk_size: int = config.chunk_size,
    chunk_overlap: int = config.chunk_overlap,
) -> List[Document]:
    """Splits documents into smaller semantic chunks using RecursiveCharacterTextSplitter.

    Args:
        documents: List of loaded Document objects.
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Overlap character count between consecutive chunks.

    Returns:
        List of chunked Document objects.
    """
    if not documents:
        logger.warning("No documents provided to split.")
        return []

    logger.info(
        f"Splitting {len(documents)} document pages (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})..."
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    logger.info(f"Generated {len(chunks)} text chunks.")
    return chunks
