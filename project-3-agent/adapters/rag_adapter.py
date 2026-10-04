"""RAG Adapter: Reusable non-invasive adapter for Project 1 (PDF RAG Application)."""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from config import config
from utils.logger import get_logger
from utils.exceptions import RAGAdapterError

logger = get_logger("rag_adapter", config.log_level)


class RAGAdapter:
    """Adapter for interacting with Project 1 (RAG system) without code modification."""

    def __init__(self, project_1_dir: Optional[Path] = None):
        """Initializes RAG Adapter and registers Project 1 directory in sys.path.

        Args:
            project_1_dir: Path to project-1-rag directory.
        """
        if project_1_dir is None:
            project_1_dir = config.project_1_dir

        self.project_1_dir = project_1_dir.resolve()
        self._ensure_sys_path()

    def _ensure_sys_path(self) -> None:
        """Injects project-1-rag directory into Python sys.path."""
        p1_str = str(self.project_1_dir)
        if p1_str in sys.path:
            sys.path.remove(p1_str)
        sys.path.insert(0, p1_str)
        if "config" in sys.modules:
            del sys.modules["config"]
        if "utils" in sys.modules and not hasattr(sys.modules["utils"], "pdf_processor"):
            del sys.modules["utils"]
        if "prompts" in sys.modules and not hasattr(sys.modules["prompts"], "SYSTEM_PROMPT"):
            del sys.modules["prompts"]
        import config as loaded_config
        if hasattr(loaded_config, "config"):
            cfg = getattr(loaded_config, "config")
            if not hasattr(cfg, "project_1_dir"):
                setattr(cfg, "project_1_dir", self.project_1_dir)
            if not hasattr(cfg, "project_2_dir"):
                setattr(cfg, "project_2_dir", self.project_1_dir.parent / "project-2-mcp")

    def health_check(self) -> Dict[str, Any]:
        """Checks readiness of Project 1 RAG database and configurations.

        Returns:
            Status dictionary with health metrics.
        """
        logger.info("Executing RAGAdapter health check...")
        try:
            self._ensure_sys_path()
            from config import config as p1_config
            
            chroma_dir = p1_config.chroma_db_dir
            db_exists = chroma_dir.exists() and any(chroma_dir.iterdir()) if chroma_dir.exists() else False

            status = {
                "status": "healthy" if db_exists else "degraded",
                "chroma_db_dir": str(chroma_dir),
                "chroma_db_exists": db_exists,
                "embedding_model": p1_config.embedding_model_name,
                "data_dir": str(p1_config.data_dir),
            }
            logger.info(f"RAGAdapter health status: {status['status']}")
            return status
        except Exception as e:
            logger.error(f"RAGAdapter health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    def query(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        """Executes a full RAG query using Project 1's query pipeline.

        Args:
            question: User question string.
            top_k: Number of context chunks to retrieve.

        Returns:
            Dictionary containing 'question', 'answer', 'context', and 'citations'.
        """
        logger.info(f"RAGAdapter querying Project 1 for: '{question}' (top_k={top_k})")
        try:
            self._ensure_sys_path()
            from query import query_rag
            
            result = query_rag(question=question, top_k=top_k)
            logger.info("RAGAdapter query successfully completed.")
            return result
        except FileNotFoundError as fnf_err:
            logger.warning(f"RAGAdapter database missing: {fnf_err}")
            return {
                "question": question,
                "answer": "The vector database is missing. Please ingest PDF documents into Project 1 first.",
                "context": [],
                "citations": [],
                "error": str(fnf_err),
            }
        except Exception as e:
            logger.error(f"RAGAdapter query execution failed: {e}", exc_info=True)
            raise RAGAdapterError(f"Failed to execute RAG query in Project 1: {e}") from e

    def retrieve(self, question: str, top_k: int = 4) -> List[Any]:
        """Retrieves raw relevant document chunks without invoking LLM generation.

        Args:
            question: Search query string.
            top_k: Number of chunks to retrieve.

        Returns:
            List of retrieved Document objects.
        """
        logger.info(f"RAGAdapter retrieving context chunks for: '{question}'")
        try:
            self._ensure_sys_path()
            from query import load_vector_store
            
            vector_store = load_vector_store()
            docs = vector_store.similarity_search(question, k=top_k)
            logger.info(f"RAGAdapter retrieved {len(docs)} context chunk(s).")
            return docs
        except Exception as e:
            logger.error(f"RAGAdapter retrieval failed: {e}")
            raise RAGAdapterError(f"Failed to retrieve context from Project 1: {e}") from e
