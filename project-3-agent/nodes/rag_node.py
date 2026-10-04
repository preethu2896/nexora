"""RAG Node Handler: Executes document search and Q&A via RAGAdapter."""

from typing import Dict, Any
from state import AgenticState
from adapters.rag_adapter import RAGAdapter
from config import config
from utils.logger import get_logger

logger = get_logger("rag_node", config.log_level)
rag_adapter = RAGAdapter()


def rag_node(state: AgenticState) -> Dict[str, Any]:
    """LangGraph Node: Invokes RAGAdapter to query Project 1 document database.

    Args:
        state: Current AgenticState.

    Returns:
        State updates dictionary containing retrieved_context, citations, final_response.
    """
    logger.info("--- Executing Graph Node: 'rag' ---")
    query = state.get("user_query", "")

    try:
        rag_res = rag_adapter.query(question=query, top_k=4)
        
        answer = rag_res.get("answer", "")
        context = rag_res.get("context", [])
        citations = rag_res.get("citations", [])
        err = rag_res.get("error")

        logger.info(f"RAG Node successfully generated answer (citations_count={len(citations)}).")
        return {
            "retrieved_context": context if isinstance(context, list) else [context],
            "citations": citations,
            "final_response": answer,
            "error": err,
        }
    except Exception as e:
        logger.error(f"RAG Node execution failed: {e}", exc_info=True)
        return {
            "retrieved_context": [],
            "citations": [],
            "final_response": "An error occurred while accessing the RAG document database.",
            "error": str(e),
        }
