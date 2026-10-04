"""Supervisor Node Handlers: Entry initialization and exit validation/formatting."""

import time
from typing import Dict, Any
from state import AgenticState
from agents.supervisor import SupervisorAgent
from config import config
from utils.logger import get_logger

logger = get_logger("supervisor_node", config.log_level)
supervisor_agent = SupervisorAgent()


def supervisor_init_node(state: AgenticState) -> Dict[str, Any]:
    """Graph Entry Node: Initializes execution and validates user input query.

    Args:
        state: Current AgenticState.

    Returns:
        State updates dictionary with metadata and start_time.
    """
    logger.info("--- Executing Graph Node: 'supervisor_init' ---")
    query = state.get("user_query", "")

    try:
        val_result = supervisor_agent.validate_input(query)
        meta = dict(state.get("metadata", {}))
        meta["start_time"] = val_result["start_time"]
        meta["query_length"] = val_result["query_length"]

        return {
            "metadata": meta,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Supervisor Init Node error: {e}")
        return {
            "error": str(e),
            "metadata": {"start_time": time.time()},
        }


def supervisor_validation_node(state: AgenticState) -> Dict[str, Any]:
    """Graph Exit Node: Normalizes response, computes duration, and validates output.

    Args:
        state: Current AgenticState.

    Returns:
        State updates dictionary with final_response and execution_duration.
    """
    logger.info("--- Executing Graph Node: 'supervisor_validation' ---")
    meta = dict(state.get("metadata", {}))
    start_time = meta.get("start_time", time.time())
    route = state.get("selected_route")
    raw_ans = state.get("final_response")
    citations = state.get("citations")
    tool_calls = state.get("tool_calls")
    err = state.get("error")

    result = supervisor_agent.format_final_output(
        raw_response=raw_ans,
        route=route,
        citations=citations,
        tool_calls=tool_calls,
        error=err,
        start_time=start_time,
    )

    meta["formatted_status"] = result["status"]
    meta["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "final_response": result["formatted_response"],
        "execution_duration": result["duration"],
        "metadata": meta,
    }
