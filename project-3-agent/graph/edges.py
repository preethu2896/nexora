"""Conditional Routing Edges for Project 3 LangGraph Workflow."""

from typing import Literal
from state import AgenticState
from config import config
from utils.logger import get_logger

logger = get_logger("graph_edges", config.log_level)


def route_edge(state: AgenticState) -> Literal["RAG", "MCP"]:
    """Evaluates state.selected_route from RouterDecision and determines next graph node.

    Args:
        state: Current AgenticState.

    Returns:
        Next node key ('RAG' or 'MCP').
    """
    selected_route = state.get("selected_route")
    logger.info(f"Evaluating Conditional Edge: selected_route='{selected_route}'")

    if selected_route == "MCP":
        return "MCP"
    elif selected_route == "RAG":
        return "RAG"
    else:
        logger.warning(f"Invalid or missing route '{selected_route}'. Falling back to 'RAG'.")
        return "RAG"
