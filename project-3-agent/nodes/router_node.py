"""Router Node Handler: Invokes RouterAgent for semantic intent classification."""

from typing import Dict, Any
from state import AgenticState
from agents.router import RouterAgent
from config import config
from utils.logger import get_logger

logger = get_logger("router_node", config.log_level)
router_agent = RouterAgent()


async def router_node(state: AgenticState) -> Dict[str, Any]:
    """LangGraph Node: Performs semantic intent classification via RouterAgent.

    Args:
        state: Current AgenticState.

    Returns:
        State updates dictionary containing selected_route, route_confidence, route_reasoning.
    """
    logger.info("--- Executing Graph Node: 'router' ---")
    query = state.get("user_query", "")
    history = state.get("messages", [])

    try:
        decision = await router_agent.aclassify(user_query=query, history=history)
        logger.info(
            f"Router Node Classification Result: Route='{decision.route}', "
            f"Confidence={decision.confidence:.2f}"
        )
        return {
            "selected_route": decision.route,
            "route_confidence": decision.confidence,
            "route_reasoning": decision.reasoning,
        }
    except Exception as e:
        logger.error(f"Router Node execution failed: {e}", exc_info=True)
        return {
            "selected_route": "RAG",
            "route_confidence": 0.0,
            "route_reasoning": f"Router exception fallback: {str(e)}",
            "error": str(e),
        }
