"""MCP Node Handler: Executes MCP tools via MCPAdapter."""

from typing import Dict, Any
from state import AgenticState
from adapters.mcp_adapter import MCPAdapter
from config import config
from utils.logger import get_logger

logger = get_logger("mcp_node", config.log_level)
mcp_adapter = MCPAdapter()


async def mcp_node(state: AgenticState) -> Dict[str, Any]:
    """LangGraph Node: Invokes MCPAdapter to execute tools in Project 2.

    Args:
        state: Current AgenticState.

    Returns:
        State updates dictionary containing tool_calls, tool_name, final_response.
    """
    logger.info("--- Executing Graph Node: 'mcp' ---")
    query = state.get("user_query", "")
    history = state.get("messages", [])

    try:
        if mcp_adapter.graph is None:
            await mcp_adapter.initialize()

        mcp_res = await mcp_adapter.execute(user_input=query, history=history)

        answer = mcp_res.get("answer", "")
        tool_calls = mcp_res.get("tool_calls", [])
        tool_name = mcp_res.get("tool_name")

        logger.info(f"MCP Node successfully executed query (tools_called={len(tool_calls)}).")
        return {
            "tool_calls": tool_calls,
            "tool_name": tool_name,
            "final_response": answer,
            "error": None,
        }
    except Exception as e:
        logger.error(f"MCP Node execution failed: {e}", exc_info=True)
        return {
            "tool_calls": [],
            "tool_name": None,
            "final_response": "An error occurred while executing the MCP tool toolset.",
            "error": str(e),
        }
