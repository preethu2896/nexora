"""LangGraph StateGraph workflow construction, nodes, and conditional routing logic."""

import asyncio
from typing import List, Dict, Any, Literal
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, BaseMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, START, END
from state import AgentState
from prompts import SYSTEM_PROMPT
from config import config
from utils.logger import get_logger
from mcp_service.client import MCPClient

logger = get_logger("graph", config.log_level)


def create_agent_graph(mcp_client: MCPClient, langchain_tools: List[StructuredTool]):
    """Builds and compiles the LangGraph StateGraph workflow.

    Args:
        mcp_client: Connected MCPClient instance.
        langchain_tools: List of LangChain tools converted from MCP server.

    Returns:
        Compiled StateGraph instance.
    """
    config.validate_api_key()

    # Initialize Chat Model (Google Gemini)
    from langchain_google_genai import ChatGoogleGenerativeAI

    model_name = config.llm_model_name
    if model_name.startswith("models/"):
        model_name = model_name[len("models/"):]

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=config.openai_api_key,
        temperature=0.0,
    )

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(langchain_tools)

    # Define Agent Node
    async def agent_node(state: AgentState) -> Dict[str, Any]:
        logger.info("Executing Graph Node: 'agent'")
        messages = list(state["messages"])

        # Inject system prompt at start if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # Define MCP Tools Execution Node
    async def tools_node(state: AgentState) -> Dict[str, Any]:
        logger.info("Executing Graph Node: 'tools'")
        last_message = state["messages"][-1]
        tool_messages = []

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for call in last_message.tool_calls:
                tool_name = call["name"]
                tool_args = call.get("args", {})
                tool_call_id = call["id"]

                logger.info(f"LangGraph Tool Node executing MCP Tool '{tool_name}' with args: {tool_args}")
                try:
                    result_content = await mcp_client.execute_tool(tool_name, tool_args)
                except Exception as e:
                    logger.error(f"Error executing MCP Tool '{tool_name}': {e}")
                    result_content = f"Error executing tool {tool_name}: {str(e)}"

                tool_msg = ToolMessage(content=str(result_content), tool_call_id=tool_call_id)
                tool_messages.append(tool_msg)

        return {"messages": tool_messages}

    # Define Conditional Router
    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.info(f"Conditional Router: Found {len(last_message.tool_calls)} tool call(s) -> Routing to 'tools' node")
            return "tools"

        logger.info("Conditional Router: No tool calls -> Routing to END")
        return END

    # Assemble StateGraph
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )
    workflow.add_edge("tools", "agent")

    compiled_graph = workflow.compile()
    logger.info("LangGraph StateGraph compiled successfully.")
    return compiled_graph
