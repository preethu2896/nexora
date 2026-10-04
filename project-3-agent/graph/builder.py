"""LangGraph Builder: Compiles and exports the Project 3 StateGraph workflow."""

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from state import AgenticState
from nodes.supervisor_node import supervisor_init_node, supervisor_validation_node
from nodes.router_node import router_node
from nodes.rag_node import rag_node
from nodes.mcp_node import mcp_node
from graph.edges import route_edge
from config import config
from utils.logger import get_logger

logger = get_logger("graph_builder", config.log_level)


def create_agentic_graph() -> CompiledStateGraph:
    """Builds, configures, and compiles the LangGraph StateGraph workflow for Project 3.

    Graph Architecture:
        START
          │
          ▼
    [supervisor_init]
          │
          ▼
       [router]
          │
          ▼  (Conditional Edge)
       ┌──┴──┐
       │     │
       ▼     ▼
     [rag] [mcp]
       │     │
       └──┬──┘
          ▼
    [supervisor_validation]
          │
          ▼
         END

    Returns:
        CompiledStateGraph instance ready for async execution.
    """
    logger.info("Building Project 3 LangGraph StateGraph...")
    builder = StateGraph(AgenticState)

    # 1. Add Nodes
    builder.add_node("supervisor_init", supervisor_init_node)
    builder.add_node("router", router_node)
    builder.add_node("rag", rag_node)
    builder.add_node("mcp", mcp_node)
    builder.add_node("supervisor_validation", supervisor_validation_node)

    # 2. Add Fixed Edges
    builder.add_edge(START, "supervisor_init")
    builder.add_edge("supervisor_init", "router")

    # 3. Add Conditional Edge from Router
    builder.add_conditional_edges(
        "router",
        route_edge,
        {
            "RAG": "rag",
            "MCP": "mcp",
        }
    )

    # 4. Add Join Edges to Supervisor Validation
    builder.add_edge("rag", "supervisor_validation")
    builder.add_edge("mcp", "supervisor_validation")
    builder.add_edge("supervisor_validation", END)

    # 5. Compile Graph
    graph = builder.compile()
    logger.info("Project 3 LangGraph StateGraph successfully compiled.")
    return graph
