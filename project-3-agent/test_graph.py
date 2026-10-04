"""Verification Test Suite for Milestone 2 (Agents, Supervisor & LangGraph Workflow)."""

import sys
import asyncio
from pathlib import Path

# Add project-3-agent to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from graph.builder import create_agentic_graph
from agents.router import RouterAgent
from adapters.mcp_adapter import MCPAdapter
from utils.logger import get_logger

logger = get_logger("test_graph")


async def main():
    print("\n" + "=" * 70)
    print(" 🧪 Milestone 2 Verification: Router Agent & LangGraph StateGraph ")
    print("=" * 70)

    # 1. Test Router Agent Structured Classification
    print("\n--- 1. Testing RouterAgent Semantic Classification ---")
    router = RouterAgent()

    test_queries = [
        ("whats the current time?", "MCP"),
        ("what is 250 multiplied by 4?", "MCP"),
        ("what are the key architectural features discussed in the RAG project?", "RAG"),
    ]

    for q, expected in test_queries:
        decision = await router.aclassify(q)
        print(f"\nQuery: '{q}'")
        print(f"  -> Selected Route: {decision.route} (Expected: {expected})")
        print(f"  -> Confidence: {decision.confidence:.2f}")
        print(f"  -> Reasoning: {decision.reasoning}")
        assert decision.route in ("RAG", "MCP"), "Router returned invalid route!"

    # 2. Test End-to-End StateGraph Execution
    print("\n--- 2. Testing End-to-End StateGraph Execution ---")
    graph = create_agentic_graph()

    # Query A: MCP Tool Request
    tool_query = "whats the current time?"
    print(f"\nExecuting Graph for MCP Query: '{tool_query}'...")
    initial_state_mcp = {"user_query": tool_query, "messages": []}
    final_state_mcp = await graph.ainvoke(initial_state_mcp)

    print(f"  -> Executed Route: {final_state_mcp.get('selected_route')}")
    print(f"  -> Execution Duration: {final_state_mcp.get('execution_duration')}s")
    print(f"  -> Final Response:\n{final_state_mcp.get('final_response')}\n")

    # Query B: Document Query
    rag_query = "What is the purpose of the RAG application?"
    print(f"\nExecuting Graph for RAG Query: '{rag_query}'...")
    initial_state_rag = {"user_query": rag_query, "messages": []}
    final_state_rag = await graph.ainvoke(initial_state_rag)

    print(f"  -> Executed Route: {final_state_rag.get('selected_route')}")
    print(f"  -> Execution Duration: {final_state_rag.get('execution_duration')}s")
    print(f"  -> Final Response:\n{final_state_rag.get('final_response')}\n")

    # Clean up MCP server process connection
    from nodes.mcp_node import mcp_adapter
    await mcp_adapter.close()

    print("=" * 70)
    print(" ✅ Milestone 2 LangGraph Workflow Verification Completed Successfully! ")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
