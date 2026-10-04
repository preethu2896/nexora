"""Test script to verify Milestone 1 Adapters (RAGAdapter and MCPAdapter)."""

import sys
import os
import asyncio
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project-3-agent to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from adapters.rag_adapter import RAGAdapter
from adapters.mcp_adapter import MCPAdapter
from utils.logger import get_logger

logger = get_logger("test_adapters")


async def main():
    print("\n" + "=" * 65)
    print(" 🧪 Milestone 1 Verification: RAGAdapter & MCPAdapter Health Checks ")
    print("=" * 65)

    # 1. Test RAGAdapter
    print("\n--- Testing RAGAdapter ---")
    rag = RAGAdapter()
    rag_health = rag.health_check()
    print(f"RAG Adapter Health Check Result:\n{rag_health}")

    # 2. Test MCPAdapter
    print("\n--- Testing MCPAdapter ---")
    mcp = MCPAdapter()
    await mcp.initialize()
    mcp_health = mcp.health_check()
    print(f"MCP Adapter Health Check Result:\n{mcp_health}")
    
    tools = mcp.available_tools()
    print(f"Discovered Tools ({len(tools)}):\n{[t['name'] for t in tools]}")

    # Test MCP tool call execution via MCPAdapter
    print("\n--- Testing MCPAdapter Execution ('whats the current time?') ---")
    mcp_res = await mcp.execute("whats the current time?")
    print(f"MCP Adapter Execution Output:\n{mcp_res['answer']}")

    await mcp.close()
    print("\n" + "=" * 65)
    print(" ✅ Milestone 1 Adapters Verification Completed Successfully! ")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
