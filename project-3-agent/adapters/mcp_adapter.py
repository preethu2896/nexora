"""MCP Adapter: Reusable non-invasive adapter for Project 2 (LangGraph + MCP Server)."""

import sys
import os
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from config import config
from utils.logger import get_logger
from utils.exceptions import MCPAdapterError

logger = get_logger("mcp_adapter", config.log_level)


class MCPAdapter:
    """Adapter for interacting with Project 2 (MCP Agent & Server) without code modification."""

    def __init__(self, project_2_dir: Optional[Path] = None):
        """Initializes MCP Adapter and registers Project 2 directory in sys.path.

        Args:
            project_2_dir: Path to project-2-mcp directory.
        """
        if project_2_dir is None:
            project_2_dir = config.project_2_dir

        self.project_2_dir = project_2_dir.resolve()
        self._ensure_sys_path()
        self.mcp_client = None
        self.graph = None
        self.raw_tools = []

    def _ensure_sys_path(self) -> None:
        """Injects project-2-mcp directory into Python sys.path."""
        p2_str = str(self.project_2_dir)
        if p2_str in sys.path:
            sys.path.remove(p2_str)
        sys.path.insert(0, p2_str)
        if "config" in sys.modules:
            del sys.modules["config"]
        if "utils" in sys.modules and hasattr(sys.modules["utils"], "pdf_processor"):
            del sys.modules["utils"]
        if "prompts" in sys.modules and hasattr(sys.modules["prompts"], "ROUTER_SYSTEM_PROMPT"):
            del sys.modules["prompts"]
        import config as loaded_config
        if hasattr(loaded_config, "config"):
            cfg = getattr(loaded_config, "config")
            if not hasattr(cfg, "project_1_dir"):
                setattr(cfg, "project_1_dir", self.project_2_dir.parent / "project-1-rag")
            if not hasattr(cfg, "project_2_dir"):
                setattr(cfg, "project_2_dir", self.project_2_dir)

    async def initialize(self) -> None:
        """Connects to MCP Server via stdio and initializes Project 2 LangGraph agent."""
        if self.graph is not None:
            return

        logger.info("Initializing MCPAdapter connection to Project 2...")
        try:
            self._ensure_sys_path()
            import importlib.util
            from mcp_service.client import MCPClient, convert_mcp_tools_to_langchain
            
            p2_graph_path = self.project_2_dir / "graph.py"
            spec = importlib.util.spec_from_file_location("p2_graph_module", p2_graph_path)
            p2_graph_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(p2_graph_module)
            create_agent_graph = p2_graph_module.create_agent_graph

            server_script = self.project_2_dir / "mcp_service" / "server.py"
            if not server_script.exists():
                raise FileNotFoundError(f"MCP Server script not found at {server_script}")

            self.mcp_client = MCPClient(server_script_path=server_script)
            self.raw_tools = await self.mcp_client.connect()
            langchain_tools = convert_mcp_tools_to_langchain(self.mcp_client, self.raw_tools)
            self.graph = create_agent_graph(self.mcp_client, langchain_tools)
            
            logger.info(f"MCPAdapter successfully connected. Discovered {len(self.raw_tools)} tool(s).")
        except Exception as e:
            logger.error(f"MCPAdapter initialization failed: {e}", exc_info=True)
            raise MCPAdapterError(f"Failed to initialize MCPAdapter connection: {e}") from e

    def health_check(self) -> Dict[str, Any]:
        """Checks readiness of Project 2 MCP server and client script.

        Returns:
            Status dictionary with health metrics.
        """
        logger.info("Executing MCPAdapter health check...")
        server_script = self.project_2_dir / "mcp_service" / "server.py"
        script_exists = server_script.exists()

        status = {
            "status": "healthy" if (script_exists and self.graph is not None) else ("degraded" if script_exists else "unhealthy"),
            "server_script": str(server_script),
            "script_exists": script_exists,
            "connected": self.graph is not None,
            "discovered_tools": len(self.raw_tools) if self.raw_tools else 0,
        }
        logger.info(f"MCPAdapter health status: {status['status']}")
        return status

    def available_tools(self) -> List[Dict[str, Any]]:
        """Returns metadata for all available MCP tools.

        Returns:
            List of tool metadata dicts.
        """
        tools_list = []
        for tool in self.raw_tools:
            tools_list.append({
                "name": getattr(tool, "name", str(tool)),
                "description": getattr(tool, "description", ""),
            })
        return tools_list

    async def execute(self, user_input: str, history: Optional[List[BaseMessage]] = None) -> Dict[str, Any]:
        """Executes a query through Project 2's LangGraph MCP agent workflow.

        Args:
            user_input: Active user query string.
            history: Optional conversation message history.

        Returns:
            Dictionary containing 'answer', 'messages', 'tool_calls', and 'tool_name'.
        """
        if self.graph is None:
            await self.initialize()

        logger.info(f"MCPAdapter executing query in Project 2: '{user_input}'")
        try:
            self._ensure_sys_path()
            messages = list(history) if history else []
            messages.append(HumanMessage(content=user_input))

            inputs = {"messages": messages}
            final_state = await self.graph.ainvoke(inputs)

            state_messages = final_state.get("messages", [])
            last_msg = state_messages[-1] if state_messages else None
            
            # Extract content text
            content = getattr(last_msg, "content", str(last_msg)) if last_msg else "No output generated."
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(str(item["text"]))
                    elif isinstance(item, str):
                        text_parts.append(item)
                    else:
                        text_parts.append(str(item))
                content = "\n".join(text_parts)

            # Check for tool call execution metadata in history
            executed_tools = []
            for msg in state_messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        executed_tools.append({
                            "name": tc.get("name"),
                            "args": tc.get("args"),
                        })

            result = {
                "answer": content,
                "messages": state_messages,
                "tool_calls": executed_tools,
                "tool_name": executed_tools[-1]["name"] if executed_tools else None,
            }
            logger.info("MCPAdapter query execution successfully completed.")
            return result
        except Exception as e:
            logger.error(f"MCPAdapter execution error: {e}", exc_info=True)
            raise MCPAdapterError(f"Failed to execute MCP query in Project 2: {e}") from e

    async def close(self) -> None:
        """Closes active MCP Client session."""
        if self.mcp_client:
            logger.info("MCPAdapter closing client connection...")
            try:
                await self.mcp_client.close()
            except Exception as e:
                logger.warning(f"Error closing MCPClient: {e}")
            self.mcp_client = None
            self.graph = None
            self.raw_tools = []
