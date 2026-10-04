import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import StructuredTool
from utils.logger import get_logger
from config import config

logger = get_logger("mcp_client", config.log_level)


class MCPClient:
    """Client for managing communication with custom MCP server over stdio transport."""

    def __init__(self, server_script_path: Optional[Path] = None):
        """Initializes the MCP client with the server script path.

        Args:
            server_script_path: Path to mcp_service/server.py script.
        """
        if server_script_path is None:
            server_script_path = config.base_dir / "mcp_service" / "server.py"

        self.server_script_path = server_script_path.resolve()
        self.session: Optional[ClientSession] = None
        self._stdio_context = None
        self._read_stream = None
        self._write_stream = None

    async def connect(self) -> List[Dict[str, Any]]:
        """Spawns the MCP server process over stdio and initializes session.

        Returns:
            List of discovered raw tool dictionaries from MCP server.
        """
        logger.info(f"Connecting to MCP Server via stdio process ({self.server_script_path})...")

        # Pass active process environment to subprocess
        env_vars = dict(os.environ)

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(self.server_script_path)],
            env=env_vars,
        )

        self._stdio_context = stdio_client(server_params)
        self._read_stream, self._write_stream = await self._stdio_context.__aenter__()

        self.session = ClientSession(self._read_stream, self._write_stream)
        await self.session.__aenter__()

        logger.info("Initializing MCP Session...")
        await self.session.initialize()

        logger.info("Listing tools from MCP Server...")
        tools_response = await self.session.list_tools()
        logger.info(f"Discovered {len(tools_response.tools)} MCP tool(s) from server.")

        return tools_response.tools

    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Executes a tool call on the MCP server via the active session.

        Args:
            name: Tool name string.
            arguments: Keyword arguments dict.

        Returns:
            Text content response string from tool execution.
        """
        if not self.session:
            raise RuntimeError("MCP Session is not connected. Call connect() first.")

        # Handle nested kwargs wrapping if passed by LangChain StructuredTool
        if isinstance(arguments, dict) and "kwargs" in arguments and len(arguments) == 1 and isinstance(arguments["kwargs"], dict):
            arguments = arguments["kwargs"]

        logger.info(f"Sending CallTool request to MCP server: tool='{name}', args={arguments}")
        result = await self.session.call_tool(name, arguments=arguments)

        # Extract text content from call tool result
        output_chunks = []
        if result.content:
            for item in result.content:
                if hasattr(item, "text"):
                    output_chunks.append(item.text)
                elif hasattr(item, "data"):
                    output_chunks.append(str(item.data))
                else:
                    output_chunks.append(str(item))

        output_str = "\n".join(output_chunks) if output_chunks else "Tool executed successfully with no output."
        logger.info(f"Received result from MCP tool '{name}': {output_str}")
        return output_str

    async def close(self) -> None:
        """Closes the MCP session and stdio process connection."""
        if self.session:
            logger.info("Closing MCP Client session...")
            try:
                await self.session.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing MCP session: {e}")
            self.session = None

        if self._stdio_context:
            try:
                await self._stdio_context.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing stdio context: {e}")
            self._stdio_context = None

        logger.info("MCP Client connection closed.")


def convert_mcp_tools_to_langchain(mcp_client: MCPClient, mcp_tools: List[Any]) -> List[StructuredTool]:
    """Converts native MCP tool representations into LangChain StructuredTools.

    Args:
        mcp_client: Connected MCPClient instance.
        mcp_tools: List of tool metadata objects returned by session.list_tools().

    Returns:
        List of LangChain StructuredTool instances ready to bind to an LLM.
    """
    langchain_tools: List[StructuredTool] = []

    for tool in mcp_tools:
        tool_name = tool.name
        tool_desc = tool.description or f"MCP tool: {tool_name}"

        # Create dynamic wrapper functions for LangChain tool execution
        def make_sync_executor(t_name: str):
            import asyncio

            def sync_wrapper(**kwargs) -> str:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(
                        mcp_client.execute_tool(t_name, kwargs), loop
                    )
                    return future.result(timeout=30)
                else:
                    return asyncio.run(mcp_client.execute_tool(t_name, kwargs))

            return sync_wrapper

        def make_async_executor(t_name: str):
            async def async_wrapper(**kwargs) -> str:
                return await mcp_client.execute_tool(t_name, kwargs)

            return async_wrapper

        lc_tool = StructuredTool.from_function(
            func=make_sync_executor(tool_name),
            coroutine=make_async_executor(tool_name),
            name=tool_name,
            description=tool_desc,
        )

        langchain_tools.append(lc_tool)

    return langchain_tools
