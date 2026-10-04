"""Custom MCP (Model Context Protocol) Server exposing tools using FastMCP."""

import sys
from pathlib import Path

# Add project root to sys.path for standalone script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mcp.server.fastmcp import FastMCP
from pydantic import Field
from mcp_service.tools import (
    calculator as calc_fn,
    current_time as time_fn,
    generate_uuid as uuid_fn,
    read_file as read_fn,
)

# Initialize FastMCP Server
mcp_server = FastMCP(name="Custom-Tools-MCP-Server")


@mcp_server.tool(
    name="calculator",
    description="Evaluates a mathematical arithmetic expression (e.g. '25 * 40', '(100 + 50) / 3', '2**8').",
)
async def tool_calculator(
    expression: str = Field(..., description="Mathematical arithmetic expression string to calculate.")
) -> str:
    return calc_fn(expression)


@mcp_server.tool(
    name="current_time",
    description="Returns the current date, time, day of week, and timezone information.",
)
async def tool_current_time(
    timezone_name: str = Field(default="UTC", description="Timezone name string (e.g. 'UTC', 'America/New_York').")
) -> str:
    return time_fn(timezone_name)


@mcp_server.tool(
    name="generate_uuid",
    description="Generates a new random UUIDv4 unique string identifier.",
)
async def tool_generate_uuid() -> str:
    return uuid_fn()


@mcp_server.tool(
    name="read_file",
    description="Reads and returns the text content of a local file path.",
)
async def tool_read_file(
    file_path: str = Field(..., description="Target file path to read from disk.")
) -> str:
    return read_fn(file_path)


def start_server() -> None:
    """Runs the MCP server over stdio transport."""
    mcp_server.run(transport="stdio")


if __name__ == "__main__":
    start_server()
