"""Main CLI Interactive Application for Project 2 (LangGraph + MCP Server)."""

import sys
import asyncio
from typing import List, Any
from langchain_core.messages import HumanMessage
from config import config
from utils.logger import get_logger
from mcp_service.client import MCPClient, convert_mcp_tools_to_langchain
from graph import create_agent_graph

logger = get_logger("app", config.log_level)


def extract_text_content(content: Any) -> str:
    """Helper to extract clean text from string or structured content items."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


async def run_cli_session():
    """Starts the interactive CLI session connecting to MCP Server and running LangGraph."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("\n" + "=" * 75)
    print("      🤖 Project 2: LangGraph + Custom MCP Server Agent (CLI) 🚀      ")
    print("=" * 75)
    print(" Connected Tools: calculator | current_time | generate_uuid | read_file")
    print(" Commands: Type your question, or 'exit' / 'quit' to close.")
    print("=" * 75 + "\n")

    mcp_client = MCPClient()

    try:
        raw_mcp_tools = await mcp_client.connect()
        langchain_tools = convert_mcp_tools_to_langchain(mcp_client, raw_mcp_tools)
        graph = create_agent_graph(mcp_client, langchain_tools)

        print("\n[+] MCP Server connected and LangGraph Agent ready!\n")

        conversation_messages = []

        while True:
            try:
                user_input = input("\n💬 User > ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit", "q"):
                    print("\n[*] Closing MCP session and exiting. Goodbye!\n")
                    break

                # Execute query through LangGraph StateMachine preserving conversation memory
                conversation_messages.append(HumanMessage(content=user_input))
                inputs = {"messages": conversation_messages}

                print("\n[*] Processing with LangGraph Agent...")
                final_state = await graph.ainvoke(inputs)

                # Persist state messages for multi-turn history
                conversation_messages = final_state["messages"]

                # Get final AI message response
                last_message = conversation_messages[-1]
                answer_text = extract_text_content(last_message.content)

                print("\n" + "-" * 60)
                print(f"🤖 Assistant > {answer_text}")
                print("-" * 60)

            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Exiting...")
                break
            except Exception as e:
                logger.error(f"Error handling query: {e}", exc_info=True)

    finally:
        await mcp_client.close()


def main():
    """Main application entrypoint."""
    try:
        asyncio.run(run_cli_session())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as err:
        logger.critical(f"Application crash: {err}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
