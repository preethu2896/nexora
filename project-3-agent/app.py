"""Project 3: Enterprise Agentic AI Assistant - Interactive CLI Application."""

import sys
import os
import time
import signal
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project-3-agent directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure UTF-8 console output for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from config import config
from graph.builder import create_agentic_graph
from nodes.rag_node import rag_adapter
from nodes.mcp_node import mcp_adapter
from utils.logger import get_logger

logger = get_logger("app_cli", config.log_level)


class TerminalColors:
    """ANSI color codes for terminal formatting."""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


class AgentCLIApp:
    """Production Interactive CLI REPL for the Unified Agentic AI Assistant."""

    def __init__(self):
        """Initializes CLI Application state."""
        self.graph = None
        self.conversation_history: List[BaseMessage] = []
        self.is_running = True

    def print_banner(self) -> None:
        """Prints a professional startup banner."""
        banner = f"""
{TerminalColors.CYAN}{TerminalColors.BOLD}══════════════════════════════════════════════════════════════════════════════
 🤖 PROJECT 3: UNIFIED ENTERPRISE AGENTIC AI ASSISTANT 🚀
 Orchestrating LangGraph | RAG Subsystem | MCP Subsystem | Google Gemini
══════════════════════════════════════════════════════════════════════════════{TerminalColors.RESET}
"""
        print(banner)

    async def initialize(self) -> bool:
        """Executes application startup initialization and system readiness checks.

        Returns:
            Boolean indicating if initialization succeeded.
        """
        print(f"{TerminalColors.BLUE}[*] Initializing System Subsystems...{TerminalColors.RESET}")
        logger.info("Starting Application Initialization...")

        # 1. Validate Configuration
        try:
            config.validate_api_key()
            print(f"  {TerminalColors.GREEN}✓ Environment & API Key validated.{TerminalColors.RESET}")
        except Exception as e:
            print(f"  {TerminalColors.RED}✗ Configuration Error: {e}{TerminalColors.RESET}")
            logger.error(f"Configuration validation failed: {e}")
            return False

        # 2. Initialize RAG Adapter Health Check
        print(f"  {TerminalColors.BLUE}• Connecting to RAG Subsystem (Project 1)...{TerminalColors.RESET}")
        rag_health = rag_adapter.health_check()
        if rag_health.get("status") in ("healthy", "degraded"):
            db_state = "Persistent ChromaDB Loaded" if rag_health.get("chroma_db_exists") else "DB Pending Document Ingestion"
            print(f"  {TerminalColors.GREEN}✓ RAG Adapter Ready ({db_state}).{TerminalColors.RESET}")
        else:
            print(f"  {TerminalColors.YELLOW}⚠️ RAG Adapter Status: {rag_health.get('status')} ({rag_health.get('error')}){TerminalColors.RESET}")

        # 3. Initialize MCP Adapter Connection
        print(f"  {TerminalColors.BLUE}• Connecting to MCP Subsystem (Project 2)...{TerminalColors.RESET}")
        try:
            await mcp_adapter.initialize()
            mcp_health = mcp_adapter.health_check()
            tools = mcp_adapter.available_tools()
            tool_names = ", ".join([t["name"] for t in tools]) if tools else "None"
            print(f"  {TerminalColors.GREEN}✓ MCP Adapter Ready (Discovered Tools: {tool_names}).{TerminalColors.RESET}")
        except Exception as e:
            print(f"  {TerminalColors.YELLOW}⚠️ MCP Adapter Warning: {e}{TerminalColors.RESET}")
            logger.warning(f"MCP Adapter initialization error: {e}")

        # 4. Compile LangGraph StateGraph
        print(f"  {TerminalColors.BLUE}• Compiling LangGraph Orchestration StateGraph...{TerminalColors.RESET}")
        try:
            self.graph = create_agentic_graph()
            print(f"  {TerminalColors.GREEN}✓ LangGraph Compiled Successfully.{TerminalColors.RESET}")
        except Exception as e:
            print(f"  {TerminalColors.RED}✗ LangGraph Compilation Error: {e}{TerminalColors.RESET}")
            logger.error(f"LangGraph compilation failed: {e}")
            return False

        print(f"\n{TerminalColors.GREEN}{TerminalColors.BOLD}[+] System Initialization Complete. Ready to accept requests!{TerminalColors.RESET}")
        print(f"{TerminalColors.DIM}Type your query, or 'help' for available commands, 'exit' to quit.{TerminalColors.RESET}\n" + "─" * 78)
        return True

    def display_help(self) -> None:
        """Displays available interactive commands."""
        help_text = f"""
{TerminalColors.BOLD}Available System Commands:{TerminalColors.RESET}
  {TerminalColors.CYAN}help{TerminalColors.RESET}       - Display this command guidance message.
  {TerminalColors.CYAN}health{TerminalColors.RESET}     - View live health metrics for RAG, MCP, Vector DB & System.
  {TerminalColors.CYAN}history{TerminalColors.RESET}    - View multi-turn conversation transcript history.
  {TerminalColors.CYAN}clear{TerminalColors.RESET}      - Clear the terminal screen and reset conversation state.
  {TerminalColors.CYAN}exit / quit{TerminalColors.RESET}- Gracefully shut down the application session.

{TerminalColors.BOLD}How the Agent Works:{TerminalColors.RESET}
  • Ask about technical documents, PDFs, or specs → Routed to {TerminalColors.BOLD}RAG Subsystem{TerminalColors.RESET}.
  • Ask for calculations, date/time, UUIDs, or reading files → Routed to {TerminalColors.BOLD}MCP Subsystem{TerminalColors.RESET}.
"""
        print(help_text)

    def display_health(self) -> None:
        """Displays live system health diagnostics."""
        print(f"\n{TerminalColors.BOLD}📊 LIVE SYSTEM HEALTH DIAGNOSTICS:{TerminalColors.RESET}")
        
        # RAG Diagnostics
        rag_h = rag_adapter.health_check()
        print(f"  • {TerminalColors.BOLD}RAG Status{TerminalColors.RESET}: {rag_h.get('status').upper()}")
        print(f"    - Vector DB Path: {rag_h.get('chroma_db_dir')}")
        print(f"    - Database Present: {rag_h.get('chroma_db_exists')}")

        # MCP Diagnostics
        mcp_h = mcp_adapter.health_check()
        tools = mcp_adapter.available_tools()
        print(f"  • {TerminalColors.BOLD}MCP Status{TerminalColors.RESET}: {mcp_h.get('status').upper()}")
        print(f"    - MCP Server Script: {mcp_h.get('server_script')}")
        print(f"    - Available Tools ({len(tools)}): {', '.join([t['name'] for t in tools])}")

        # General Metrics
        print(f"  • {TerminalColors.BOLD}LLM Model{TerminalColors.RESET}: {config.llm_model_name}")
        print(f"  • {TerminalColors.BOLD}Conversation History{TerminalColors.RESET}: {len(self.conversation_history)} message(s)")
        print(f"  • {TerminalColors.BOLD}Overall Status{TerminalColors.RESET}: {TerminalColors.GREEN}HEALTHY{TerminalColors.RESET}\n")

    def display_history(self) -> None:
        """Displays multi-turn conversation history."""
        if not self.conversation_history:
            print(f"\n{TerminalColors.YELLOW}No conversation history available in current session.{TerminalColors.RESET}\n")
            return

        print(f"\n{TerminalColors.BOLD}📜 MULTI-TURN CONVERSATION TRANSCRIPT ({len(self.conversation_history)} messages):{TerminalColors.RESET}")
        for idx, msg in enumerate(self.conversation_history, 1):
            role = "💬 User" if isinstance(msg, HumanMessage) else "🤖 Assistant"
            content = getattr(msg, "content", str(msg))
            print(f"\n[{idx}] {TerminalColors.BOLD}{role}{TerminalColors.RESET}:")
            print(f"    {content}")
        print()

    async def process_query(self, user_query: str) -> None:
        """Executes query through the LangGraph Agent workflow.

        Args:
            user_query: Input request string.
        """
        logger.info(f"CLI Input received: '{user_query}'")
        print(f"\n{TerminalColors.BLUE}[*] Routing & Processing Request with Agentic Graph...{TerminalColors.RESET}")

        initial_state = {
            "user_query": user_query,
            "messages": list(self.conversation_history),
        }

        try:
            final_state = await self.graph.ainvoke(initial_state)

            route = final_state.get("selected_route", "RAG")
            confidence = final_state.get("route_confidence", 1.0)
            reasoning = final_state.get("route_reasoning", "Semantic classification")
            response_text = final_state.get("final_response", "No output generated.")
            duration = final_state.get("execution_duration", 0.0)

            # Update conversation history with Human and AI Messages
            self.conversation_history.append(HumanMessage(content=user_query))
            self.conversation_history.append(AIMessage(content=response_text))

            # Display execution summary
            print(f"\n{TerminalColors.DIM}┌────────────────────────────────────────────────────────────┐{TerminalColors.RESET}")
            print(f"{TerminalColors.DIM}│{TerminalColors.RESET} {TerminalColors.BOLD}Selected Route{TerminalColors.RESET}: {TerminalColors.CYAN}{route}{TerminalColors.RESET} (Confidence: {confidence:.2f})")
            print(f"{TerminalColors.DIM}│{TerminalColors.RESET} {TerminalColors.BOLD}Latency{TerminalColors.RESET}: {duration:.2f}s")
            print(f"{TerminalColors.DIM}└────────────────────────────────────────────────────────────┘{TerminalColors.RESET}")

            # Display formatted assistant answer
            print(f"\n{TerminalColors.BOLD}🤖 Assistant:{TerminalColors.RESET}\n{response_text}\n")
            print("─" * 78)

        except Exception as e:
            logger.error(f"Error processing CLI query: {e}", exc_info=True)
            print(f"\n{TerminalColors.RED}⚠️ System Exception: Failed to process query.{TerminalColors.RESET}")
            print(f"{TerminalColors.DIM}Reason: {str(e)}{TerminalColors.RESET}\n" + "─" * 78)

    async def run(self) -> None:
        """Starts the interactive REPL loop."""
        self.print_banner()
        success = await self.initialize()
        if not success:
            print(f"{TerminalColors.RED}Initialization failed. Exiting.{TerminalColors.RESET}")
            await self.shutdown()
            return

        while self.is_running:
            try:
                user_input = input(f"{TerminalColors.BOLD}💬 User > {TerminalColors.RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n")
                logger.info("Shutdown signal received via keyboard interrupt.")
                break

            if not user_input:
                continue

            cmd_lower = user_input.lower()
            if cmd_lower in ("exit", "quit"):
                print(f"\n{TerminalColors.YELLOW}[*] Exit command received.{TerminalColors.RESET}")
                break
            elif cmd_lower == "help":
                self.display_help()
                continue
            elif cmd_lower == "health":
                self.display_health()
                continue
            elif cmd_lower == "history":
                self.display_history()
                continue
            elif cmd_lower == "clear":
                os.system("cls" if os.name == "nt" else "clear")
                self.print_banner()
                self.conversation_history.clear()
                print(f"{TerminalColors.GREEN}Terminal screen cleared and conversation history reset.{TerminalColors.RESET}\n")
                continue

            await self.process_query(user_input)

        await self.shutdown()

    async def shutdown(self) -> None:
        """Executes graceful application shutdown."""
        print(f"{TerminalColors.YELLOW}[*] Closing MCP Client session & releasing resources...{TerminalColors.RESET}")
        logger.info("Shutting down Application...")
        try:
            await mcp_adapter.close()
        except Exception as e:
            logger.warning(f"Error closing MCP Adapter during shutdown: {e}")
            
        print(f"{TerminalColors.GREEN}[*] Graceful shutdown complete. Goodbye!{TerminalColors.RESET}\n")
        self.is_running = False


def main():
    """Main CLI Entrypoint."""
    app = AgentCLIApp()
    
    # Handle CTRL+C signals
    def signal_handler(sig, frame):
        print("\n")
        logger.info("Signal interrupt caught.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\nGracefully exiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
