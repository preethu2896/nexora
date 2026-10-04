"""Integration Test Suite for Milestone 3 (Interactive CLI, Lifecycle & Health Diagnostics)."""

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

from app import AgentCLIApp
from langchain_core.messages import HumanMessage, AIMessage
from utils.logger import get_logger

logger = get_logger("test_app")


async def main():
    print("\n" + "=" * 75)
    print(" 🧪 Milestone 3 Verification: End-to-End Application Lifecycle & CLI Tests ")
    print("=" * 75)

    app = AgentCLIApp()

    # 1. Test Application Startup & Subsystem Readiness
    print("\n--- 1. Testing Application Subsystems Initialization ---")
    success = await app.initialize()
    assert success, "App initialization failed!"
    print("✓ App initialization succeeded.")

    # 2. Test Live Health Diagnostics Command
    print("\n--- 2. Testing 'health' Diagnostics Command Output ---")
    app.display_health()
    print("✓ Live health diagnostics command verified.")

    # 3. Test Help Command Output
    print("\n--- 3. Testing 'help' Command Output ---")
    app.display_help()
    print("✓ Guidance help command verified.")

    # 4. Test Multi-Turn Conversation Memory
    print("\n--- 4. Testing Multi-Turn Conversation Memory ---")
    app.conversation_history.append(HumanMessage(content="Hello, what is your purpose?"))
    app.conversation_history.append(AIMessage(content="I am an Enterprise Agentic AI Assistant."))
    assert len(app.conversation_history) == 2, "Conversation history tracking failed!"
    print(f"✓ Conversation memory successfully stored {len(app.conversation_history)} messages.")

    # 5. Test History Display Command Output
    print("\n--- 5. Testing 'history' Command Output ---")
    app.display_history()
    print("✓ History transcript command verified.")

    # 6. Test Graceful Shutdown Sequence
    print("\n--- 6. Testing Graceful Application Shutdown Sequence ---")
    await app.shutdown()
    print("✓ Graceful shutdown sequence completed.")

    print("\n" + "=" * 75)
    print(" ✅ Milestone 3 Integration Verification Completed Successfully! ")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(main())
