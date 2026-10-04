# Production LangGraph + Custom MCP Server Application (Project 2)

A production-ready **LangGraph** application in Python 3.12+ communicating with a custom **MCP (Model Context Protocol)** server over stdio transport. The application uses a state graph engine to dynamically evaluate user intent and execute MCP tools.

---

## 🌟 Key Features

- **Model Context Protocol (MCP)**: Custom MCP server exposing 4 tools via the official `mcp` Python SDK (`FastMCP`).
- **LangGraph StateMachine**: Graph workflow (`START` -> `agent` -> `should_continue` -> `tools` -> `agent` -> `END`) managing message states and tool executions.
- **Custom MCP Tools**:
  1. `calculator`: Safe arithmetic calculation evaluator.
  2. `current_time`: ISO & human-readable timestamp string for any timezone.
  3. `generate_uuid`: Standard UUIDv4 unique string generator.
  4. `read_file`: Local file text content reader.
- **Dynamic Tool Conversion**: Automatic adapter converting MCP tool schemas into LangChain `StructuredTool` objects bound to LLMs.
- **CLI Interactive Chat**: Full REPL session displaying graph node transitions, tool invocations, and answers.

---

## 🏗️ Architecture Breakdown

```
                      [ User Input via CLI (app.py) ]
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │ LangGraph State Machine      │
                      │ (Agent Node in graph.py)     │
                      └──────────────────────────────┘
                                     │
                  Does LLM decide to call an MCP Tool?
                                ╱         ╲
                             YES           NO
                             ╱               ╲
                            ▼                 ▼
          ┌──────────────────────────┐   ┌──────────────────────────┐
          │ LangGraph Tool Node      │   │ Return Final Answer      │
          │ (Executes via MCPClient) │   │ to User (END Node)       │
          └──────────────────────────┘   └──────────────────────────┘
                        │
                        ▼
          ┌──────────────────────────┐
          │ MCP Client (mcp/client.py)│
          └──────────────────────────┘
                        │
                  (STDIO / MCP Protocol)
                        │
                        ▼
          ┌──────────────────────────┐
          │ MCP Server (mcp/server.py)│
          └──────────────────────────┘
                        │
             Executes Custom Tool in mcp/tools.py:
             - Calculator
             - Current Time
             - UUID Generator
             - File Reader
                        │
                        ▼
            [ Result returned to Agent State ]
```

---

## 📁 Project Structure

```
project-2-mcp/
│
├── .env.example            # Environment variables template
├── .env                    # Local environment configuration
├── README.md               # Architecture documentation & guide
├── requirements.txt        # Dependencies (LangGraph, MCP SDK, LangChain, etc.)
├── config.py               # Settings loader
├── state.py                # LangGraph AgentState definition
├── prompts.py              # System prompt definitions
├── graph.py                # LangGraph state machine & node logic
├── app.py                  # Main CLI entrypoint
│
├── mcp/
│   ├── __init__.py
│   ├── server.py           # Custom MCP Server (FastMCP)
│   ├── client.py           # MCP Client adapter & LangChain tool binder
│   └── tools.py            # Implementation of Calculator, Time, UUID, File Reader
│
└── utils/
    ├── __init__.py
    └── logger.py           # Colorized logging handler
```

---

## 🚀 Setup & Execution

### 1. Virtual Environment Setup
```bash
# Navigate into project directory
cd project-2-mcp

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
OPENAI_API_KEY=your_actual_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o-mini
LOG_LEVEL=INFO
```

### 4. Run the Application
Launch the interactive CLI chat session:
```bash
python app.py
```

---

## 💬 Example CLI Interactions

```text
💬 User > Calculate 25*40
⚙️  Processing with LangGraph Agent...
🤖 Assistant > The result of 25 * 40 is 1000.

💬 User > What time is it now?
⚙️  Processing with LangGraph Agent...
🤖 Assistant > The current time is 2026-07-31 14:35:00 UTC (Day: Friday).

💬 User > Generate a UUID for me
⚙️  Processing with LangGraph Agent...
🤖 Assistant > Here is your generated UUIDv4: 7d9a2b8e-4c1f-4b82-9e3d-1a8f9c0b2d3e.

💬 User > Read file requirements.txt
⚙️  Processing with LangGraph Agent...
🤖 Assistant > Here is the content of requirements.txt:
mcp>=1.0.0
langgraph>=0.1.0
...
```
