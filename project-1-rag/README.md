# Production PDF RAG Application (Project 1)

A production-quality **Retrieval-Augmented Generation (RAG)** CLI application built with **Python 3.12+**, **LangChain**, **ChromaDB**, and **PyPDFLoader**. Answers questions grounded strictly in PDF documents with zero hallucination.

---

## 🌟 Key Features

- **Document Processing**: Automatic parsing of single or multiple PDF documents using `PyPDFLoader`.
- **Semantic Text Chunking**: Smart chunking using `RecursiveCharacterTextSplitter`.
- **Vector Database & Persistence**: Embedded vector storage with `ChromaDB` persisted locally.
- **Configurable LLM & Embeddings**: Support for OpenAI models or any OpenAI-compatible API (Ollama, LM Studio, vLLM).
- **Anti-Hallucination Guardrails**: Custom prompt instructions enforcing that if information is not found in the documents, the model responds with a clear disclaimer rather than inventing answers.
- **Source Citations**: Returns exact document filenames and page numbers for answer transparency.
- **Flexible CLI**: Commands for single-shot query execution, batch ingestion, and continuous interactive REPL chat mode.

---

## 🏗️ Architecture Breakdown

```
             [ User PDF Files in data/ ]
                         │
                         ▼
        ┌──────────────────────────────────┐
        │  PyPDFLoader (ingest.py / utils) │
        └──────────────────────────────────┘
                         │
                         ▼
   ┌────────────────────────────────────────────┐
   │ RecursiveCharacterTextSplitter (Chunking)  │
   │ (chunk_size=1000, chunk_overlap=200)       │
   └────────────────────────────────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────┐
        │   OpenAIEmbeddings (Configurable) │
        └──────────────────────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────┐
        │    Chroma VectorStore (Persisted) │
        │     Directory: ./chroma_db       │
        └──────────────────────────────────┘
                         │
        ═════════════════╪═════════════════ (Ingestion / Query Boundary)
                         │
                         ▼
                 [ User CLI Query ]
                         │
                         ▼
        ┌──────────────────────────────────┐
        │    Chroma VectorStore Retriever  │
        │          (Top-K Similarity)       │
        └──────────────────────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────┐
        │     Strict Anti-Hallucination    │
        │          Prompt Template         │
        └──────────────────────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────┐
        │   OpenAI-compatible LLM (query)  │
        └──────────────────────────────────┘
                         │
                         ▼
            [ Verified Answer + Sources ]
```

---

## 📁 Folder Structure

```
project-1-rag/
│
├── .env.example            # Environment variables template
├── .env                    # Local environment settings
├── README.md               # Application documentation
├── requirements.txt        # Python dependency specifications
├── config.py               # Centralized configuration & environment loader
├── ingest.py               # Document loading, chunking, embedding, vector DB creation
├── query.py                # Retrieval engine and question-answering logic
├── app.py                  # Main CLI entrypoint (ingest, query, chat subcommands)
│
├── data/                   # Directory to place PDF files to ingest
├── chroma_db/              # Persisted ChromaDB store directory
│
└── utils/
    ├── __init__.py
    ├── logger.py           # Custom colorized logging utility
    ├── pdf_processor.py    # PyPDFLoader document chunking helpers
    └── rag_chain.py        # LCEL RAG chain construction & anti-hallucination prompt
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.12 or higher
- An OpenAI API Key (or an OpenAI-compatible local server like Ollama or LM Studio)

### 2. Virtual Environment Setup
```bash
# Navigate into project directory
cd project-1-rag

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Copy `.env.example` to `.env` and set your API keys and configuration parameters:
```bash
cp .env.example .env
```

Open `.env` and update:
```env
OPENAI_API_KEY=your_actual_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL_NAME=text-embedding-3-small
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=4
```

---

## 💻 Usage Instructions

### 1. Add Documents
Place any PDF files you wish to question into the `data/` directory.

### 2. Ingest Documents
Process and embed all PDFs into ChromaDB:
```bash
python app.py ingest
```
To ingest a specific PDF file or directory:
```bash
python app.py ingest --source path/to/document.pdf
```
To reset and re-create the vector database collection:
```bash
python app.py ingest --reset
```

### 3. Single Question Query
Ask a direct question from the CLI:
```bash
python app.py query -q "What are the key findings discussed in the document?"
```

### 4. Interactive Chat REPL
Start an interactive conversation loop:
```bash
python app.py chat
```

---

## 🛡️ Anti-Hallucination Guarantee

If a query asks for information not contained within the ingested PDFs, the system responds with:
> *"The requested information is unavailable in the provided document(s)."*
