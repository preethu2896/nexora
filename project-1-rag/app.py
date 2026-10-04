"""Main CLI Entrypoint: Provides subcommands for document ingestion, single query, and interactive chat session."""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
from pathlib import Path
from config import config
from utils.logger import get_logger
from ingest import ingest_documents
from query import query_rag, print_query_result

logger = get_logger("app", config.log_level)


def run_interactive_chat(top_k: int = config.top_k) -> None:
    """Launches an interactive command-line chat session for querying PDFs.

    Args:
        top_k: Number of relevant document chunks to retrieve per query.
    """
    print("\n" + "═" * 70)
    print("      📚 Production PDF RAG Assistant — Interactive REPL Session 🤖      ")
    print("═" * 70)
    print(" Type your questions below. Commands: 'exit' or 'quit' to end session.")
    print("═" * 70 + "\n")

    while True:
        try:
            user_input = input("\n💬 Enter question: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("\n👋 Ending chat session. Goodbye!\n")
                break

            result = query_rag(question=user_input, top_k=top_k)
            print_query_result(result)

        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!\n")
            break
        except Exception as err:
            logger.error(f"Error executing query: {err}")


def main() -> None:
    """Main CLI entrypoint parser."""
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="Production PDF RAG (Retrieval-Augmented Generation) Application CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF documents into ChromaDB vector store.")
    ingest_parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to a PDF file or directory (default: data_dir in .env)",
    )
    ingest_parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset existing vector collection before indexing",
    )

    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Ask a single question against the PDF vector store.")
    query_parser.add_argument(
        "-q",
        "--question",
        type=str,
        required=True,
        help="Question text to query.",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=config.top_k,
        help="Number of document chunks to retrieve (default: top_k in .env)",
    )
    query_parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Hide source file citations",
    )

    # Chat subcommand
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session.")
    chat_parser.add_argument(
        "--top-k",
        type=int,
        default=config.top_k,
        help="Number of document chunks to retrieve per question",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        source_p = Path(args.source) if args.source else None
        count = ingest_documents(source_path=source_p, clear_existing=args.reset)
        if count == 0:
            sys.exit(1)

    elif args.command == "query":
        res = query_rag(question=args.question, top_k=args.top_k)
        print_query_result(res, show_sources=not args.no_sources)

    elif args.command == "chat":
        run_interactive_chat(top_k=args.top_k)

    else:
        print("\n" + "═" * 70)
        print("      📚 Production PDF RAG Assistant 🤖      ")
        print("═" * 70)
        print(" Choose an action:")
        print("   [1] Ingest a PDF document")
        print("   [2] Start interactive Q&A chat session")
        print("   [3] Exit")
        print("═" * 70)
        try:
            choice = input("\nSelect option (1/2/3): ").strip()
            if choice == "1":
                ingest_documents(clear_existing=False)
            elif choice == "2":
                run_interactive_chat()
            else:
                print("Goodbye!")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")


if __name__ == "__main__":
    main()
