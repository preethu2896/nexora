"""Implementation of custom MCP tools: Calculator, Current Time, UUID Generator, File Reader."""

import ast
import operator
import uuid
import datetime
import zoneinfo
from pathlib import Path
from typing import Union
from utils.logger import get_logger
from config import config

logger = get_logger("mcp_tools", config.log_level)

# Supported AST operators for safe calculator tool
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_node(node: ast.AST) -> Union[int, float]:
    """Recursively evaluates abstract syntax tree nodes for safe arithmetic calculations.

    Args:
        node: Parsed AST node.

    Returns:
        Calculated numeric result.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp):
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        op_type = type(node.op)

        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported operator in expression: {op_type.__name__}")

        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("Division or modulo by zero is not allowed.")

        return ALLOWED_OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval_node(node.operand)
        op_type = type(node.op)

        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")

        return ALLOWED_OPERATORS[op_type](operand)

    raise ValueError(f"Invalid arithmetic expression element: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluates an arithmetic mathematical expression safely.

    Args:
        expression: Mathematical expression string (e.g. '25 * 40', '(100 + 50) / 3', '2**8').

    Returns:
        Result string or error message.
    """
    logger.info(f"Executing Tool 'calculator' with expression: '{expression}'")
    try:
        clean_expr = expression.strip()
        parsed = ast.parse(clean_expr, mode="eval")
        result = _safe_eval_node(parsed.body)
        return str(result)
    except Exception as e:
        logger.error(f"Calculator tool error evaluating '{expression}': {e}")
        return f"Error evaluating calculation: {str(e)}"


def current_time(timezone_name: str = "Asia/Kolkata") -> str:
    """Returns the current date, time, and timezone information.

    Args:
        timezone_name: Timezone string (defaults to 'Asia/Kolkata' or local system timezone).

    Returns:
        Formatted timestamp string.
    """
    logger.info(f"Executing Tool 'current_time' for timezone: '{timezone_name}'")
    try:
        local_now = datetime.datetime.now().astimezone()
        utc_now = datetime.datetime.now(datetime.timezone.utc)

        formatted_local = local_now.strftime("%I:%M:%S %p %Z (Day: %A, Date: %Y-%m-%d)")
        formatted_utc = utc_now.strftime("%I:%M:%S %p UTC")

        return (
            f"Current Local Time: {formatted_local}\n"
            f"UTC Equivalent: {formatted_utc}"
        )
    except Exception as e:
        logger.error(f"Current time tool error: {e}")
        return f"Error getting current time: {str(e)}"


def generate_uuid() -> str:
    """Generates a random UUIDv4 string.

    Returns:
        Standard UUIDv4 string (e.g., '123e4567-e89b-12d3-a456-426614174000').
    """
    logger.info("Executing Tool 'generate_uuid'")
    new_uuid = str(uuid.uuid4())
    return new_uuid


def read_file(file_path: str) -> str:
    """Reads the text content of a file (supports text files and PDF documents).

    Args:
        file_path: Path to the target file to read.

    Returns:
        Content string or error message.
    """
    logger.info(f"Executing Tool 'read_file' for path: '{file_path}'")
    try:
        target_path = Path(file_path).resolve()

        if not target_path.exists():
            return f"Error: File '{file_path}' does not exist."

        if not target_path.is_file():
            return f"Error: Target path '{file_path}' is not a regular file."

        # Handle PDF files using pypdf
        if target_path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(target_path)
                pages_text = []
                for idx, page in enumerate(reader.pages):
                    txt = page.extract_text()
                    if txt:
                        pages_text.append(f"--- Page {idx+1} ---\n{txt}")
                if pages_text:
                    return "\n\n".join(pages_text)
                return f"PDF file '{file_path}' was loaded but contains no extractable text."
            except Exception as pdf_err:
                logger.warning(f"pypdf extraction error for '{file_path}': {pdf_err}")
                return f"Error reading PDF file: {pdf_err}"

        # Standard text file reading
        content = target_path.read_text(encoding="utf-8", errors="replace")
        return content
    except Exception as e:
        logger.error(f"Read file tool error for '{file_path}': {e}")
        return f"Error reading file '{file_path}': {str(e)}"
