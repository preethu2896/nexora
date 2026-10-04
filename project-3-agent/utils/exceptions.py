"""Custom Exception hierarchy for Project 3 Agentic Assistant."""


class Project3BaseException(Exception):
    """Base exception for all Project 3 errors."""
    pass


class AdapterError(Project3BaseException):
    """Base exception for Subsystem Adapters."""
    pass


class RAGAdapterError(AdapterError):
    """Raised when Project 1 RAG adapter encounters an error."""
    pass


class MCPAdapterError(AdapterError):
    """Raised when Project 2 MCP adapter encounters an error."""
    pass


class RouterError(Project3BaseException):
    """Raised when LLM intent router classification fails."""
    pass


class SupervisorError(Project3BaseException):
    """Raised when Supervisor pre/post-execution validation fails."""
    pass


class ConfigurationError(Project3BaseException):
    """Raised when invalid settings or missing API keys are encountered."""
    pass
