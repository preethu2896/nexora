"""Typed state definition for LangGraph workflow execution in Project 3."""

from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage


class AgenticState(TypedDict):
    """Strongly-typed LangGraph State schema for Project 3 Assistant."""

    messages: List[BaseMessage]                         # Conversation message history
    user_query: str                                     # Active user request string
    
    # Router Agent Outputs
    selected_route: Optional[str]                       # 'RAG' or 'MCP'
    route_confidence: Optional[float]                   # Float score (0.0 to 1.0)
    route_reasoning: Optional[str]                      # Semantic decision rationale
    
    # RAG Subsystem Outputs
    retrieved_context: Optional[List[Dict[str, Any]]]    # Retrieved text chunks
    citations: Optional[List[Dict[str, Any]]]            # Page & document citations
    
    # MCP Subsystem Outputs
    tool_name: Optional[str]                            # Name of executed tool
    tool_calls: Optional[List[Dict[str, Any]]]           # Executed tool call details
    
    # Supervisor Outputs & Execution Metrics
    final_response: Optional[str]                       # Human-readable formatted output
    execution_duration: Optional[float]                 # Execution latency in seconds
    metadata: Dict[str, Any]                            # Detailed metrics trace
    error: Optional[str]                                # Exception error trace string


# Alias for backward compatibility with Project 2 graph state import
AgentState = AgenticState
