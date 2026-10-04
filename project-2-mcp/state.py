"""LangGraph State definition for Project 2."""

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """The State schema for the LangGraph agent state machine.

    Attributes:
        messages: List of conversation messages updated using `add_messages` reducer.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
