"""Agent state flowing through the LangGraph nodes."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class Source(TypedDict):
    """A single research source with provenance metadata."""

    title: str
    url: str
    snippet: str
    is_scientific: bool


class AgentState(TypedDict):
    """Top-level state for the cite-scout graph."""

    # Conversation history (add_messages reducer).
    messages: Annotated[list[BaseMessage], add_messages]
    # Verified research findings.
    research_notes: list[str]
    # All discovered sources with review metadata.
    sources: list[Source]
    # Original user research query.
    query: str
