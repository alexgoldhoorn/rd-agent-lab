"""Agent state definition for the cite-scout research agent."""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class Source(TypedDict):
    """A single research source with provenance metadata."""

    # Human-readable title of the source.
    title: str
    # URL or identifier (e.g. arXiv ID) pointing to the source.
    url: str
    # Short description of what the source contributes.
    snippet: str
    # Whether the source passed the scientific/technical verification.
    is_scientific: bool


class AgentState(TypedDict):
    """Top-level state that flows through every node in the graph.

    Attributes:
        messages: Conversation history managed by LangGraph's message reducer.
        research_notes: Accumulated findings that passed scientific verification.
        sources: Registry of all discovered sources with metadata.
        query: The original user research query.
    """

    # Conversation history — uses the built-in add_messages reducer so each
    # node can simply return new messages instead of the full list.
    messages: Annotated[list[BaseMessage], add_messages]
    # Verified research findings, one string per accepted source.
    research_notes: list[str]
    # Full source metadata collected during the search phase.
    sources: list[Source]
    # The research question the user submitted.
    query: str
