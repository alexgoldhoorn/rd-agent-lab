"""Tests for cite_scout.state."""

from cite_scout.state import AgentState, Source


def test_source_creation():
    src = Source(title="T", url="http://x", snippet="S", is_scientific=True)
    assert src["title"] == "T"
    assert src["is_scientific"] is True


def test_source_defaults_false():
    src = Source(title="T", url="", snippet="", is_scientific=False)
    assert src["is_scientific"] is False


def test_agent_state_type_hints():
    """AgentState is a TypedDict — verify expected keys exist."""
    keys = set(AgentState.__annotations__)
    assert keys == {"messages", "research_notes", "sources", "query"}
