"""Tests for cite_scout.graph."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage, SystemMessage

from cite_scout.config import Settings
from cite_scout.graph import build_graph, input_node
from cite_scout.state import AgentState


# Shared minimal settings (no real API calls).
_CFG = Settings(
    llm_provider="openai",
    openai_model="gpt-4o",
    ollama_model="mistral",
    ollama_base_url="http://localhost:11434",
    tavily_api_key="",
    arxiv_enabled=False,
    search_max_results=2,
    max_search_hops=1,
    checkpoint_db="",
)


def test_input_node_returns_messages():
    state: AgentState = {
        "messages": [],
        "research_notes": [],
        "sources": [],
        "query": "test query",
    }
    result = input_node(state)
    assert len(result["messages"]) == 2
    assert isinstance(result["messages"][0], SystemMessage)
    assert isinstance(result["messages"][1], HumanMessage)
    assert "test query" in result["messages"][1].content


def test_build_graph_has_expected_nodes():
    with patch("cite_scout.graph.build_llm", return_value=MagicMock()):
        graph = build_graph(settings=_CFG)
    node_names = set(graph.nodes)
    assert {"input", "search", "review", "summary"} <= node_names


def test_build_graph_compiles():
    with patch("cite_scout.graph.build_llm", return_value=MagicMock()):
        graph = build_graph(settings=_CFG)
        compiled = graph.compile()
    assert compiled is not None
