"""Tests for cite_scout.tools."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from cite_scout.config import Settings
from cite_scout.tools import search_all, search_arxiv, search_tavily, verify_source


# ── Fixtures ──────────────────────────────────────────────────────────

# A settings object with search backends disabled.
_NO_KEYS = Settings(
    llm_provider="openai",
    openai_model="gpt-4o",
    ollama_model="mistral",
    ollama_base_url="http://localhost:11434",
    tavily_api_key="",
    arxiv_enabled=False,
    search_max_results=3,
    max_search_hops=1,
    checkpoint_db="",
)


# ── search_tavily ────────────────────────────────────────────────────


def test_tavily_skipped_without_key():
    assert search_tavily("test", settings=_NO_KEYS) == []


def test_tavily_returns_normalised(monkeypatch: pytest.MonkeyPatch):
    cfg = Settings(
        llm_provider="openai",
        openai_model="gpt-4o",
        ollama_model="mistral",
        ollama_base_url="http://localhost:11434",
        tavily_api_key="fake-key",
        arxiv_enabled=False,
        search_max_results=2,
        max_search_hops=1,
        checkpoint_db="",
    )
    fake_results = [
        {"title": "A", "url": "http://a", "content": "snippet-a"},
        {"title": "B", "url": "http://b", "content": "snippet-b"},
    ]
    with patch("langchain_community.tools.tavily_search.TavilySearchResults") as mock_cls:
        mock_cls.return_value.invoke.return_value = fake_results
        results = search_tavily("q", settings=cfg)

    assert len(results) == 2
    assert results[0]["title"] == "A"
    assert results[1]["snippet"] == "snippet-b"


# ── search_arxiv ─────────────────────────────────────────────────────


def test_arxiv_skipped_when_disabled():
    assert search_arxiv("test", settings=_NO_KEYS) == []


def test_arxiv_returns_normalised():
    cfg = Settings(
        llm_provider="openai",
        openai_model="gpt-4o",
        ollama_model="mistral",
        ollama_base_url="http://localhost:11434",
        tavily_api_key="",
        arxiv_enabled=True,
        search_max_results=2,
        max_search_hops=1,
        checkpoint_db="",
    )
    fake_doc = MagicMock()
    fake_doc.metadata = {"Title": "Paper X", "Entry ID": "http://arxiv/1"}
    fake_doc.page_content = "abstract text"

    with patch("langchain_community.utilities.ArxivAPIWrapper") as mock_cls:
        mock_cls.return_value.get_summaries_as_docs.return_value = [fake_doc]
        results = search_arxiv("q", settings=cfg)

    assert len(results) == 1
    assert results[0]["title"] == "Paper X"


# ── search_all ───────────────────────────────────────────────────────


def test_search_all_merges():
    with (
        patch("cite_scout.tools.search_tavily", return_value=[{"title": "W", "url": "", "snippet": ""}]),
        patch("cite_scout.tools.search_arxiv", return_value=[{"title": "A", "url": "", "snippet": ""}]),
    ):
        results = search_all("q", settings=_NO_KEYS)
    assert len(results) == 2


def test_search_all_tolerates_failure():
    mock_tavily = MagicMock(side_effect=RuntimeError("boom"), __name__="search_tavily")
    mock_arxiv = MagicMock(return_value=[{"title": "A", "url": "", "snippet": ""}], __name__="search_arxiv")
    with (
        patch("cite_scout.tools.search_tavily", mock_tavily),
        patch("cite_scout.tools.search_arxiv", mock_arxiv),
    ):
        results = search_all("q", settings=_NO_KEYS)
    assert len(results) == 1


# ── verify_source ────────────────────────────────────────────────────


def test_verify_source_scientific():
    llm = MagicMock()
    llm.invoke.return_value.content = '{"is_scientific": true, "reason": "peer-reviewed"}'
    result = verify_source({"title": "T", "url": "U", "snippet": "S"}, llm)
    assert result["is_scientific"] is True


def test_verify_source_not_scientific():
    llm = MagicMock()
    llm.invoke.return_value.content = '{"is_scientific": false, "reason": "marketing"}'
    result = verify_source({"title": "T", "url": "U", "snippet": "S"}, llm)
    assert result["is_scientific"] is False


def test_verify_source_json_parse_error_fallback():
    llm = MagicMock()
    llm.invoke.return_value.content = "not json"
    result = verify_source({"title": "T", "url": "U", "snippet": "S"}, llm)
    # Falls back to accepting.
    assert result["is_scientific"] is True
