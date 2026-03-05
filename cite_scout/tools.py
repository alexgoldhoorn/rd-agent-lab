"""Search tool wrappers for Tavily (web) and arXiv (papers)."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from cite_scout.config import Settings, get_settings
from cite_scout.prompts import SOURCE_REVIEW_PROMPT

logger = logging.getLogger(__name__)


# ── Tavily ────────────────────────────────────────────────────────────


def search_tavily(
    query: str,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Web search via Tavily. Returns [] if no API key is configured."""
    cfg = settings or get_settings()
    if not cfg.tavily_api_key:
        logger.info("TAVILY_API_KEY not set — skipping web search.")
        return []

    from langchain_community.tools.tavily_search import TavilySearchResults

    tool = TavilySearchResults(max_results=cfg.search_max_results)
    raw = tool.invoke(query)
    return [
        {
            "title": r.get("title", r.get("url", "Untitled")),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in raw
    ]


# ── arXiv ─────────────────────────────────────────────────────────────


def search_arxiv(
    query: str,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Paper search via arXiv. Returns [] if disabled in config."""
    cfg = settings or get_settings()
    if not cfg.arxiv_enabled:
        logger.info("arXiv search disabled — skipping.")
        return []

    from langchain_community.utilities import ArxivAPIWrapper

    wrapper = ArxivAPIWrapper(
        top_k_results=cfg.search_max_results,
        load_max_docs=cfg.search_max_results,
    )
    docs = wrapper.get_summaries_as_docs(query)
    return [
        {
            "title": d.metadata.get("Title", d.metadata.get("title", "Untitled")),
            "url": d.metadata.get("Entry ID", d.metadata.get("entry_id", "")),
            "snippet": d.page_content[:500],
        }
        for d in docs
    ]


# ── Combined search ──────────────────────────────────────────────────


def search_all(
    query: str,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Run all enabled search backends and merge results."""
    results: list[dict[str, Any]] = []
    for fn in (search_tavily, search_arxiv):
        try:
            results.extend(fn(query, settings=settings))
        except Exception:
            logger.warning("%s failed — continuing.", fn.__name__, exc_info=True)
    return results


# ── Source verification ──────────────────────────────────────────────


def verify_source(result: dict[str, Any], llm: BaseChatModel) -> dict[str, Any]:
    """Ask the LLM whether *result* is scientific/technical.

    Returns *result* augmented with ``is_scientific`` and ``review_reason``.
    """
    prompt = SOURCE_REVIEW_PROMPT.format(
        title=result.get("title", ""),
        url=result.get("url", ""),
        snippet=result.get("snippet", ""),
    )
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)

    try:
        verdict = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Could not parse review JSON — accepting by default.")
        verdict = {"is_scientific": True, "reason": "parse-error fallback"}

    result["is_scientific"] = bool(verdict.get("is_scientific", False))
    result["review_reason"] = verdict.get("reason", "")
    return result
