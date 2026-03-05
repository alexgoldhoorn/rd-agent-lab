"""Search tool wrappers for the cite-scout research agent.

Provides thin wrappers around Tavily (web) and arXiv (papers) that return
results in a normalised format consumed by the graph nodes.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper

logger = logging.getLogger(__name__)

# Maximum results returned per search invocation.
_DEFAULT_MAX_RESULTS = 5


def get_tavily_tool(max_results: int = _DEFAULT_MAX_RESULTS) -> TavilySearchResults:
    """Return a configured Tavily search tool.

    Requires the TAVILY_API_KEY environment variable to be set.
    """
    return TavilySearchResults(max_results=max_results)


def get_arxiv_wrapper(
    top_k: int = _DEFAULT_MAX_RESULTS,
) -> ArxivAPIWrapper:
    """Return a configured arXiv API wrapper."""
    return ArxivAPIWrapper(top_k_results=top_k, load_max_docs=top_k)


def search_tavily(query: str, max_results: int = _DEFAULT_MAX_RESULTS) -> list[dict[str, Any]]:
    """Run a Tavily web search and return normalised result dicts.

    Each dict contains keys: title, url, snippet.
    """
    tool = get_tavily_tool(max_results=max_results)
    raw_results = tool.invoke(query)

    normalised: list[dict[str, Any]] = []
    for item in raw_results:
        normalised.append(
            {
                "title": item.get("title", item.get("url", "Untitled")),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            }
        )
    return normalised


def search_arxiv(query: str, max_results: int = _DEFAULT_MAX_RESULTS) -> list[dict[str, Any]]:
    """Run an arXiv search and return normalised result dicts.

    Each dict contains keys: title, url, snippet.
    """
    wrapper = get_arxiv_wrapper(top_k=max_results)
    docs = wrapper.get_summaries_as_docs(query)

    normalised: list[dict[str, Any]] = []
    for doc in docs:
        # arXiv docs expose metadata via the Document object.
        entry_id = doc.metadata.get("Entry ID", doc.metadata.get("entry_id", ""))
        title = doc.metadata.get("Title", doc.metadata.get("title", "Untitled"))
        normalised.append(
            {
                "title": title,
                "url": entry_id,
                "snippet": doc.page_content[:500],
            }
        )
    return normalised


def search_all(query: str, max_results: int = _DEFAULT_MAX_RESULTS) -> list[dict[str, Any]]:
    """Run both Tavily and arXiv searches and merge the results."""
    results: list[dict[str, Any]] = []

    # Tavily web search.
    try:
        results.extend(search_tavily(query, max_results=max_results))
    except Exception:
        logger.warning("Tavily search failed — continuing with arXiv only.", exc_info=True)

    # arXiv paper search.
    try:
        results.extend(search_arxiv(query, max_results=max_results))
    except Exception:
        logger.warning("arXiv search failed — continuing with Tavily results only.", exc_info=True)

    return results


def verify_source_dict(result: dict[str, Any], llm: Any) -> dict[str, Any]:
    """Use the LLM to decide if a single search result is scientific/technical.

    Returns the original *result* dict augmented with ``is_scientific`` (bool)
    and ``review_reason`` (str).
    """
    from prompts import SOURCE_REVIEW_PROMPT

    prompt_text = SOURCE_REVIEW_PROMPT.format(
        title=result.get("title", ""),
        url=result.get("url", ""),
        snippet=result.get("snippet", ""),
    )

    response = llm.invoke(prompt_text)
    content = response.content if hasattr(response, "content") else str(response)

    # Attempt to parse the JSON verdict from the LLM.
    try:
        verdict = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: conservatively accept the source.
        logger.warning("Could not parse review JSON — accepting source by default.")
        verdict = {"is_scientific": True, "reason": "parse-error fallback"}

    result["is_scientific"] = bool(verdict.get("is_scientific", False))
    result["review_reason"] = verdict.get("reason", "")
    return result
