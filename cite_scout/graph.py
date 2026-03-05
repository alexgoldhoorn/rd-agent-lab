"""LangGraph nodes and graph assembly for cite-scout."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from cite_scout.config import Settings, get_settings
from cite_scout.llm import build_llm
from cite_scout.prompts import RESEARCH_SPECIALIST_PROMPT
from cite_scout.state import AgentState, Source
from cite_scout.tools import search_all, verify_source

logger = logging.getLogger(__name__)


# ── Node implementations ─────────────────────────────────────────────


def input_node(state: AgentState) -> dict[str, Any]:
    """Seed the conversation with system prompt + user query."""
    return {
        "messages": [
            SystemMessage(content=RESEARCH_SPECIALIST_PROMPT),
            HumanMessage(content=state["query"]),
        ],
    }


def _build_search_node(
    llm: BaseChatModel,
    settings: Settings,
) -> Any:
    """Return a search node closure bound to *llm* and *settings*."""

    def search_node(state: AgentState) -> dict[str, Any]:
        query = state["query"]
        all_results: list[dict[str, Any]] = []

        for hop in range(settings.max_search_hops):
            logger.info("Search hop %d/%d — %s", hop + 1, settings.max_search_hops, query)
            all_results.extend(search_all(query, settings=settings))

            # Ask LLM whether another hop is needed.
            hop_prompt = (
                f"I found results for '{query}'. "
                f"Total so far: {len(all_results)}.\n"
                "Reply DONE if sufficient, otherwise reply with a refined query only."
            )
            resp = llm.invoke(
                [
                    SystemMessage(content=RESEARCH_SPECIALIST_PROMPT),
                    HumanMessage(content=state["query"]),
                    AIMessage(content=hop_prompt),
                ]
            )
            answer = resp.content.strip()
            if answer.upper() == "DONE":
                logger.info("Sufficient after hop %d.", hop + 1)
                break
            query = answer

        sources = [
            Source(title=r["title"], url=r["url"], snippet=r["snippet"], is_scientific=False)
            for r in all_results
        ]
        return {
            "messages": [AIMessage(content=f"Collected {len(sources)} candidate sources.")],
            "sources": sources,
        }

    return search_node


def _build_review_node(llm: BaseChatModel) -> Any:
    """Return a review node closure bound to *llm*."""

    def review_node(state: AgentState) -> dict[str, Any]:
        accepted_notes: list[str] = list(state.get("research_notes", []))
        reviewed: list[Source] = []

        for src in state.get("sources", []):
            verdict = verify_source(dict(src), llm)
            is_sci = verdict.get("is_scientific", False)
            reviewed.append(
                Source(
                    title=src["title"],
                    url=src["url"],
                    snippet=src["snippet"],
                    is_scientific=is_sci,
                )
            )
            if is_sci:
                accepted_notes.append(
                    f"**{src['title']}**\nURL: {src['url']}\n{src['snippet'][:300]}"
                )
                logger.info("✓ %s", src["title"])
            else:
                logger.info("✗ %s — %s", src["title"], verdict.get("review_reason", ""))

        return {
            "messages": [
                AIMessage(content=f"{len(accepted_notes)} accepted / {len(reviewed)} total.")
            ],
            "sources": reviewed,
            "research_notes": accepted_notes,
        }

    return review_node


def _build_summary_node(llm: BaseChatModel) -> Any:
    """Return a summary node closure bound to *llm*."""

    def summary_node(state: AgentState) -> dict[str, Any]:
        notes = "\n\n---\n\n".join(state.get("research_notes", []))
        accepted = [s for s in state.get("sources", []) if s.get("is_scientific")]
        refs = "\n".join(f"[{i + 1}] {s['url']}" for i, s in enumerate(accepted)) or "(none)"

        prompt = (
            f"Original question: {state['query']}\n\n"
            f"## Verified Research Notes\n{notes}\n\n"
            f"## References\n{refs}\n\n"
            "Write a concise Markdown report with a **Summary** section, "
            "inline [1] [2] citations, and a **References** section. "
            "Do NOT invent sources."
        )
        resp = llm.invoke(
            [SystemMessage(content=RESEARCH_SPECIALIST_PROMPT), HumanMessage(content=prompt)]
        )
        return {"messages": [AIMessage(content=resp.content)]}

    return summary_node


# ── Graph assembly ───────────────────────────────────────────────────


def build_graph(settings: Settings | None = None) -> StateGraph:
    """Construct the cite-scout StateGraph.

    Flow: input → search → review → summary → END
    """
    cfg = settings or get_settings()
    llm = build_llm(cfg)

    graph = StateGraph(AgentState)
    graph.add_node("input", input_node)
    graph.add_node("search", _build_search_node(llm, cfg))
    graph.add_node("review", _build_review_node(llm))
    graph.add_node("summary", _build_summary_node(llm))

    graph.set_entry_point("input")
    graph.add_edge("input", "search")
    graph.add_edge("search", "review")
    graph.add_edge("review", "summary")
    graph.add_edge("summary", END)

    return graph
