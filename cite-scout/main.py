#!/usr/bin/env python3
"""cite-scout — CLI research agent with multi-hop search and scientific citations.

Usage:
    python main.py "What are the latest advances in retrieval-augmented generation?"

Requires environment variables:
    OPENAI_API_KEY   — for the LLM calls.
    TAVILY_API_KEY   — for Tavily web search.
"""

from __future__ import annotations

import argparse
import logging
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from prompts import RESEARCH_SPECIALIST_PROMPT
from state import AgentState, Source
from tools import search_all, verify_source_dict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("cite-scout")

# Path to the local SQLite checkpoint database (resides next to this file).
_CHECKPOINT_DB = Path(__file__).resolve().parent / ".citescout_checkpoints.db"

# Maximum number of multi-hop search iterations.
_MAX_SEARCH_HOPS = 3


# ---------------------------------------------------------------------------
# Shared LLM instance (lazily initialised)
# ---------------------------------------------------------------------------

def _get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI instance used by every graph node."""
    return ChatOpenAI(model="gpt-4o", temperature=0.0)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def input_node(state: AgentState) -> dict[str, Any]:
    """Seed the conversation with the system prompt and the user query."""
    return {
        "messages": [
            SystemMessage(content=RESEARCH_SPECIALIST_PROMPT),
            HumanMessage(content=state["query"]),
        ],
    }


def search_node(state: AgentState) -> dict[str, Any]:
    """Run multi-hop searches across Tavily and arXiv.

    The LLM is asked to refine the query on each hop.  Results accumulate in
    ``sources`` across hops.
    """
    llm = _get_llm()
    query = state["query"]
    all_results: list[dict[str, Any]] = []

    for hop in range(_MAX_SEARCH_HOPS):
        logger.info("Search hop %d/%d — query: %s", hop + 1, _MAX_SEARCH_HOPS, query)
        results = search_all(query)
        all_results.extend(results)

        # Ask the LLM whether we need another hop.
        hop_msg = (
            f"I found {len(results)} results for '{query}'. "
            f"Total results so far: {len(all_results)}.\n\n"
            "If the results are sufficient to answer the original research "
            "question, reply with exactly: DONE\n"
            "Otherwise, reply with a refined search query (just the query, "
            "nothing else)."
        )
        response = llm.invoke(
            [
                SystemMessage(content=RESEARCH_SPECIALIST_PROMPT),
                HumanMessage(content=state["query"]),
                AIMessage(content=hop_msg),
            ]
        )
        answer = response.content.strip()

        if answer.upper() == "DONE":
            logger.info("LLM decided results are sufficient after hop %d.", hop + 1)
            break

        # Use the refined query for the next hop.
        query = answer

    # Convert raw dicts into Source-shaped dicts (is_scientific filled later).
    sources: list[Source] = [
        Source(
            title=r.get("title", "Untitled"),
            url=r.get("url", ""),
            snippet=r.get("snippet", ""),
            is_scientific=False,
        )
        for r in all_results
    ]

    return {
        "messages": [
            AIMessage(content=f"Search complete — collected {len(sources)} candidate sources."),
        ],
        "sources": sources,
    }


def review_node(state: AgentState) -> dict[str, Any]:
    """Verify each source for scientific / technical quality.

    Only sources that pass verification are added to ``research_notes``.
    """
    llm = _get_llm()
    accepted_notes: list[str] = list(state.get("research_notes", []))
    reviewed_sources: list[Source] = []

    for src in state.get("sources", []):
        # Build a plain dict for the verification helper.
        result_dict: dict[str, Any] = {
            "title": src["title"],
            "url": src["url"],
            "snippet": src["snippet"],
        }
        verified = verify_source_dict(result_dict, llm)

        is_sci = verified.get("is_scientific", False)
        reviewed_sources.append(
            Source(
                title=src["title"],
                url=src["url"],
                snippet=src["snippet"],
                is_scientific=is_sci,
            )
        )

        if is_sci:
            note = (
                f"**{src['title']}**\n"
                f"URL: {src['url']}\n"
                f"Finding: {src['snippet'][:300]}"
            )
            accepted_notes.append(note)
            logger.info("✓ Accepted: %s", src["title"])
        else:
            logger.info("✗ Rejected: %s — %s", src["title"], verified.get("review_reason", ""))

    return {
        "messages": [
            AIMessage(
                content=(
                    f"Review complete — {len(accepted_notes)} source(s) accepted "
                    f"out of {len(reviewed_sources)} candidates."
                )
            ),
        ],
        "sources": reviewed_sources,
        "research_notes": accepted_notes,
    }


def summary_node(state: AgentState) -> dict[str, Any]:
    """Synthesise a Markdown report with inline citations and a References section."""
    llm = _get_llm()

    # Build the context block the LLM will use to write the summary.
    notes_block = "\n\n---\n\n".join(state.get("research_notes", []))

    # Build a references list from accepted sources.
    accepted = [s for s in state.get("sources", []) if s.get("is_scientific")]
    ref_lines = [f"[{i + 1}] {s['url']}" for i, s in enumerate(accepted)]
    ref_block = "\n".join(ref_lines) if ref_lines else "(no references)"

    synthesis_prompt = (
        f"Original question: {state['query']}\n\n"
        f"## Verified Research Notes\n{notes_block}\n\n"
        f"## Available References\n{ref_block}\n\n"
        "Using ONLY the verified research notes and references above, write a "
        "concise Markdown report that:\n"
        "1. Has a **Summary** section answering the question.\n"
        "2. Uses inline bracket citations like [1], [2] mapped to the references.\n"
        "3. Ends with a **References** section listing each numbered URL.\n"
        "Do NOT invent sources that are not listed above."
    )

    response = llm.invoke(
        [
            SystemMessage(content=RESEARCH_SPECIALIST_PROMPT),
            HumanMessage(content=synthesis_prompt),
        ]
    )

    return {
        "messages": [AIMessage(content=response.content)],
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Construct the cite-scout LangGraph state graph.

    Flow: input → search → review → summary → END
    """
    graph = StateGraph(AgentState)

    # Register nodes.
    graph.add_node("input", input_node)
    graph.add_node("search", search_node)
    graph.add_node("review", review_node)
    graph.add_node("summary", summary_node)

    # Define edges (linear pipeline).
    graph.set_entry_point("input")
    graph.add_edge("input", "search")
    graph.add_edge("search", "review")
    graph.add_edge("review", "summary")
    graph.add_edge("summary", END)

    return graph


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and execute the research graph."""
    parser = argparse.ArgumentParser(
        prog="cite-scout",
        description="Multi-hop research agent with scientific citations.",
    )
    parser.add_argument(
        "query",
        type=str,
        help="The research question to investigate.",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help="Resume a previous session by providing its thread ID.",
    )
    args = parser.parse_args()

    # Use a deterministic or user-supplied thread ID for checkpoint resumption.
    thread_id = args.thread_id or str(uuid.uuid4())
    logger.info("Session thread ID: %s", thread_id)

    # Compile the graph with a local SQLite checkpointer.
    graph = build_graph()
    with SqliteSaver.from_conn_string(str(_CHECKPOINT_DB)) as checkpointer:
        compiled = graph.compile(checkpointer=checkpointer)

        # Initial state seeded with the user query.
        initial_state: dict[str, Any] = {
            "messages": [],
            "research_notes": [],
            "sources": [],
            "query": args.query,
        }

        config = {"configurable": {"thread_id": thread_id}}

        # Stream events so the user sees incremental progress in the terminal.
        logger.info("Starting research for: %s", args.query)
        final_state: dict[str, Any] | None = None
        for event in compiled.stream(initial_state, config=config):
            # Each event is a dict keyed by node name.
            for node_name, node_output in event.items():
                logger.info("Completed node: %s", node_name)
            final_state = event

        # Print the final synthesised report.
        if final_state:
            # The last node output contains the summary message.
            last_output = list(final_state.values())[-1]
            messages = last_output.get("messages", [])
            if messages:
                print("\n" + "=" * 72)
                print("CITE-SCOUT REPORT")
                print("=" * 72 + "\n")
                print(messages[-1].content)
                print()

    logger.info("Checkpoints saved to %s", _CHECKPOINT_DB)
    logger.info("To resume this session, pass --thread-id %s", thread_id)


if __name__ == "__main__":
    main()
