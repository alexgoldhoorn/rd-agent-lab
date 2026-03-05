"""CLI entry point — ``python -m cite_scout "your question"``."""

from __future__ import annotations

import argparse
import logging
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from cite_scout.config import get_settings
from cite_scout.graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("cite-scout")


@contextmanager
def _checkpointer(db_path: str) -> Iterator[Any]:
    """Yield a SqliteSaver if *db_path* is set, otherwise ``None``."""
    if db_path:
        from langgraph.checkpoint.sqlite import SqliteSaver

        with SqliteSaver.from_conn_string(db_path) as saver:
            yield saver
    else:
        yield None


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cite-scout",
        description="Multi-hop research agent with scientific citations.",
    )
    parser.add_argument("query", help="Research question to investigate.")
    parser.add_argument("--thread-id", default=None, help="Resume a previous session.")
    args = parser.parse_args()

    settings = get_settings()
    thread_id = args.thread_id or str(uuid.uuid4())
    logger.info("thread=%s  provider=%s", thread_id, settings.llm_provider)

    graph = build_graph(settings)

    with _checkpointer(settings.checkpoint_db) as ckpt:
        compiled = graph.compile(checkpointer=ckpt)
        initial: dict[str, Any] = {
            "messages": [],
            "research_notes": [],
            "sources": [],
            "query": args.query,
        }
        config = {"configurable": {"thread_id": thread_id}}

        final = None
        for event in compiled.stream(initial, config=config):
            for node in event:
                logger.info("✔ %s", node)
            final = event

        if final:
            last = list(final.values())[-1]
            msgs = last.get("messages", [])
            if msgs:
                print("\n" + "=" * 72)
                print("CITE-SCOUT REPORT")
                print("=" * 72 + "\n")
                print(msgs[-1].content)
                print()

    if settings.checkpoint_db:
        logger.info("Session saved. Resume with: --thread-id %s", thread_id)


if __name__ == "__main__":
    main()
