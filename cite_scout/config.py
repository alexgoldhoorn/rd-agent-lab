"""Centralised configuration — all tunables come from env vars / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _bool(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of runtime settings."""

    # LLM
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "mistral"))
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    # Search
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    arxiv_enabled: bool = field(
        default_factory=lambda: _bool(os.getenv("ARXIV_ENABLED", "true")),
    )
    search_max_results: int = field(
        default_factory=lambda: int(os.getenv("SEARCH_MAX_RESULTS", "5")),
    )

    # Agent
    max_search_hops: int = field(
        default_factory=lambda: int(os.getenv("MAX_SEARCH_HOPS", "3")),
    )

    # Persistence
    checkpoint_db: str = field(default_factory=lambda: os.getenv("CHECKPOINT_DB", ""))


def get_settings() -> Settings:
    """Return a fresh Settings instance (re-reads env on every call)."""
    return Settings()
