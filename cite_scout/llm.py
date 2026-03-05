"""LLM factory — returns a ChatModel based on the configured provider."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from cite_scout.config import Settings, get_settings


def build_llm(settings: Settings | None = None) -> BaseChatModel:
    """Instantiate the LLM specified by *settings*.

    Supports ``openai`` and ``ollama`` providers.
    """
    cfg = settings or get_settings()

    if cfg.llm_provider == "ollama":
        # Deferred import so Ollama deps are only required when used.
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=cfg.ollama_model,
            base_url=cfg.ollama_base_url,
            temperature=0.0,
        )

    # Default: OpenAI
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=cfg.openai_model, temperature=0.0)
