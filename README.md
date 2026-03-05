# cite-scout

CLI research agent that performs multi-hop web searches, verifies source quality, and produces Markdown reports with scientific-style citations.

Built with **LangGraph** + **LangChain**.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in your keys
python -m cite_scout "What are the latest advances in retrieval-augmented generation?"
```

## Configuration

All settings live in `.env` (or as environment variables). See `.env.example` for the full list.

### LLM provider

| Provider | `LLM_PROVIDER` | Required env vars |
|----------|----------------|-------------------|
| OpenAI   | `openai`       | `OPENAI_API_KEY`, optionally `OPENAI_MODEL` (default `gpt-4o`) |
| Ollama   | `ollama`       | `OLLAMA_MODEL`, optionally `OLLAMA_BASE_URL` (default `http://localhost:11434`) |

**Recommended local models** (Ollama, with tool-calling support):

- **Mistral 7B** — fast, reliable tool use
- **Llama 3.1 8B / 70B** — strong reasoning
- **Command-R 35B** — purpose-built for RAG + tools
- **Qwen 2.5 7B / 72B** — good multilingual + tool support

```bash
ollama pull mistral
LLM_PROVIDER=ollama OLLAMA_MODEL=mistral python -m cite_scout "your question"
```

### Search tools

- **Tavily** — set `TAVILY_API_KEY`. Skipped gracefully if unset.
- **arXiv** — free, enabled by default. Disable with `ARXIV_ENABLED=false`.

### Persistence

Set `CHECKPOINT_DB` to a file path for SQLite-backed session resumption. Leave empty for in-memory only.

```bash
python -m cite_scout --thread-id <id> "follow-up question"
```

## Project structure

```
cite_scout/
  __init__.py      # package marker
  __main__.py      # CLI entry point
  config.py        # settings loaded from env / .env
  llm.py           # LLM factory (OpenAI / Ollama)
  state.py         # AgentState TypedDict
  prompts.py       # system prompts
  tools.py         # Tavily + arXiv search wrappers
  graph.py         # LangGraph nodes + graph assembly
tests/
  test_state.py
  test_tools.py
  test_graph.py
```

## Tests

```bash
pytest
```

## License

MIT
