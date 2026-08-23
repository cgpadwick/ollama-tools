# ollama-tools

[![CI](https://github.com/cgpadwick/ollama-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/cgpadwick/ollama-tools/actions/workflows/ci.yml)

Utilities for getting models into [Ollama](https://ollama.com).

## Tools

| Directory | Description |
| --- | --- |
| [`gguf-to-ollama/`](gguf-to-ollama/) | Download GGUFs from any Hugging Face repo (unsloth, bartowski, …), merge split shards, and import into Ollama |

## Requirements

- [uv](https://docs.astral.sh/uv/) for Python environments
- [Ollama](https://ollama.com) on your `PATH`

Each tool directory is a self-contained uv project — see its README for usage.

## Development

CI runs on every pull request: `pytest` on Python 3.10 and 3.13, plus `shellcheck` on the
installer. Run the same locally:

```bash
cd gguf-to-ollama
uv sync --group dev
uv run pytest
```
