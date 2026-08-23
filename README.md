# ollama-tools

Utilities for getting models into [Ollama](https://ollama.com).

## Tools

| Directory | Description |
| --- | --- |
| [`gguf-to-ollama/`](gguf-to-ollama/) | Download GGUFs from any Hugging Face repo (unsloth, bartowski, …), merge split shards, and import into Ollama |

## Requirements

- [uv](https://docs.astral.sh/uv/) for Python environments
- [Ollama](https://ollama.com) on your `PATH`

Each tool directory is a self-contained uv project — see its README for usage.
