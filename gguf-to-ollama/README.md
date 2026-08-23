# Hugging Face GGUF → Ollama

Downloads GGUFs from a Hugging Face repo, merges split shards when needed, writes a
`Modelfile`, and imports the result into Ollama.

Works with any repo that follows the common GGUF layout used by
[unsloth](https://huggingface.co/unsloth), [bartowski](https://huggingface.co/bartowski)
and others:

```
<Model>-<quant>.gguf                        single-file quant
<dir>/<Model>-<quant>-00001-of-0000N.gguf   sharded quant (merged with llama-gguf-split)
mmproj-*.gguf                               optional vision projector
```

The default repo is [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF);
pass `--repo` for anything else.

## Setup

```bash
cd gguf-to-ollama
uv sync
```

Only if you want a **sharded** build (e.g. BF16 on most repos, or large models where every
quant is split), also install `llama-gguf-split`:

```bash
./install-llama-tools.sh
```

This drops prebuilt llama.cpp binaries into `tools/`, which the script finds automatically.
The installer is idempotent and keeps only the newest build. Set `GITHUB_TOKEN` if you hit
the unauthenticated GitHub API rate limit.

### Do I need `HF_TOKEN`?

**No, for public repos** — everything under `unsloth/*` and `bartowski/*` downloads
anonymously. The script was tested without any token.

**Yes, for gated repos** such as `meta-llama/*` or `google/gemma*`: listing works
anonymously but downloads fail with `GatedRepoError 401`. Accept the license on the model
page, then either:

```bash
export HF_TOKEN=hf_...
```

or run `uv run hf auth login` once (the token is cached under `~/.cache/huggingface`).
The script prints a hint when a download fails for this reason.

## Usage

```bash
uv run python gguf-to-ollama.py --list                                   # quants in the default repo
uv run python gguf-to-ollama.py --quant UD-Q4_K_M                        # download + import
uv run python gguf-to-ollama.py --quant BF16                             # sharded; auto-finds tools/*/llama-gguf-split

uv run python gguf-to-ollama.py --repo bartowski/Meta-Llama-3.1-8B-Instruct-GGUF --list
uv run python gguf-to-ollama.py --repo bartowski/Meta-Llama-3.1-8B-Instruct-GGUF --quant Q6_K --no-mmproj
uv run python gguf-to-ollama.py --repo unsloth/gemma-3-27b-it-GGUF --quant Q4_K_M
```

Then:

```bash
ollama run qwen3.8-27b:ud-q4_k_m          # default name is <model>:<quant>, lower-cased
```

### Options

| Flag | Description |
| --- | --- |
| `--repo` | Hugging Face repo id (default `unsloth/Qwen3.8-27B-GGUF`) |
| `--quant` | Quant label as shown by `--list` (default `UD-Q4_K_M`) |
| `--list` | List available quants and exit |
| `--dir` | Download directory (default `~/models/<repo name>`) |
| `--name` | Ollama model name (default `<model>:<quant>`, e.g. `qwen3.8-27b:ud-q4_k_m`) |
| `--preset` | Modelfile sampling preset: `qwen3`, `qwen3-thinking`, `none` (default `qwen3` for Qwen3 repos, else `none`) |
| `--num-ctx` | `PARAMETER num_ctx` (default `8192`; `0` to omit and use Ollama's default) |
| `--gguf-split` | Path to `llama-gguf-split`; auto-detected from `tools/` if installed |
| `--no-mmproj` | Skip the vision projector |
| `--download-only` | Download and merge, but skip `ollama create` |

Environment: `HF_WORKERS` sets download parallelism (default 4); `HF_TOKEN` for gated repos
(see above); `HF_HOME` to relocate the Hugging Face cache.

### Tests

```bash
uv sync --group dev
uv run pytest
```

## Notes

- Quant labels are derived by stripping the repo's model name (`<repo name>` minus `-GGUF`)
  from each filename, so `Qwen3.8-27B-UD-Q4_K_M.gguf` → `UD-Q4_K_M`. Files that don't carry
  the prefix keep their full stem. `mmproj-*`, `imatrix*` and the `MTP/` directory are skipped.
- If the repo has an `mmproj-*.gguf`, it is fetched by default (F16 preferred) and added as a
  second `FROM` line in the generated `Modelfile`. Repos without one need `--no-mmproj`.
- Sampling presets follow Unsloth's recommendations for Qwen3: `qwen3` is non-thinking mode
  (`temperature 0.7`, `top_p 0.8`), `qwen3-thinking` is (`temperature 0.6`, `top_p 0.95`).
  For other families use `none` and/or edit the generated `Modelfile`, then re-run
  `ollama create`. `num_ctx 8192` is a memory tradeoff; raise it with `--num-ctx`.
- Merging requires `llama-gguf-split` (see `install-llama-tools.sh`). If the merged file
  already exists, the tool is not needed and the shards are not re-downloaded, so you can
  delete them after a successful merge. The merge writes to a `.part` file and renames it
  on success; an interrupted merge is cleaned up and redone next run. The merged file is
  ~96 bytes larger than a natively-unsplit one because llama.cpp leaves `split.*` metadata
  keys behind with `split.count = 0`; tensor data is identical and loaders ignore them.
- If `~/.cache/huggingface` is not writable, the script automatically falls back to
  `~/.cache/huggingface-local`.
- Sharded builds need roughly their total size in *additional* free space to merge (the
  script checks before starting); e.g. Qwen3.8-27B BF16 is ~54 GB.
