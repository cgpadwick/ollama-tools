# Hugging Face GGUF → Ollama

[![CI](https://github.com/cgpadwick/ollama-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/cgpadwick/ollama-tools/actions/workflows/ci.yml)

Downloads GGUFs from a Hugging Face repo, merges split shards when needed, writes a
`Modelfile`, and imports the result into Ollama.

Works with any repo that follows the common GGUF layout used by
[unsloth](https://huggingface.co/unsloth), [bartowski](https://huggingface.co/bartowski)
and others:

```
<Model>-<quant>.gguf                        single-file quant (also <Model>.<quant>.gguf)
<dir>/<Model>-<quant>-00001-of-0000N.gguf   sharded quant (merged with llama-gguf-split)
mmproj-*.gguf                               optional vision projector
```

Tested against `unsloth/*`, `bartowski/*` and `mradermacher/*` (incl. `-i1-` imatrix) repos.

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

### Updating Ollama

Ollama only understands model architectures it has been taught. Newly released families
(e.g. Qwen3.8) and the multi-`FROM` Modelfile used for vision projectors need a recent
Ollama, otherwise `ollama create` fails with something like
`Error: unsupported architecture` or `unknown model architecture`. Check and upgrade
before importing:

```bash
ollama --version
```

```bash
curl -fsSL https://ollama.com/install.sh | sh      # Linux: re-running the installer upgrades in place
```

On macOS/Windows use the app's built-in updater or grab the latest build from
[ollama.com/download](https://ollama.com/download). Restart the Ollama server after
upgrading (`systemctl restart ollama` on Linux if installed as a service), then re-run the
import — the downloaded GGUFs are untouched and will not be fetched again.

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
uv run python gguf-to-ollama.py --repo bartowski/Meta-Llama-3.1-8B-Instruct-GGUF --quant Q6_K
uv run python gguf-to-ollama.py --repo mradermacher/Qwen3-8B-i1-GGUF                     # no --quant: picks Q4_K_M
uv run python gguf-to-ollama.py --repo unsloth/gemma-3-27b-it-GGUF --quant Q4_K_M

uv run python gguf-to-ollama.py --quant BF16 --num-ctx 65536 --name qwen3.8-27b:bf16-64k   # long-context import
```

Then:

```bash
ollama run qwen3.8-27b:ud-q4_k_m          # default name is <model>:<quant>, lower-cased
```

### Options

| Flag | Description |
| --- | --- |
| `--repo` | Hugging Face repo id (default `unsloth/Qwen3.8-27B-GGUF`) |
| `--quant` | Quant label as shown by `--list` (default: `UD-Q4_K_M` if the repo has it, else `Q4_K_M`) |
| `--list` | List available quants and exit |
| `--dir` | Download directory (default `~/models/<org>/<repo name>`) |
| `--name` | Ollama model name (default `<model>:<quant>`, e.g. `qwen3.8-27b:ud-q4_k_m`) |
| `--preset` | Modelfile sampling preset: `qwen3`, `qwen3-thinking`, `none` (default `qwen3` for Qwen3 repos, else `none`) |
| `--num-ctx` | `PARAMETER num_ctx` (default: **auto** — largest context that fits in memory, capped at the model's native length; `0` to omit and use Ollama's default) |
| `--gguf-split` | Path to `llama-gguf-split`; auto-detected from `tools/` if installed |
| `--no-mmproj` | Skip the vision projector even if the repo has one |
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
  plus a `-`, `.` or `_` separator from each filename, so `Qwen3.8-27B-UD-Q4_K_M.gguf` →
  `UD-Q4_K_M`. If no file carries that prefix (e.g. `mradermacher/X-i1-GGUF` ships
  `X.i1-Q4_K_M.gguf`), the longest common filename prefix is stripped instead. `*mmproj*`,
  `*imatrix*` and the `MTP/` directory are skipped.
- If the repo has an `mmproj` GGUF, it is fetched by default (F16 preferred) and added as a
  second `FROM` line in the generated `Modelfile`; text-only repos just import without one.
- The default Ollama name is `<model>:<quant>` without the org, so `unsloth/X-GGUF` and
  `Qwen/X-GGUF` would both import as `x:<quant>` — pass `--name` to keep both. Download
  directories do include the org and never collide.
- Sampling presets follow Unsloth's recommendations for Qwen3: `qwen3` is non-thinking mode
  (`temperature 0.7`, `top_p 0.8`), `qwen3-thinking` is (`temperature 0.6`, `top_p 0.95`).
  The default is chosen from the repo name (`*Qwen3*Thinking*` → `qwen3-thinking`,
  `*Qwen3*` → `qwen3`, embedding/reranker → `none`, anything else → `none`) and printed;
  override with `--preset`, or edit the generated `Modelfile` and re-run `ollama create`.
- **Context length (`num_ctx`)**: by default the script reads the model's GGUF header
  (native context length, layers, KV heads) and sets the largest `num_ctx` — in 4096
  steps, capped at the native length — whose f16 KV cache fits in currently available
  memory next to the weights, and prints the calculation. Pass `--num-ctx N` to pin a
  value (e.g. `--num-ctx 65536`), or `--num-ctx 0` to omit the parameter and use
  Ollama's own default. The estimate uses system RAM (`MemAvailable`); on a
  discrete-GPU box where the KV cache must fit in VRAM, set `--num-ctx` yourself.
  Mind the tradeoff: on a 27B model a 64K context adds tens of GB, so a q8_0 or q4
  quant leaves far more headroom than BF16. Re-running with a different
  `--num-ctx`/`--name` reuses the downloaded GGUFs — only `ollama create` runs again.
- Merging requires `llama-gguf-split` (see `install-llama-tools.sh`). If the merged file
  already exists, the tool is not needed and the shards are not re-downloaded, so you can
  delete them after a successful merge. The merge writes to a `.part` file and renames it
  on success; an interrupted merge is cleaned up and redone next run. The merged file is
  ~96 bytes larger than a natively-unsplit one because llama.cpp leaves `split.*` metadata
  keys behind with `split.count = 0`; tensor data is identical and loaders ignore them.
- If `~/.cache/huggingface` is not writable, the script automatically falls back to
  `~/.cache/huggingface-local`.
- Sharded builds need roughly their total size in *additional* free space to merge. The
  script checks for download + merge space up front (using sizes from the Hub) before
  pulling anything; e.g. Qwen3.8-27B BF16 is ~54 GB.
