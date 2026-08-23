# Qwen3.8-27B → Ollama

Downloads Qwen3.8-27B GGUFs from Hugging Face, merges split shards when needed, and imports the result into Ollama.

> The GGUFs are in [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).
> The plain [`unsloth/Qwen3.8-27B`](https://huggingface.co/unsloth/Qwen3.8-27B) repo is safetensors only.
> Only the **BF16** build is sharded (2 files); every quant is a single file, so merging is BF16-only.

## Setup

```bash
cd qwen
uv sync
```

Only if you want the **BF16** build (the sole sharded one), also install `llama-gguf-split`:

```bash
./install-llama-tools.sh
```

This drops prebuilt llama.cpp binaries into `tools/`, which the script finds automatically.

## Usage

```bash
uv run python qwen38-27b-ollama.py --list                 # show available quants
uv run python qwen38-27b-ollama.py --quant UD-Q4_K_M      # download + import
uv run python qwen38-27b-ollama.py --quant BF16           # auto-finds tools/*/llama-gguf-split
```

Add `--download-only` to skip the Ollama import, or `--dir` to change where GGUFs land.

Then:

```bash
ollama run qwen3.8-27b:ud-q4_k_m
```

### Options

| Flag | Description |
| --- | --- |
| `--quant` | Quant label (default `UD-Q4_K_M`) |
| `--list` | List available quants and exit |
| `--dir` | Download directory (default `~/models/Qwen3.8-27B-GGUF`) |
| `--name` | Ollama model name (default `qwen3.8-27b:<quant>`) |
| `--gguf-split` | Path to `llama-gguf-split`; auto-detected from `tools/` if installed |
| `--no-mmproj` | Skip the vision projector (`mmproj-F16.gguf`) |
| `--download-only` | Download and merge, but skip `ollama create` |

Set `HF_WORKERS` to change download parallelism (default 4).

## Notes

- The model is multimodal, so `mmproj-F16.gguf` is fetched by default and added as a
  second `FROM` line in the generated `Modelfile`.
- Sampling defaults in the `Modelfile` follow Unsloth's recommendations for Qwen3.
- Merging BF16 requires `llama-gguf-split` (see `install-llama-tools.sh`). If the merged
  file already exists, the tool is not needed. The merged file is ~96 bytes larger than a
  natively-unsplit one because llama.cpp leaves `split.*` metadata keys behind with
  `split.count = 0`; tensor data is identical and loaders ignore them.
- If `~/.cache/huggingface` is not writable, the script automatically falls back to
  `~/.cache/huggingface-local`.
- BF16 is ~54 GB across both shards, and merging needs roughly that much *additional*
  free space before the shards can be deleted.
