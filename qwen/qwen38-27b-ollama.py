#!/usr/bin/env python3
"""Download Qwen3.8-27B GGUFs from Hugging Face, merge split shards, and import into Ollama.

Note: https://huggingface.co/unsloth/Qwen3.8-27B holds safetensors, not GGUFs.
The GGUFs live in https://huggingface.co/unsloth/Qwen3.8-27B-GGUF (used below).
Only the BF16 build is sharded (2 files); every quant is a single file.

Requires: huggingface_hub  (pip install huggingface_hub)
Optional: llama-gguf-split from llama.cpp (only needed for split builds such as BF16)

Examples:
    ./qwen38-27b-ollama.py --list
    ./qwen38-27b-ollama.py --quant UD-Q4_K_M
    ./qwen38-27b-ollama.py --quant BF16 --gguf-split /opt/llama.cpp/build/bin/llama-gguf-split
    ./qwen38-27b-ollama.py --quant UD-Q6_K --no-mmproj --name qwen38:q6k
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

REPO_ID = "unsloth/Qwen3.8-27B-GGUF"
DEFAULT_QUANT = "UD-Q4_K_M"
SCRIPT_DIR = Path(__file__).resolve().parent
# Matches "<stem>-00001-of-00002.gguf"
SPLIT_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")

# Must be set before huggingface_hub is imported for it to take effect.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

# Some machines ship a root-owned ~/.cache/huggingface, which makes the hub cache
# unwritable. Fall back to a per-user cache so downloads still work.
if "HF_HOME" not in os.environ:
    _default_hf_home = Path.home() / ".cache" / "huggingface"
    if _default_hf_home.exists() and not os.access(_default_hf_home, os.W_OK):
        _fallback = Path.home() / ".cache" / "huggingface-local"
        _fallback.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(_fallback)
        print(f"note: {_default_hf_home} is not writable; using {_fallback}", file=sys.stderr)


def die(msg: str) -> NoReturn:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def list_repo_ggufs() -> list[str]:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        die("huggingface_hub is not installed. Run: pip install huggingface_hub")

    return sorted(f for f in HfApi().list_repo_files(REPO_ID) if f.endswith(".gguf"))


def group_quants(files: list[str]) -> dict[str, list[str]]:
    """Map quant label -> ordered list of repo file paths belonging to it."""
    quants: dict[str, list[str]] = {}
    for path in files:
        name = Path(path).name
        if name.startswith(("mmproj", "imatrix")) or path.startswith("MTP/"):
            continue

        m = SPLIT_RE.match(name)
        stem = m.group("stem") if m else name[: -len(".gguf")]
        # "Qwen3.8-27B-UD-Q4_K_M" -> "UD-Q4_K_M"
        quants.setdefault(stem.replace("Qwen3.8-27B-", "", 1), []).append(path)

    for paths in quants.values():
        paths.sort()
    return quants


def hf_download(patterns: list[str], dest: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        die("huggingface_hub is not installed. Run: pip install huggingface_hub")

    dest.mkdir(parents=True, exist_ok=True)
    print(f"==> downloading {patterns} -> {dest}")
    snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=patterns,
        local_dir=str(dest),
        max_workers=int(os.environ.get("HF_WORKERS", "4")),
    )


def find_gguf_split(explicit: str | None) -> str:
    candidates: list[str] = [c for c in (explicit,) if c]
    # Binaries vendored next to this script, e.g. tools/llama-b10599/llama-gguf-split
    candidates += [str(p) for p in sorted(SCRIPT_DIR.glob("tools/*/llama-gguf-split"), reverse=True)]
    candidates += ["llama-gguf-split", "gguf-split"]

    for cand in candidates:
        found = shutil.which(cand) or (cand if Path(cand).is_file() else None)
        if found:
            return str(Path(found).resolve())
    die(
        "llama-gguf-split not found; it is needed to merge split shards.\n"
        "Run ./install-llama-tools.sh, or pass --gguf-split /path/to/llama-gguf-split"
    )


def merge_shards(first_shard: Path, out_path: Path, tool: str | None) -> Path:
    # Resolve the tool lazily so an existing merge does not require llama.cpp.
    print(f"==> merging shards into {out_path}")
    subprocess.run(
        [find_gguf_split(tool), "--merge", str(first_shard), str(out_path)], check=True
    )
    return out_path


def resolve_model_file(paths: list[str], workdir: Path, tool: str | None) -> Path:
    local = [workdir / p for p in paths]
    if len(local) == 1:
        return local[0]

    first = local[0]
    m = SPLIT_RE.match(first.name)
    if not m:
        die(f"unexpected multi-file quant layout: {[p.name for p in local]}")

    merged = first.parent / (m.group("stem") + ".gguf")
    if merged.exists():
        print(f"==> merged file already exists: {merged}")
        return merged
    return merge_shards(first, merged, tool)


def write_modelfile(model: Path, mmproj: Path | None, workdir: Path) -> Path:
    lines = [f"FROM {model.resolve()}"]
    if mmproj is not None:
        lines.append(f"FROM {mmproj.resolve()}")
    lines += [
        "",
        # Unsloth's recommended sampling defaults for Qwen3.
        "PARAMETER temperature 0.7",
        "PARAMETER top_p 0.8",
        "PARAMETER top_k 20",
        "PARAMETER min_p 0.0",
        "PARAMETER repeat_penalty 1.05",
        "PARAMETER num_ctx 8192",
        "",
    ]
    path = workdir / "Modelfile"
    path.write_text("\n".join(lines))
    print(f"==> wrote {path}")
    return path


def ollama_create(name: str, modelfile: Path) -> None:
    if shutil.which("ollama") is None:
        print("warning: ollama not on PATH; skipping import", file=sys.stderr)
        print(f"run manually: ollama create {name} -f {modelfile}")
        return
    print(f"==> ollama create {name}")
    subprocess.run(["ollama", "create", name, "-f", str(modelfile)], check=True)
    print(f"\nDone. Run it with:  ollama run {name}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--quant", default=DEFAULT_QUANT, help=f"quant label (default: {DEFAULT_QUANT})")
    ap.add_argument("--list", action="store_true", help="list available quants and exit")
    ap.add_argument("--dir", type=Path, default=Path.home() / "models" / "Qwen3.8-27B-GGUF")
    ap.add_argument("--name", help="Ollama model name (default: qwen3.8-27b:<quant>)")
    ap.add_argument("--gguf-split", help="path to llama-gguf-split (for split builds)")
    ap.add_argument("--no-mmproj", action="store_true", help="skip the vision projector")
    ap.add_argument("--download-only", action="store_true", help="download/merge but do not import")
    args = ap.parse_args()

    quants = group_quants(list_repo_ggufs())

    if args.list:
        for label, paths in sorted(quants.items()):
            tag = f"  ({len(paths)} shards, needs merge)" if len(paths) > 1 else ""
            print(f"{label}{tag}")
        return

    if args.quant not in quants:
        die(f"unknown quant {args.quant!r}. Use --list to see options.")

    paths = quants[args.quant]
    patterns = list(paths)
    if not args.no_mmproj:
        patterns.append("mmproj-F16.gguf")

    hf_download(patterns, args.dir)

    model = resolve_model_file(paths, args.dir, args.gguf_split)
    mmproj = None if args.no_mmproj else args.dir / "mmproj-F16.gguf"

    modelfile = write_modelfile(model, mmproj, args.dir)
    if args.download_only:
        print(f"Skipping import. Later: ollama create <name> -f {modelfile}")
        return

    ollama_create(args.name or f"qwen3.8-27b:{args.quant.lower()}", modelfile)


if __name__ == "__main__":
    main()
