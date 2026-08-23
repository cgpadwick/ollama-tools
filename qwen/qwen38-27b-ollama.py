#!/usr/bin/env python3
"""Download Qwen3.8-27B GGUFs from Hugging Face, merge split shards, and import into Ollama.

Note: https://huggingface.co/unsloth/Qwen3.8-27B holds safetensors, not GGUFs.
The GGUFs live in https://huggingface.co/unsloth/Qwen3.8-27B-GGUF (used below).
Only the BF16 build is sharded (2 files); every quant is a single file.

Requires: huggingface_hub  (run `uv sync` in this directory)
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
# Matches the build number in vendored dirs like "tools/llama-b10599"
BUILD_RE = re.compile(r"b(\d+)")

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


def _import_hf():
    try:
        import huggingface_hub
    except ImportError:
        die("huggingface_hub is not installed. Run: uv sync")
    return huggingface_hub


def fetch_repo_ggufs() -> tuple[list[str], dict[str, int]]:
    """Return (sorted .gguf paths in the repo, {path: size in bytes})."""
    hf = _import_hf()
    try:
        info = hf.HfApi().model_info(REPO_ID, files_metadata=True)
    except Exception as e:  # network, auth, 404 — all fatal here
        die(f"could not list files in {REPO_ID}: {e}")
    sizes = {s.rfilename: s.size or 0 for s in info.siblings if s.rfilename.endswith(".gguf")}
    return sorted(sizes), sizes


def list_repo_ggufs() -> list[str]:
    return fetch_repo_ggufs()[0]


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


def find_mmproj(files: list[str]) -> str | None:
    """Pick the vision projector from the repo listing; prefer F16, else any mmproj."""
    mmprojs = sorted(f for f in files if Path(f).name.startswith("mmproj"))
    for f in mmprojs:
        if Path(f).name == "mmproj-F16.gguf":
            return f
    return mmprojs[0] if mmprojs else None


def hf_workers() -> int:
    raw = os.environ.get("HF_WORKERS", "4")
    try:
        n = int(raw)
    except ValueError:
        die(f"HF_WORKERS must be an integer, got {raw!r}")
    if n < 1:
        die(f"HF_WORKERS must be >= 1, got {n}")
    return n


def hf_download(patterns: list[str], dest: Path) -> None:
    hf = _import_hf()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"==> downloading {patterns} -> {dest}")
    try:
        hf.snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=patterns,
            local_dir=str(dest),
            max_workers=hf_workers(),
        )
    except Exception as e:
        die(f"download from {REPO_ID} failed: {e}")


def vendored_gguf_splits(root: Path) -> list[Path]:
    """llama-gguf-split binaries under root/tools/*/, newest build first."""

    def build_num(p: Path) -> int:
        m = BUILD_RE.search(p.parent.name)
        return int(m.group(1)) if m else -1

    return sorted(root.glob("tools/*/llama-gguf-split"), key=build_num, reverse=True)


def find_gguf_split(explicit: str | None) -> str:
    def resolve(cand: str) -> str | None:
        found = shutil.which(cand) or (cand if Path(cand).is_file() else None)
        return str(Path(found).resolve()) if found else None

    if explicit:
        found = resolve(explicit)
        if not found:
            die(f"--gguf-split {explicit!r} not found (not a file and not on PATH)")
        return found

    # Binaries vendored next to this script, e.g. tools/llama-b10599/llama-gguf-split
    candidates = [str(p) for p in vendored_gguf_splits(SCRIPT_DIR)]
    candidates += ["llama-gguf-split", "gguf-split"]
    for cand in candidates:
        found = resolve(cand)
        if found:
            return found
    die(
        "llama-gguf-split not found; it is needed to merge split shards.\n"
        "Run ./install-llama-tools.sh, or pass --gguf-split /path/to/llama-gguf-split"
    )


def check_free_space(need: int, where: Path, what: str) -> None:
    where.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(where).free
    if free < need:
        die(
            f"not enough free space in {where} to {what}: need ~{need / 2**30:.1f} GiB, "
            f"have {free / 2**30:.1f} GiB"
        )


def merge_shards(shards: list[Path], out_path: Path, tool: str | None) -> Path:
    # Resolve the tool lazily so an existing merge does not require llama.cpp.
    tool_path = find_gguf_split(tool)
    missing = [p for p in shards if not p.exists()]
    if missing:
        die(f"shards missing on disk: {[p.name for p in missing]}")
    check_free_space(sum(p.stat().st_size for p in shards), out_path.parent, "merge")

    # Write to a .part file and rename on success so an interrupted merge never
    # leaves a truncated file that a later run mistakes for a finished one.
    # llama-gguf-split refuses to overwrite, so clear any stale .part first.
    part = out_path.with_name(out_path.name + ".part")
    part.unlink(missing_ok=True)
    print(f"==> merging shards into {out_path}")
    try:
        subprocess.run([tool_path, "--merge", str(shards[0]), str(part)], check=True)
        part.replace(out_path)
    except subprocess.CalledProcessError as e:
        die(f"llama-gguf-split failed with exit code {e.returncode}")
    except OSError as e:
        die(f"could not run {tool_path}: {e}")
    finally:
        part.unlink(missing_ok=True)
    return out_path


def merged_path_for(paths: list[str], workdir: Path) -> Path | None:
    """Return the merged output path for a sharded quant, or None if single-file."""
    first = workdir / paths[0]
    m = SPLIT_RE.match(first.name)
    if not m:
        if len(paths) == 1:
            return None
        die(f"unexpected multi-file quant layout: {[Path(p).name for p in paths]}")
    total = int(m.group("total"))
    if total != len(paths):
        die(f"quant declares {total} shards but the repo lists {len(paths)}: {paths}")
    if total == 1:
        return None  # "-00001-of-00001": nothing to merge, use the file as-is
    return first.parent / (m.group("stem") + ".gguf")


def resolve_model_file(paths: list[str], workdir: Path, tool: str | None) -> Path:
    merged = merged_path_for(paths, workdir)
    if merged is None:
        return workdir / paths[0]
    if merged.exists():
        return merged
    return merge_shards([workdir / p for p in paths], merged, tool)


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
    try:
        subprocess.run(["ollama", "create", name, "-f", str(modelfile)], check=True)
    except subprocess.CalledProcessError as e:
        die(f"ollama create failed with exit code {e.returncode}")
    print(f"\nDone. Run it with:  ollama run {name}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        epilog="Set HF_WORKERS to change download parallelism (default 4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--quant", default=DEFAULT_QUANT, help=f"quant label (default: {DEFAULT_QUANT})")
    ap.add_argument("--list", action="store_true", help="list available quants and exit")
    ap.add_argument("--dir", type=Path, default=Path.home() / "models" / "Qwen3.8-27B-GGUF")
    ap.add_argument("--name", help="Ollama model name (default: qwen3.8-27b:<quant>)")
    ap.add_argument("--gguf-split", help="path to llama-gguf-split (for split builds)")
    ap.add_argument("--no-mmproj", action="store_true", help="skip the vision projector")
    ap.add_argument("--download-only", action="store_true", help="download/merge but do not import")
    args = ap.parse_args()

    files, sizes = fetch_repo_ggufs()
    quants = group_quants(files)

    if args.list:
        for label, paths in sorted(quants.items()):
            tag = f"  ({len(paths)} shards, needs merge)" if len(paths) > 1 else ""
            print(f"{label}{tag}")
        return

    if args.quant not in quants:
        die(f"unknown quant {args.quant!r}. Use --list to see options.")

    paths = quants[args.quant]
    patterns: list[str] = []

    # Skip re-downloading shards when a previous run already produced the merged file
    # (the shards may have been deleted to reclaim space).
    merged = merged_path_for(paths, args.dir)
    if merged is not None and merged.exists():
        print(f"==> merged file already exists, skipping shard download: {merged}")
    else:
        patterns += paths

    mmproj_repo = None
    if not args.no_mmproj:
        mmproj_repo = find_mmproj(files)
        if mmproj_repo is None:
            die(f"no mmproj-*.gguf in {REPO_ID}; pass --no-mmproj to import text-only")
        patterns.append(mmproj_repo)

    if patterns:
        # Check disk up front: bytes still to download, plus the merged copy if sharded.
        need = sum(sizes.get(p, 0) for p in patterns if not (args.dir / p).exists())
        if merged is not None and not merged.exists():
            need += sum(sizes.get(p, 0) for p in paths)
        check_free_space(need, args.dir, "download and merge" if merged else "download")
        hf_download(patterns, args.dir)

    model = resolve_model_file(paths, args.dir, args.gguf_split)
    mmproj = None
    if mmproj_repo is not None:
        mmproj = args.dir / mmproj_repo
        if not mmproj.exists():
            die(f"expected {mmproj} after download but it is missing")

    modelfile = write_modelfile(model, mmproj, args.dir)
    if args.download_only:
        print(f"Skipping import. Later: ollama create <name> -f {modelfile}")
        return

    ollama_create(args.name or f"qwen3.8-27b:{args.quant.lower()}", modelfile)


if __name__ == "__main__":
    main()
