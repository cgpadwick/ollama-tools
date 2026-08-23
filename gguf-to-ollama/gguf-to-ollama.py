#!/usr/bin/env python3
"""Download GGUFs from a Hugging Face repo, merge split shards, and import into Ollama.

Works with any repo laid out like the unsloth/bartowski GGUF repos:
    <Model>-<quant>.gguf                       single-file quant
    <dir>/<Model>-<quant>-00001-of-0000N.gguf  sharded quant (merged with llama-gguf-split)
    mmproj-*.gguf                              optional vision projector

Requires: huggingface_hub  (run `uv sync` in this directory)
Optional: llama-gguf-split from llama.cpp (only needed for sharded builds such as BF16)
Optional: HF_TOKEN for gated repos (e.g. meta-llama/*, google/gemma*) — public repos need none.

Examples:
    ./gguf-to-ollama.py --list
    ./gguf-to-ollama.py --quant UD-Q4_K_M
    ./gguf-to-ollama.py --repo bartowski/Meta-Llama-3.1-8B-Instruct-GGUF --list
    ./gguf-to-ollama.py --repo bartowski/Meta-Llama-3.1-8B-Instruct-GGUF --quant Q6_K
    ./gguf-to-ollama.py --quant BF16 --gguf-split /opt/llama.cpp/build/bin/llama-gguf-split
    ./gguf-to-ollama.py --quant UD-Q6_K --no-mmproj --name qwen38:q6k
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

DEFAULT_REPO = "unsloth/Qwen3.8-27B-GGUF"
# Tried in order when --quant is not given; first one the repo actually has wins.
DEFAULT_QUANTS = ("UD-Q4_K_M", "Q4_K_M")
# Repo sub-directories that hold non-model GGUFs (e.g. Qwen's MTP draft heads).
SKIP_DIRS = ("MTP",)

# (regex on repo id, preset) — first match wins; see default_preset().
PRESET_RULES: list[tuple[str, str]] = [
    (r"embedding|rerank", "none"),
    (r"qwen3.*thinking", "qwen3-thinking"),
    (r"qwen3", "qwen3"),
]

# Modelfile sampling presets. "none" emits no sampling PARAMETERs (Ollama defaults apply).
PRESETS: dict[str, list[str]] = {
    "none": [],
    # Unsloth's recommended defaults for Qwen3 in non-thinking mode.
    "qwen3": [
        "PARAMETER temperature 0.7",
        "PARAMETER top_p 0.8",
        "PARAMETER top_k 20",
        "PARAMETER min_p 0.0",
        "PARAMETER repeat_penalty 1.05",
    ],
    # Unsloth's recommended defaults for Qwen3 in thinking mode.
    "qwen3-thinking": [
        "PARAMETER temperature 0.6",
        "PARAMETER top_p 0.95",
        "PARAMETER top_k 20",
        "PARAMETER min_p 0.0",
    ],
}
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


def repo_base(repo_id: str) -> str:
    """'unsloth/Qwen3.8-27B-GGUF' -> 'Qwen3.8-27B' (the filename prefix used in the repo)."""
    name = repo_id.rsplit("/", 1)[-1]
    if name.lower().endswith("-gguf"):
        name = name[: -len("-gguf")]
    return name


def default_preset(repo_id: str) -> str:
    rid = repo_id.lower()
    for pattern, preset in PRESET_RULES:
        if re.search(pattern, rid):
            return preset
    return "none"


def default_workdir(repo_id: str) -> Path:
    # Keep the org so unsloth/X-GGUF and Qwen/X-GGUF never share a directory.
    return Path.home() / "models" / Path(*repo_id.split("/"))


def is_mmproj(name: str) -> bool:
    return "mmproj" in name.lower()


def default_ollama_name(repo_id: str, quant: str) -> str:
    return f"{repo_base(repo_id).lower()}:{quant.lower()}"


def fetch_repo_ggufs(repo_id: str) -> tuple[list[str], dict[str, int]]:
    """Return (sorted .gguf paths in the repo, {path: size in bytes})."""
    hf = _import_hf()
    try:
        info = hf.HfApi().model_info(repo_id, files_metadata=True)
    except Exception as e:  # network, auth, 404 — all fatal here
        die(f"could not list files in {repo_id}: {e}")
    sizes = {s.rfilename: s.size or 0 for s in info.siblings if s.rfilename.endswith(".gguf")}
    return sorted(sizes), sizes


def list_repo_ggufs(repo_id: str) -> list[str]:
    return fetch_repo_ggufs(repo_id)[0]


def group_quants(files: list[str], base: str) -> dict[str, list[str]]:
    """Map quant label -> ordered list of repo file paths belonging to it.

    `base` is the model-name prefix (see repo_base); it is stripped case-insensitively
    from each filename to produce the label, e.g. "Qwen3.8-27B-UD-Q4_K_M" -> "UD-Q4_K_M".
    Files that do not carry the prefix keep their full stem as the label.
    """
    stems: list[tuple[str, str]] = []  # (stem, repo path)
    for path in files:
        name = Path(path).name
        if is_mmproj(name) or "imatrix" in name.lower():
            continue
        if any(path.startswith(d + "/") for d in SKIP_DIRS):
            continue
        m = SPLIT_RE.match(name)
        stems.append((m.group("stem") if m else name[: -len(".gguf")], path))

    # "<base>-Q4_K_M", "<base>.Q4_K_M" (TheBloke/mradermacher style) or "<base>_Q4_K_M"
    prefix_re = re.compile(rf"^{re.escape(base)}[-._]", re.IGNORECASE)
    if not any(prefix_re.match(stem) for stem, _ in stems):
        # Repo name is not the filename prefix (e.g. mradermacher/X-i1-GGUF ships
        # "X.i1-Q4_K_M.gguf"): fall back to the longest common filename prefix.
        prefix_re = re.compile(rf"^{re.escape(_common_prefix([st for st, _ in stems]))}")

    quants: dict[str, list[str]] = {}
    for stem, path in stems:
        label = prefix_re.sub("", stem, count=1) or stem
        quants.setdefault(label, []).append(path)

    for paths in quants.values():
        paths.sort()
    return quants


def _common_prefix(stems: list[str]) -> str:
    """Longest common prefix of stems, cut back to the last [-._] separator ('' if none)."""
    if len(stems) < 2:
        return ""
    p = os.path.commonprefix(stems)
    cut = max(p.rfind(c) for c in "-._")
    return p[: cut + 1] if cut >= 0 else ""


def find_mmproj(files: list[str]) -> str | None:
    """Pick the vision projector from the repo listing; prefer F16, else any mmproj."""
    mmprojs = sorted(f for f in files if is_mmproj(Path(f).name))
    for f in mmprojs:
        if re.search(r"(?<![a-z0-9])f16", Path(f).name.lower()):  # F16 but not BF16
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


def hf_download(repo_id: str, patterns: list[str], dest: Path) -> None:
    hf = _import_hf()
    from huggingface_hub.errors import GatedRepoError

    dest.mkdir(parents=True, exist_ok=True)
    print(f"==> downloading {patterns} from {repo_id} -> {dest}")
    try:
        hf.snapshot_download(
            repo_id=repo_id,
            allow_patterns=patterns,
            local_dir=str(dest),
            max_workers=hf_workers(),
        )
    except GatedRepoError as e:
        die(
            f"{repo_id} is gated: accept its license on huggingface.co/{repo_id}, then set "
            f"HF_TOKEN (or run `hf auth login`) and retry.\n{e}"
        )
    except Exception as e:
        die(f"download from {repo_id} failed: {e}")


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


def read_gguf_metadata(path: Path) -> dict | None:
    """Parse the metadata KV section of a GGUF file (header only, no tensor data).

    Returns None if the file is not a readable GGUF. Array values are skipped
    (returned as None) except arrays are not needed for anything we look up.
    """
    _SCALARS = {
        0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
        5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8),
        12: ("<d", 8),
    }
    try:
        with path.open("rb") as f:
            def take(n: int) -> bytes:
                b = f.read(n)
                if len(b) != n:
                    raise EOFError
                return b

            def scalar(t: int):
                fmt, n = _SCALARS[t]
                return struct.unpack(fmt, take(n))[0]

            def string() -> str:
                (n,) = struct.unpack("<Q", take(8))
                return take(n).decode("utf-8", "replace")

            def value(t: int):
                if t in _SCALARS:
                    return scalar(t)
                if t == 8:
                    return string()
                if t == 9:  # array: elem type, count, elems (parsed but unused)
                    et, n = struct.unpack("<IQ", take(12))
                    return [value(et) for _ in range(n)]
                raise ValueError(f"unknown GGUF value type {t}")

            if take(4) != b"GGUF":
                return None
            version, _tensors, n_kv = struct.unpack("<IQQ", take(20))
            if version < 2:  # v1 used 32-bit counts; nobody ships those any more
                return None
            meta: dict = {}
            for _ in range(n_kv):
                key = string()
                (t,) = struct.unpack("<I", take(4))
                meta[key] = value(t)
            return meta
    except (OSError, EOFError, ValueError, KeyError, struct.error):
        return None


def native_ctx(meta: dict) -> int | None:
    arch = meta.get("general.architecture")
    v = meta.get(f"{arch}.context_length") if arch else None
    return int(v) if isinstance(v, (int, float)) and v > 0 else None


def kv_bytes_per_token(meta: dict) -> int | None:
    """Approximate KV-cache bytes per token at f16 (Ollama's default cache type)."""
    arch = meta.get("general.architecture")
    if not arch:
        return None
    layers = meta.get(f"{arch}.block_count")
    kv_heads = meta.get(f"{arch}.attention.head_count_kv")
    head_dim = meta.get(f"{arch}.attention.key_length")
    if head_dim is None:
        emb = meta.get(f"{arch}.embedding_length")
        heads = meta.get(f"{arch}.attention.head_count")
        head_dim = emb // heads if emb and heads else None
    if not (layers and kv_heads and head_dim):
        return None
    return 2 * int(layers) * int(kv_heads) * int(head_dim) * 2  # K+V, f16


def auto_num_ctx(model: Path) -> int | None:
    """The model's native context length from its GGUF header; None if unreadable."""
    meta = read_gguf_metadata(model)
    if meta is None:
        return None
    native = native_ctx(meta)
    if native is None:
        return None
    note = ""
    kv = kv_bytes_per_token(meta)
    if kv is not None:
        note = f"; f16 KV cache at full context ~{native * kv / 2**30:.0f} GiB"
    print(f"==> num_ctx {native} (model's native context{note}; override with --num-ctx)")
    return native


def write_modelfile(
    model: Path, mmproj: Path | None, workdir: Path, *, preset: str, num_ctx: int | None
) -> Path:
    lines = [f"FROM {model.resolve()}"]
    if mmproj is not None:
        lines.append(f"FROM {mmproj.resolve()}")
    params = list(PRESETS[preset])
    if num_ctx is not None:
        params.append(f"PARAMETER num_ctx {num_ctx}")
    if params:
        lines += [""] + params
    lines.append("")
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
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"Hugging Face GGUF repo (default: {DEFAULT_REPO})")
    ap.add_argument("--quant", help=f"quant label (default: first of {', '.join(DEFAULT_QUANTS)} the repo has)")
    ap.add_argument("--list", action="store_true", help="list available quants and exit")
    ap.add_argument("--dir", type=Path, help="download directory (default: ~/models/<org>/<repo name>)")
    ap.add_argument("--name", help="Ollama model name (default: <model>:<quant>, lower-cased)")
    ap.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="Modelfile sampling preset (default: qwen3 for Qwen3 repos, else none)",
    )
    ap.add_argument(
        "--num-ctx",
        type=int,
        help="PARAMETER num_ctx (default: the model's native context length from its "
        "GGUF header; 0 to omit and use Ollama's default)",
    )
    ap.add_argument("--gguf-split", help="path to llama-gguf-split (for split builds)")
    ap.add_argument("--no-mmproj", action="store_true", help="skip the vision projector even if the repo has one")
    ap.add_argument("--download-only", action="store_true", help="download/merge but do not import")
    args = ap.parse_args()

    repo = args.repo
    base = repo_base(repo)
    workdir: Path = args.dir or default_workdir(repo)
    preset = args.preset or default_preset(repo)

    files, sizes = fetch_repo_ggufs(repo)
    quants = group_quants(files, base)
    if not quants:
        die(f"no model GGUFs found in {repo}")

    if args.list:
        for label, paths in sorted(quants.items()):
            tag = f"  ({len(paths)} shards, needs merge)" if len(paths) > 1 else ""
            print(f"{label}{tag}")
        return

    quant = args.quant
    if quant is None:
        quant = next((q for q in DEFAULT_QUANTS if q in quants), None)
        if quant is None:
            die(
                f"--quant not given and none of the defaults ({', '.join(DEFAULT_QUANTS)}) exist "
                f"in {repo}. Available: {', '.join(sorted(quants))}"
            )
        print(f"==> no --quant given; using {quant}")
    if quant not in quants:
        die(f"unknown quant {quant!r}. Use --list to see options.")

    paths = quants[quant]
    patterns: list[str] = []

    # Skip re-downloading shards when a previous run already produced the merged file
    # (the shards may have been deleted to reclaim space).
    merged = merged_path_for(paths, workdir)
    if merged is not None and merged.exists():
        print(f"==> merged file already exists, skipping shard download: {merged}")
    else:
        patterns += paths

    mmproj_repo = None
    if not args.no_mmproj:
        mmproj_repo = find_mmproj(files)
        if mmproj_repo is None:
            print(f"==> no mmproj in {repo}; importing text-only")
        else:
            patterns.append(mmproj_repo)

    if patterns:
        # Check disk up front: bytes still to download, plus the merged copy if sharded.
        need = sum(sizes.get(p, 0) for p in patterns if not (workdir / p).exists())
        if merged is not None and not merged.exists():
            need += sum(sizes.get(p, 0) for p in paths)
        check_free_space(need, workdir, "download and merge" if merged else "download")
        hf_download(repo, patterns, workdir)

    model = resolve_model_file(paths, workdir, args.gguf_split)
    mmproj = None
    if mmproj_repo is not None:
        mmproj = workdir / mmproj_repo
        if not mmproj.exists():
            die(f"expected {mmproj} after download but it is missing")

    if args.num_ctx is None:
        num_ctx = auto_num_ctx(model) or 8192
    else:
        num_ctx = args.num_ctx or None  # 0 -> omit the PARAMETER entirely
    print(f"==> sampling preset: {preset}" + ("" if args.preset else " (auto; override with --preset)"))
    modelfile = write_modelfile(model, mmproj, workdir, preset=preset, num_ctx=num_ctx)
    if args.download_only:
        print(f"Skipping import. Later: ollama create <name> -f {modelfile}")
        return

    ollama_create(args.name or default_ollama_name(repo, quant), modelfile)


if __name__ == "__main__":
    main()
