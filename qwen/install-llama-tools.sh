#!/usr/bin/env bash
# Download prebuilt llama.cpp binaries (for llama-gguf-split) into ./tools.
# Only needed to merge split GGUF builds such as BF16.
set -euo pipefail

REPO="ggml-org/llama.cpp"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tools"

case "$(uname -m)" in
    aarch64 | arm64) ASSET_ARCH="arm64" ;;
    x86_64 | amd64)  ASSET_ARCH="x64" ;;
    *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "this installer only handles Linux; build llama.cpp manually" >&2
    exit 1
fi

echo "==> finding latest llama.cpp release with an ubuntu-${ASSET_ARCH} build"
read -r TAG URL < <(
    curl -sfL "https://api.github.com/repos/${REPO}/releases?per_page=10" |
    python3 -c "
import json, sys
arch = '${ASSET_ARCH}'
want = f'bin-ubuntu-{arch}.tar.gz'
for rel in json.load(sys.stdin):
    for a in rel['assets']:
        if a['name'].endswith(want):
            print(rel['tag_name'], a['browser_download_url'])
            sys.exit(0)
sys.exit('no matching release asset found')
"
)

echo "==> downloading ${TAG}"
mkdir -p "$DEST"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fL --progress-bar -o "$tmp/llama.tar.gz" "$URL"
tar xzf "$tmp/llama.tar.gz" -C "$DEST"

BIN="$(find "$DEST" -name llama-gguf-split -type f | sort -r | head -1)"
[[ -n "$BIN" ]] || { echo "llama-gguf-split missing from archive" >&2; exit 1; }
chmod +x "$BIN"

echo "==> installed: $BIN"
"$BIN" --help >/dev/null 2>&1 && echo "==> verified working"
echo "qwen38-27b-ollama.py will now find it automatically."
