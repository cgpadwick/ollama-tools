#!/usr/bin/env bash
# Download prebuilt llama.cpp binaries (for llama-gguf-split) into ./tools.
# Only needed to merge split GGUF builds such as BF16.
# Set GITHUB_TOKEN to raise the unauthenticated GitHub API rate limit (60 req/hr).
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

AUTH=()
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    AUTH=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
fi

echo "==> finding latest llama.cpp release with an ubuntu-${ASSET_ARCH} build"
read -r TAG URL < <(
    curl -sfL "${AUTH[@]}" "https://api.github.com/repos/${REPO}/releases?per_page=10" |
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
) || { echo "could not determine latest release (rate limited? offline?)" >&2; exit 1; }

# Release archives extract to a single top-level dir "llama-<tag>/", e.g. llama-b10599/.
BUILD_DIR="$DEST/llama-$TAG"
if [[ -x "$BUILD_DIR/llama-gguf-split" ]]; then
    echo "==> $TAG already installed at $BUILD_DIR; nothing to do"
    exit 0
fi

echo "==> downloading ${TAG}"
mkdir -p "$DEST"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fL --progress-bar -o "$tmp/llama.tar.gz" "$URL"
tar xzf "$tmp/llama.tar.gz" -C "$DEST"

BIN="$BUILD_DIR/llama-gguf-split"
[[ -f "$BIN" ]] || { echo "llama-gguf-split missing from archive (expected $BIN)" >&2; exit 1; }
chmod +x "$BIN"

# Remove older vendored builds so only the one we just verified remains.
find "$DEST" -mindepth 1 -maxdepth 1 -type d -name 'llama-b*' ! -name "llama-$TAG" -exec rm -rf {} +

echo "==> installed: $BIN"
if ! "$BIN" --help >/dev/null 2>&1; then
    echo "error: $BIN does not run (missing shared libraries?). Output:" >&2
    "$BIN" --help 2>&1 | head -5 >&2 || true
    exit 1
fi
echo "==> verified working"
echo "qwen38-27b-ollama.py will now find it automatically."
