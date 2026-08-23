import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "qwen38-27b-ollama.py"
_spec = importlib.util.spec_from_file_location("qwen_tool", _SCRIPT)
tool = importlib.util.module_from_spec(_spec)
sys.modules["qwen_tool"] = tool
_spec.loader.exec_module(tool)


def test_group_quants_groups_shards_and_skips_extras():
    files = [
        "BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf",
        "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf",
        "Qwen3.8-27B-UD-Q4_K_M.gguf",
        "mmproj-F16.gguf",
        "imatrix_unsloth.gguf",
        "MTP/Qwen3.8-27B-MTP.gguf",
    ]
    quants = tool.group_quants(files)
    assert quants == {
        "BF16": [
            "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf",
            "BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf",
        ],
        "UD-Q4_K_M": ["Qwen3.8-27B-UD-Q4_K_M.gguf"],
    }


def test_find_mmproj_prefers_f16():
    files = ["mmproj-BF16.gguf", "mmproj-F16.gguf", "Qwen3.8-27B-Q8_0.gguf"]
    assert tool.find_mmproj(files) == "mmproj-F16.gguf"


def test_find_mmproj_falls_back_to_any_mmproj():
    assert tool.find_mmproj(["mmproj-BF16.gguf", "x.gguf"]) == "mmproj-BF16.gguf"
    assert tool.find_mmproj(["x.gguf"]) is None


def test_vendored_tools_sorted_numerically(tmp_path):
    for tag in ("llama-b9999", "llama-b10599", "llama-b123"):
        d = tmp_path / "tools" / tag
        d.mkdir(parents=True)
        (d / "llama-gguf-split").write_text("")
    found = tool.vendored_gguf_splits(tmp_path)
    assert [p.parent.name for p in found] == ["llama-b10599", "llama-b9999", "llama-b123"]


def test_merged_path_and_shard_count(tmp_path):
    paths = ["BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf"]
    with pytest.raises(SystemExit):
        tool.merged_path_for(paths, tmp_path)  # declares 2 shards, only 1 listed


def test_merged_path_ok(tmp_path):
    paths = [
        "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf",
        "BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf",
    ]
    assert tool.merged_path_for(paths, tmp_path) == tmp_path / "BF16" / "Qwen3.8-27B-BF16.gguf"
    assert tool.merged_path_for(["Qwen3.8-27B-Q8_0.gguf"], tmp_path) is None


def test_merge_shards_leaves_no_partial_on_failure(tmp_path):
    first = tmp_path / "m-00001-of-00002.gguf"
    first.write_bytes(b"x")
    (tmp_path / "m-00002-of-00002.gguf").write_bytes(b"x")
    out = tmp_path / "m.gguf"
    bad_tool = tmp_path / "fake-split"
    bad_tool.write_text("#!/bin/sh\necho partial > \"$3\"\nexit 1\n")
    bad_tool.chmod(0o755)
    with pytest.raises(SystemExit):
        tool.merge_shards(first, out, str(bad_tool))
    assert not out.exists()
    assert not out.with_name(out.name + ".part").exists()


def test_merge_shards_renames_on_success(tmp_path):
    first = tmp_path / "m-00001-of-00002.gguf"
    first.write_bytes(b"x")
    (tmp_path / "m-00002-of-00002.gguf").write_bytes(b"x")
    out = tmp_path / "m.gguf"
    good_tool = tmp_path / "fake-split"
    good_tool.write_text("#!/bin/sh\necho merged > \"$3\"\n")
    good_tool.chmod(0o755)
    assert tool.merge_shards(first, out, str(good_tool)) == out
    assert out.read_text().strip() == "merged"
    assert not out.with_name(out.name + ".part").exists()


def test_find_gguf_split_dies_on_bad_explicit_path(tmp_path):
    with pytest.raises(SystemExit):
        tool.find_gguf_split(str(tmp_path / "nope"))
