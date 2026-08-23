import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "gguf-to-ollama.py"
_spec = importlib.util.spec_from_file_location("gguf_tool", _SCRIPT)
tool = importlib.util.module_from_spec(_spec)
sys.modules["gguf_tool"] = tool
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
    quants = tool.group_quants(files, "Qwen3.8-27B")
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
    # A lone "-00001-of-00001" file needs no merge.
    assert tool.merged_path_for(["Qwen3.8-27B-Q8_0-00001-of-00001.gguf"], tmp_path) is None


def _fake_merge(tmp_path, script_body):
    """Two dummy shards, an output path, and a fake llama-gguf-split running script_body."""
    shards = [tmp_path / f"m-0000{i}-of-00002.gguf" for i in (1, 2)]
    for p in shards:
        p.write_bytes(b"x")
    out = tmp_path / "m.gguf"
    fake = tmp_path / "fake-split"
    fake.write_text("#!/bin/sh\n" + script_body)
    fake.chmod(0o755)
    return shards, out, str(fake)


def test_merge_shards_leaves_no_partial_on_failure(tmp_path):
    shards, out, fake = _fake_merge(tmp_path, 'echo partial > "$3"\nexit 1\n')
    with pytest.raises(SystemExit):
        tool.merge_shards(shards, out, fake)
    assert not out.exists()
    assert not out.with_name(out.name + ".part").exists()


def test_merge_shards_renames_on_success(tmp_path):
    shards, out, fake = _fake_merge(tmp_path, 'echo merged > "$3"\n')
    assert tool.merge_shards(shards, out, fake) == out
    assert out.read_text().strip() == "merged"
    assert not out.with_name(out.name + ".part").exists()


def test_merge_shards_clears_stale_part(tmp_path):
    # llama-gguf-split refuses to overwrite an existing output; emulate that.
    shards, out, fake = _fake_merge(
        tmp_path, 'if [ -e "$3" ]; then echo exists >&2; exit 1; fi\necho merged > "$3"\n'
    )
    out.with_name(out.name + ".part").write_text("stale")
    assert tool.merge_shards(shards, out, fake) == out
    assert out.read_text().strip() == "merged"


def test_merge_shards_non_executable_tool_dies_cleanly(tmp_path):
    shards, out, fake = _fake_merge(tmp_path, "")
    Path(fake).chmod(0o644)
    with pytest.raises(SystemExit):
        tool.merge_shards(shards, out, fake)
    assert not out.with_name(out.name + ".part").exists()


def test_merge_shards_missing_shard_dies(tmp_path):
    shards, out, fake = _fake_merge(tmp_path, "")
    shards[1].unlink()
    with pytest.raises(SystemExit):
        tool.merge_shards(shards, out, fake)


def test_find_gguf_split_dies_on_bad_explicit_path(tmp_path):
    with pytest.raises(SystemExit):
        tool.find_gguf_split(str(tmp_path / "nope"))


def test_repo_base_strips_gguf_suffix():
    assert tool.repo_base("unsloth/Qwen3.8-27B-GGUF") == "Qwen3.8-27B"
    assert tool.repo_base("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF") == "Meta-Llama-3.1-8B-Instruct"
    assert tool.repo_base("someone/model-gguf") == "model"
    assert tool.repo_base("someone/plain") == "plain"


def test_group_quants_without_repo_prefix_uses_common_prefix():
    files = ["Other-Model-Q4_K_M.gguf", "Other-Model-Q8_0.gguf"]
    assert tool.group_quants(files, "Qwen3.8-27B") == {
        "Q4_K_M": ["Other-Model-Q4_K_M.gguf"],
        "Q8_0": ["Other-Model-Q8_0.gguf"],
    }


def test_group_quants_case_insensitive_prefix():
    files = ["meta-llama-3.1-8b-instruct-Q4_K_M.gguf"]
    assert tool.group_quants(files, "Meta-Llama-3.1-8B-Instruct") == {
        "Q4_K_M": ["meta-llama-3.1-8b-instruct-Q4_K_M.gguf"]
    }


def test_default_preset_detection():
    assert tool.default_preset("unsloth/Qwen3.8-27B-GGUF") == "qwen3"
    assert tool.default_preset("unsloth/Qwen3-8B-GGUF") == "qwen3"
    assert tool.default_preset("unsloth/Qwen3-235B-A22B-Thinking-2507-GGUF") == "qwen3-thinking"
    assert tool.default_preset("unsloth/Qwen3-Embedding-8B-GGUF") == "none"
    assert tool.default_preset("unsloth/gemma-3-27b-it-GGUF") == "none"


def test_default_workdir_keeps_org():
    assert tool.default_workdir("unsloth/Qwen3-8B-GGUF") != tool.default_workdir("Qwen/Qwen3-8B-GGUF")
    assert tool.default_workdir("unsloth/Qwen3-8B-GGUF").parts[-2:] == ("unsloth", "Qwen3-8B-GGUF")


def test_group_quants_dot_separator():
    files = ["Qwen3-8B.Q4_K_M.gguf", "Qwen3-8B.i1-Q6_K.gguf", "Qwen3-8B-Q8_0.gguf"]
    assert tool.group_quants(files, "Qwen3-8B") == {
        "Q4_K_M": ["Qwen3-8B.Q4_K_M.gguf"],
        "i1-Q6_K": ["Qwen3-8B.i1-Q6_K.gguf"],
        "Q8_0": ["Qwen3-8B-Q8_0.gguf"],
    }


def test_group_quants_falls_back_to_common_prefix():
    files = ["Qwen3-8B.i1-IQ1_M.gguf", "Qwen3-8B.i1-Q4_K_M.gguf", "Qwen3-8B.i1-Q6_K.gguf"]
    assert tool.group_quants(files, "Qwen3-8B-i1") == {
        "IQ1_M": ["Qwen3-8B.i1-IQ1_M.gguf"],
        "Q4_K_M": ["Qwen3-8B.i1-Q4_K_M.gguf"],
        "Q6_K": ["Qwen3-8B.i1-Q6_K.gguf"],
    }


def test_group_quants_single_unrelated_file_keeps_stem():
    assert tool.group_quants(["weird.gguf"], "Other") == {"weird": ["weird.gguf"]}


def test_mmproj_detected_anywhere_in_name():
    files = ["gemma-3-4b-it-mmproj-f16.gguf", "gemma-3-4b-it-Q4_K_M.gguf"]
    assert tool.group_quants(files, "gemma-3-4b-it") == {"Q4_K_M": ["gemma-3-4b-it-Q4_K_M.gguf"]}
    assert tool.find_mmproj(files) == "gemma-3-4b-it-mmproj-f16.gguf"


def test_default_ollama_name():
    assert tool.default_ollama_name("unsloth/Qwen3.8-27B-GGUF", "UD-Q4_K_M") == "qwen3.8-27b:ud-q4_k_m"
    assert tool.default_ollama_name("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Q8_0") == "meta-llama-3.1-8b-instruct:q8_0"


def test_modelfile_contents(tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"")
    path = tool.write_modelfile(model, None, tmp_path, preset="qwen3", num_ctx=4096)
    text = path.read_text()
    assert text.startswith(f"FROM {model.resolve()}\n")
    assert "PARAMETER temperature 0.7" in text
    assert "PARAMETER num_ctx 4096" in text
    path = tool.write_modelfile(model, None, tmp_path, preset="none", num_ctx=None)
    assert "PARAMETER" not in path.read_text()
