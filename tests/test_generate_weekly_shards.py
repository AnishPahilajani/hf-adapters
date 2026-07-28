# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for .github/scripts/generate_weekly_shards.py.

The property under test is the split between testable and rejected shards.
Rejected rows spawn no workers, so packing them at the testable shard sizes
would schedule dozens of Spyre-pod jobs to do nothing but write to ClickHouse.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "generate_weekly_shards.py"
)


def _load_module():
    """Import the script by path; it lives outside any importable package."""
    spec = importlib.util.spec_from_file_location("generate_weekly_shards", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def shards_module():
    return _load_module()


def make_rows(n_testable: int, n_rejected: int, n_curated: int = 0) -> list[dict]:
    rows: list[dict] = [
        {
            "model_id": f"org/ok{i}",
            "rejection_reason": None,
            "curated": i < n_curated,
            "model_info": object(),  # must be stripped before serialization
        }
        for i in range(n_testable)
    ]
    rows += [
        {
            "model_id": f"org/bad{i}",
            "rejection_reason": "not_an_embedder",
            "curated": False,
            "model_info": object(),
        }
        for i in range(n_rejected)
    ]
    return rows


def _patch_fetchers(module, monkeypatch, generative, embedding):
    monkeypatch.setattr(module, "fetch_top_generative_models", lambda limit: generative)
    monkeypatch.setattr(module, "fetch_top_embedding_models", lambda limit: embedding)


def test_chunk_splits_and_keeps_remainder(shards_module):
    assert shards_module._chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert shards_module._chunk([], 10) == []


def test_testable_and_rejected_are_sharded_separately(
    shards_module, monkeypatch, tmp_path
):
    _patch_fetchers(
        shards_module,
        monkeypatch,
        generative=make_rows(5, 7),
        embedding=make_rows(3, 4),
    )

    matrix = shards_module.generate_shards(
        top_k=100,
        shard_size_generative=2,
        shard_size_embedding=2,
        output_dir=tmp_path,
        shard_size_rejected=100,
    )

    kinds = {(e["mode"], e["kind"]) for e in matrix}
    assert kinds == {
        ("generative", "testable"),
        ("generative", "rejected"),
        ("embedding", "testable"),
        ("embedding", "rejected"),
    }

    # 5 testable at size 2 -> 3 shards; 7 rejected at size 100 -> 1 shard.
    gen_testable = [
        e for e in matrix if e["mode"] == "generative" and e["kind"] == "testable"
    ]
    gen_rejected = [
        e for e in matrix if e["mode"] == "generative" and e["kind"] == "rejected"
    ]
    assert len(gen_testable) == 3
    assert len(gen_rejected) == 1


def test_large_rejected_shard_size_collapses_job_count(
    shards_module, monkeypatch, tmp_path
):
    """The whole point: rejected rows must not multiply pod jobs.

    6000 rejected rows at the testable size (500) would be 12 extra matrix
    entries; at 5000 it is 2.
    """
    _patch_fetchers(
        shards_module,
        monkeypatch,
        generative=[],
        embedding=make_rows(0, 6000),
    )

    matrix = shards_module.generate_shards(
        top_k=100,
        shard_size_generative=250,
        shard_size_embedding=500,
        output_dir=tmp_path,
        shard_size_rejected=5000,
    )
    assert len([e for e in matrix if e["kind"] == "rejected"]) == 2


def test_shard_files_are_written_and_model_info_stripped(
    shards_module, monkeypatch, tmp_path
):
    _patch_fetchers(
        shards_module, monkeypatch, generative=make_rows(2, 1), embedding=[]
    )

    matrix = shards_module.generate_shards(
        top_k=100,
        shard_size_generative=10,
        shard_size_embedding=10,
        output_dir=tmp_path,
    )

    for entry in matrix:
        path = tmp_path / entry["shard_file"]
        assert path.exists(), f"{entry['shard_file']} not written"
        rows = json.loads(path.read_text())
        assert rows, "empty shard written"
        for row in rows:
            assert "model_info" not in row, "non-serializable model_info survived"


def test_shard_filenames_encode_mode_and_kind(shards_module, monkeypatch, tmp_path):
    _patch_fetchers(
        shards_module, monkeypatch, generative=make_rows(1, 1), embedding=[]
    )
    matrix = shards_module.generate_shards(
        top_k=10,
        shard_size_generative=10,
        shard_size_embedding=10,
        output_dir=tmp_path,
    )
    names = sorted(e["shard_file"] for e in matrix)
    assert names == [
        "generative-rejected-shard-000.json",
        "generative-testable-shard-000.json",
    ]


def test_no_shard_emitted_for_an_empty_kind(shards_module, monkeypatch, tmp_path):
    """A run with zero rejections must not create an empty rejected shard/job."""
    _patch_fetchers(
        shards_module, monkeypatch, generative=make_rows(2, 0), embedding=[]
    )
    matrix = shards_module.generate_shards(
        top_k=10,
        shard_size_generative=10,
        shard_size_embedding=10,
        output_dir=tmp_path,
    )
    assert all(e["kind"] == "testable" for e in matrix)
    assert len(matrix) == 1


def test_matrix_is_json_serializable_for_github_output(
    shards_module, monkeypatch, tmp_path
):
    """generate_shards' return value is emitted via GITHUB_OUTPUT as JSON."""
    _patch_fetchers(
        shards_module,
        monkeypatch,
        generative=make_rows(1, 1),
        embedding=make_rows(1, 0),
    )
    matrix = shards_module.generate_shards(
        top_k=10,
        shard_size_generative=10,
        shard_size_embedding=10,
        output_dir=tmp_path,
    )
    decoded = json.loads(json.dumps(matrix))
    assert all({"mode", "kind", "shard_index", "shard_file"} <= set(e) for e in decoded)
