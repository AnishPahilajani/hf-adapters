# Copyright 2026 The Torch-Spyre Authors.
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

"""Targeted regression tests for ppc64le model exclusions.

Tests:
1. Schema validation for tests/model_lists/exclude.yaml in manage_cache.py.
2. Cache download list under simulated ppc64le vs x86_64, including extra_models.
3. --model-path overrides (mixed, all-excluded, and non-excluded) in conftest.
4. Perf-list validation on both x86_64 and ppc64le architectures.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_CONFIG_DIR = REPO_ROOT / ".github" / "cache_config"
if str(CACHE_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CACHE_CONFIG_DIR))

import manage_cache  # noqa: E402


def test_manage_cache_rejects_non_mapping_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """exclude.yaml must be a YAML mapping."""
    bad_file = tmp_path / "exclude.yaml"
    bad_file.write_text("- not-a-dict\n")
    monkeypatch.setattr(manage_cache, "EXCLUDE_FILE_PATH", bad_file)

    with pytest.raises(ValueError, match="did not parse to a mapping"):
        manage_cache._load_excluded_paths()


@pytest.mark.parametrize("key", ["models", "ppc64le_models"])
@pytest.mark.parametrize(
    "invalid_val",
    [
        "just-a-string",
        123,
        {"nested": "dict"},
        ["valid-model", 456],
    ],
)
def test_manage_cache_validates_section_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, invalid_val: object
) -> None:
    """Each exclusion section must be a list of strings."""
    bad_file = tmp_path / "exclude.yaml"
    bad_file.write_text(yaml.dump({key: invalid_val}))
    monkeypatch.setattr(manage_cache, "EXCLUDE_FILE_PATH", bad_file)

    with pytest.raises(ValueError, match=f"'{key}' section must be a list of strings"):
        manage_cache._load_excluded_paths()


def test_manage_cache_ppc64le_excludes_registry_and_extra_models() -> None:
    """Under simulated ppc64le, Qwen models from registry and extra_models are excluded."""
    extra = ["Qwen/Qwen3-0.6B", "custom-org/valid-extra-model"]

    with patch("platform.machine", return_value="ppc64le"):
        excluded = manage_cache._load_excluded_paths()
        models = sorted((set(manage_cache._registry_model_paths(excluded=excluded)) | set(extra)) - excluded)

        assert not any("Qwen" in m or "qwen" in m for m in models)
        assert "custom-org/valid-extra-model" in models
        assert "Qwen/Qwen3-0.6B" not in models


def test_manage_cache_x86_includes_qwen_models() -> None:
    """Under simulated x86_64, Qwen models remain present."""
    with patch("platform.machine", return_value="x86_64"):
        excluded = manage_cache._load_excluded_paths()
        models = manage_cache._registry_model_paths(excluded=excluded)

        assert any("Qwen" in m or "qwen" in m for m in models)


def test_pytest_generate_tests_mixed_overrides_ppc64le() -> None:
    """When some models are excluded on ppc64le, only the valid ones are parametrized."""
    import tests.conftest as conftest_mod

    metafunc = MagicMock()
    metafunc.config.getoption.return_value = ["Qwen/Qwen3-0.6B", "ibm-granite/granite-4.1-8b"]
    metafunc.fixturenames = ["model_path"]
    metafunc.definition.iter_markers.return_value = []
    metafunc.definition.own_markers = []
    metafunc.definition.get_closest_marker.return_value = MagicMock(args=["causal"])

    with patch("platform.machine", return_value="ppc64le"):
        import tests.model_registry as mr
        importlib.reload(mr)
        conftest_mod.pytest_generate_tests(metafunc)

        call_args = metafunc.parametrize.call_args
        assert call_args is not None
        assert call_args[1]["ids"] == ["ibm-granite/granite-4.1-8b"]


def test_pytest_generate_tests_all_excluded_overrides_ppc64le() -> None:
    """When ALL overrides are excluded on ppc64le, tests are parametrized as skipped."""
    import tests.conftest as conftest_mod

    metafunc = MagicMock()
    metafunc.config.getoption.return_value = ["Qwen/Qwen3-0.6B"]
    metafunc.fixturenames = ["model_path"]
    metafunc.definition.iter_markers.return_value = []
    metafunc.definition.own_markers = []
    metafunc.definition.get_closest_marker.return_value = MagicMock(args=["causal"])

    with patch("platform.machine", return_value="ppc64le"):
        import tests.model_registry as mr
        importlib.reload(mr)
        conftest_mod.pytest_generate_tests(metafunc)

        call_args = metafunc.parametrize.call_args
        assert call_args is not None
        assert call_args[1]["ids"] == ["Qwen/Qwen3-0.6B"]
        # The single param should have a skip marker
        param_obj = call_args[0][1][0]
        assert hasattr(param_obj, "marks")
        assert any(m.name == "skip" for m in param_obj.marks)


def test_perf_list_validation_passes_on_both_architectures() -> None:
    """test_every_perf_list_file_is_benchmarkable must pass on both x86_64 and ppc64le."""
    from tests.test_perf_matrix_is_causal_only import (
        test_every_perf_list_file_is_benchmarkable,
    )

    for arch in ("x86_64", "ppc64le"):
        with patch("platform.machine", return_value=arch):
            import tests.model_registry as mr
            importlib.reload(mr)
            test_every_perf_list_file_is_benchmarkable()
