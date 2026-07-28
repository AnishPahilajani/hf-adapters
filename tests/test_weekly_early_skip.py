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

"""Tests for weekly_test.classify_early_skip.

This function decides which rows are settled without spawning a worker, and in
what order the causes are considered. The ordering is the whole point: a row can
match several branches at once, and picking the wrong one writes an incorrect
failure_category that the sink then treats as terminal for 10 days.

Imported from tests/spyre/weekly_generation/, which pytest excludes by default —
but weekly_test.py itself has no module-level torch_spyre dependency (its Spyre
imports are inside function bodies), so this file runs on a plain CPU host.
"""

import pytest

from tests.spyre.weekly_generation.weekly_test import (
    FAILURE_CATEGORY_MODEL_TOO_LARGE,
    FAILURE_CATEGORY_MOE,
    FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER,
    MAX_NUMBER_PARAMS,
    classify_early_skip,
)


def row(**overrides) -> dict:
    """A shard row that would otherwise be evaluated normally."""
    base = {
        "model_id": "org/model",
        "downloads": 1000,
        "model_type": "llama",
        "architectures": "LlamaForCausalLM",
        "parameters": 7_000_000_000,
        "config_class": "LlamaConfig",
        "is_supported": True,
        "is_moe": False,
        "rejection_reason": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Pass-through
# ---------------------------------------------------------------------------


def test_normal_model_is_evaluated():
    assert classify_early_skip(row()) == (None, "")


# ---------------------------------------------------------------------------
# Each cause in isolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"rejection_reason": "not_an_embedder"}, "not_an_embedder"),
        ({"rejection_reason": "reranker"}, "reranker"),
        ({"rejection_reason": "no_loadable_weights"}, "no_loadable_weights"),
        ({"rejection_reason": "curated_fetch_failed"}, "curated_fetch_failed"),
        (
            {"is_supported": False, "config_class": "OPTConfig"},
            FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER,
        ),
        ({"parameters": MAX_NUMBER_PARAMS + 1}, FAILURE_CATEGORY_MODEL_TOO_LARGE),
        ({"is_moe": True}, FAILURE_CATEGORY_MOE),
    ],
)
def test_each_cause_is_reported(overrides, expected):
    category, suffix = classify_early_skip(row(**overrides))
    assert category == expected
    assert suffix, "a skipped row must carry a human-readable explanation"


def test_fetch_rejection_reason_is_passed_through_verbatim():
    """The reason string goes straight into failure_category, so it must not be
    rewritten or namespaced on the way through.
    """
    category, _ = classify_early_skip(row(rejection_reason="remote_code"))
    assert category == "remote_code"


# ---------------------------------------------------------------------------
# Ordering — the reason this lives in one function
# ---------------------------------------------------------------------------


def test_rejection_reason_wins_over_unsupported():
    """A rejected row's config class was never resolved, so the generic
    no-adapter branch would otherwise claim it and hide the real cause.
    """
    category, _ = classify_early_skip(
        row(
            rejection_reason="not_an_embedder",
            is_supported=False,
            config_class="BertConfig",
        )
    )
    assert category == "not_an_embedder"


def test_rejection_reason_wins_over_moe_and_too_large():
    category, _ = classify_early_skip(
        row(rejection_reason="nsfw", is_moe=True, parameters=MAX_NUMBER_PARAMS + 1)
    )
    assert category == "nsfw"


def test_unsupported_wins_over_too_large():
    """Cheapest terminal verdict first; both are equally terminal."""
    category, _ = classify_early_skip(
        row(
            is_supported=False,
            config_class="OPTConfig",
            parameters=MAX_NUMBER_PARAMS + 1,
        )
    )
    assert category == FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER


# ---------------------------------------------------------------------------
# The gated / config-unreadable pass-through
# ---------------------------------------------------------------------------


def test_undetermined_support_reaches_the_worker():
    """is_supported=None means "not determined" — typically a gated repo whose
    config AutoConfig could not read in the shard-building environment. The pod
    has the token and licence acceptances, so it must get the chance to try.
    """
    assert classify_early_skip(row(is_supported=None, config_class=None)) == (None, "")


def test_stale_shard_gated_row_still_reaches_the_worker():
    """Backward compatibility: a shard written before is_supported became
    tri-state reports False with no config_class for a gated model. That must
    not be recorded as not-implemented-adapter either.
    """
    assert classify_early_skip(row(is_supported=False, config_class=None)) == (None, "")


def test_unsupported_requires_a_resolved_config_class():
    assert classify_early_skip(row(is_supported=False, config_class=""))[0] is None
    assert (
        classify_early_skip(row(is_supported=False, config_class="OPTConfig"))[0]
        == FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER
    )


# ---------------------------------------------------------------------------
# Edge cases in the size check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("params", [None, ""])
def test_unknown_parameter_count_is_not_too_large(params):
    """Unknown size is not a rejection — the in-worker guard is the backstop."""
    assert classify_early_skip(row(parameters=params)) == (None, "")


def test_model_exactly_at_the_limit_is_allowed():
    assert classify_early_skip(row(parameters=MAX_NUMBER_PARAMS)) == (None, "")
