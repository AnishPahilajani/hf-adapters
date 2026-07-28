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

"""Tests for the model-catalog fetch/classify pipeline.

Covers the three invariants the weekly scan's correctness rests on:

* ``diagnose`` short-circuits on the first failing gate, so a candidate
  rejected by a cheap metadata check never pays for a network call.
* Gated repos are NOT rejected — they are passed through for the Spyre pod,
  which holds the token and licence acceptances, to judge.
* ``build_catalog`` records rejected candidates as rows instead of dropping
  them, without spending config-class downloads on them.

Everything here runs offline against fake ModelInfo objects; no HF API calls.
"""

from datetime import datetime
from types import SimpleNamespace

import pytest

from utils import hf_model_catalog as cat
from utils.hf_model_catalog import (
    REASON_CURATED_FETCH_FAILED,
    REASON_NO_CONFIG,
    REASON_NO_LOADABLE_WEIGHTS,
    REASON_NON_NATIVE_FORMAT,
    REASON_NOT_AN_EMBEDDER,
    REASON_REMOTE_CODE,
    build_catalog,
    diagnose,
    load_curated_model_ids,
    merge_curated,
)


def make_model(
    model_id: str = "org/model",
    *,
    model_type: str = "llama",
    architectures: list[str] | None = None,
    tags: list[str] | None = None,
    library_name: str = "transformers",
    gated: bool | str = False,
    downloads: int = 1000,
    likes: int = 10,
    params: int | None = 7_000_000_000,
    has_config: bool = True,
) -> SimpleNamespace:
    """Build a ModelInfo-shaped stand-in carrying only the fields we read."""
    return SimpleNamespace(
        id=model_id,
        config=(
            {
                "model_type": model_type,
                "architectures": architectures or ["LlamaForCausalLM"],
            }
            if has_config
            else None
        ),
        tags=tags or [],
        library_name=library_name,
        gated=gated,
        downloads=downloads,
        likes=likes,
        safetensors=(
            SimpleNamespace(parameters={"F16": params}) if params is not None else None
        ),
        created_at=datetime(2025, 1, 1),
    )


# ---------------------------------------------------------------------------
# diagnose()
# ---------------------------------------------------------------------------


def test_diagnose_returns_none_when_all_gates_pass():
    gates = (
        ("cheap", lambda m, t: True),
        ("expensive", lambda m, t: True),
    )
    assert diagnose(make_model(), gates, False) is None


def test_diagnose_short_circuits_and_never_calls_later_gates():
    """The cost guarantee: a cheap rejection must not pay for a network call.

    If this breaks, ~20k candidates per run each gain an AutoConfig download
    plus a list_repo_files call, and the fetch job stops fitting in its timeout.
    """
    calls: list[str] = []

    def spy(name: str, verdict: bool):
        def _predicate(model, token):
            calls.append(name)
            return verdict

        return _predicate

    gates = (
        ("first_pass", spy("first_pass", True)),
        ("cheap_fail", spy("cheap_fail", False)),
        ("expensive", spy("expensive", True)),
    )
    assert diagnose(make_model(), gates, False) == "cheap_fail"
    assert calls == ["first_pass", "cheap_fail"], "gates after the failure ran"


def test_diagnose_attributes_a_raising_gate_to_that_gate():
    """A raising gate is a rejection *named after that gate*, not a silent drop.

    This is what made the HF_HOME PermissionError incident invisible: every
    AutoConfig call raised and every model vanished. Now it would surface as
    thousands of identically-blamed rows.
    """

    def boom(model, token):
        raise OSError("HF_HOME is not writable")

    gates = (("remote_code", boom),)
    assert diagnose(make_model(), gates, False) == "remote_code"


def test_diagnose_passes_token_through_to_predicates():
    seen: list[object] = []
    gates = (("gate", lambda m, t: seen.append(t) or True),)
    diagnose(make_model(), gates, "tok-123")
    assert seen == ["tok-123"]


# ---------------------------------------------------------------------------
# Gate predicates
# ---------------------------------------------------------------------------


def test_has_config_and_native_format_and_nsfw():
    assert cat.has_config(make_model()) is True
    assert cat.has_config(make_model(has_config=False)) is False

    assert cat.is_native_format(make_model("org/model")) is True
    assert cat.is_native_format(make_model("org/model-GGUF")) is False
    assert cat.is_native_format(make_model("org/model-onnx")) is False
    assert cat.is_native_format(make_model("org/x", library_name="mlx")) is False

    assert cat.is_not_nsfw(make_model()) is True
    assert cat.is_not_nsfw(make_model(tags=["NSFW"])) is False


@pytest.mark.parametrize("gated_value", [True, "manual", "auto"])
def test_gated_models_pass_both_expensive_gates(gated_value, monkeypatch):
    """Gated repos must reach the pod, which has the token and licences.

    Both expensive gates fail closed on a gated repo without an accepted
    licence — contains_remote_code because AutoConfig raises OSError (verified
    live against google/gemma-2-9b), has_loadable_weights because
    list_repo_files 404s. Without the short-circuits, removing the explicit
    `model.gated` filter would achieve nothing: gated models would just be
    rejected here instead, under a misleading reason.
    """

    def explode(*args, **kwargs):
        raise AssertionError("network call made for a gated repo")

    monkeypatch.setattr(cat.AutoConfig, "from_pretrained", explode)

    model = make_model(gated=gated_value)
    assert cat.contains_remote_code(model, False) is False
    assert cat.has_loadable_weights(model, False) is True


# ---------------------------------------------------------------------------
# merge_curated()
# ---------------------------------------------------------------------------


def test_merge_curated_appends_and_flags():
    fetched = [make_model("org/a"), make_model("org/b")]
    curated = [make_model("org/z")]
    merged, curated_ids = merge_curated(fetched, curated)

    assert [m.id for m in merged] == ["org/a", "org/b", "org/z"]
    assert curated_ids == {"org/z"}


def test_merge_curated_dedups_case_insensitively():
    """Hub ids resolve case-insensitively and this repo already has casing
    drift for the same model, so a differently-spelled curated id must not
    produce a second row — but it must still be flagged curated.
    """
    fetched = [make_model("google/gemma-4-31B")]
    curated = [make_model("google/gemma-4-31b")]
    merged, curated_ids = merge_curated(fetched, curated)

    assert len(merged) == 1, "case-differing ids produced a duplicate row"
    assert merged[0].id == "google/gemma-4-31B", "fetched (canonical) spelling lost"
    assert "google/gemma-4-31B" in curated_ids, "overlapping model not flagged curated"


def test_merge_curated_flags_model_present_in_both_lists():
    fetched = [make_model("org/a"), make_model("org/shared")]
    curated = [make_model("org/shared")]
    merged, curated_ids = merge_curated(fetched, curated)

    assert len(merged) == 2
    assert "org/shared" in curated_ids
    assert "org/a" not in curated_ids


# ---------------------------------------------------------------------------
# build_catalog()
# ---------------------------------------------------------------------------

_ALWAYS_PASS = (("gate", lambda m, t: True),)


def _no_network(monkeypatch, config_class: str | None = "LlamaConfig") -> list[str]:
    """Stub get_config_type; return the list it records calls into."""
    called: list[str] = []

    def fake_get_config_type(model_id, token):
        called.append(model_id)
        return config_class

    monkeypatch.setattr(cat, "get_config_type", fake_get_config_type)
    return called


def test_build_catalog_records_rejected_rows_instead_of_dropping(monkeypatch):
    _no_network(monkeypatch)
    models = [make_model("org/good"), make_model("org/bad", has_config=False)]
    gates = ((REASON_NO_CONFIG, lambda m, t: cat.has_config(m)),)

    rows = build_catalog(
        fetch_fn=lambda lim: models,
        gates=gates,
        limit=10,
        output_csv=None,
        label="test",
        token=False,
    )

    assert len(rows) == 2, "a rejected model was dropped instead of recorded"
    by_id = {r["model_id"]: r for r in rows}
    assert by_id["org/good"]["rejection_reason"] is None
    assert by_id["org/bad"]["rejection_reason"] == REASON_NO_CONFIG


def test_build_catalog_skips_config_enrichment_for_rejected_rows(monkeypatch):
    """Rejected rows must not each cost an AutoConfig download."""
    called = _no_network(monkeypatch)
    models = [make_model(f"org/m{i}") for i in range(5)]
    # Reject everything except m0.
    gates = ((REASON_NOT_AN_EMBEDDER, lambda m, t: m.id == "org/m0"),)

    build_catalog(
        fetch_fn=lambda lim: models,
        gates=gates,
        limit=10,
        output_csv=None,
        label="test",
        token=False,
    )

    assert called == ["org/m0"], f"config class fetched for rejected rows: {called}"


def test_build_catalog_is_supported_is_tristate(monkeypatch):
    """None ("not determined") must be distinguishable from False ("no adapter").

    weekly_test.py compares with `is False`, and a rejected/gated row reporting
    False would be recorded as not-implemented-adapter — wrong, and terminal for
    10 days.
    """
    _no_network(monkeypatch, config_class=None)
    rows = build_catalog(
        fetch_fn=lambda lim: [make_model("org/unreadable")],
        gates=_ALWAYS_PASS,
        limit=10,
        output_csv=None,
        label="test",
        token=False,
    )
    assert rows[0]["config_class"] is None
    assert rows[0]["is_supported"] is None, "unresolved config reported as False"

    _no_network(monkeypatch, config_class="NotARealConfig")
    rows = build_catalog(
        fetch_fn=lambda lim: [make_model("org/unsupported")],
        gates=_ALWAYS_PASS,
        limit=10,
        output_csv=None,
        label="test",
        token=False,
    )
    assert rows[0]["is_supported"] is False, "resolved-but-unsupported must be False"


def test_build_catalog_ranks_only_passing_models(monkeypatch):
    _no_network(monkeypatch)
    models = [make_model(f"org/m{i}") for i in range(4)]
    gates = ((REASON_REMOTE_CODE, lambda m, t: m.id != "org/m2"),)

    rows = build_catalog(
        fetch_fn=lambda lim: models,
        gates=gates,
        limit=10,
        output_csv=None,
        label="test",
        token=False,
    )

    ranked = [r for r in rows if r["rejection_reason"] is None]
    rejected = [r for r in rows if r["rejection_reason"] is not None]
    assert [r["rank"] for r in ranked] == [1, 2, 3], "ranks not contiguous from 1"
    assert all(r["rank"] is None for r in rejected), "rejected row carries a rank"


def test_build_catalog_limit_applies_to_passing_rows_only(monkeypatch):
    """--top-k counts testable models; diagnostics are additional."""
    _no_network(monkeypatch)
    models = [make_model(f"org/m{i}") for i in range(10)]
    # Reject the odd-numbered half.
    gates = ((REASON_NON_NATIVE_FORMAT, lambda m, t: int(m.id[-1]) % 2 == 0),)

    rows = build_catalog(
        fetch_fn=lambda lim: models,
        gates=gates,
        limit=3,
        output_csv=None,
        label="test",
        token=False,
    )

    passing = [r for r in rows if r["rejection_reason"] is None]
    assert len(passing) == 3, "limit did not cap the passing rows"
    assert len(rows) > 3, "rejected rows were folded into the limit"


def test_build_catalog_keeps_bool_types_for_json_round_trip(monkeypatch):
    """weekly_test.py compares is_supported with `is False`, so these must stay
    real bools (or None) — a string or int would silently skip that branch.
    """
    _no_network(monkeypatch)
    rows = build_catalog(
        fetch_fn=lambda lim: [make_model("org/a")],
        gates=_ALWAYS_PASS,
        limit=10,
        output_csv=None,
        label="test",
        token=False,
    )
    assert isinstance(rows[0]["is_supported"], bool)
    assert isinstance(rows[0]["is_moe"], bool)
    assert isinstance(rows[0]["curated"], bool)


def test_build_catalog_attaches_is_moe_and_model_info(monkeypatch):
    _no_network(monkeypatch)
    rows = build_catalog(
        fetch_fn=lambda lim: [
            make_model("org/a"),
            make_model("org/moe", model_type="mixtral"),
        ],
        gates=_ALWAYS_PASS,
        limit=10,
        output_csv=None,
        label="test",
        token=False,
    )
    by_id = {r["model_id"]: r for r in rows}
    assert by_id["org/a"]["is_moe"] is False
    assert by_id["org/moe"]["is_moe"] is True
    assert by_id["org/a"]["model_info"] is not None


# ---------------------------------------------------------------------------
# Curated models through build_catalog
# ---------------------------------------------------------------------------


def test_build_catalog_curated_survive_truncation(monkeypatch):
    """A curated id is named deliberately, so a small limit must not drop it."""
    _no_network(monkeypatch)
    fetched = [make_model(f"org/top{i}") for i in range(5)]
    monkeypatch.setattr(
        cat,
        "fetch_curated_model_infos",
        lambda ids, *, api, label: ([make_model("org/curated")], []),
    )

    rows = build_catalog(
        fetch_fn=lambda lim: fetched,
        gates=_ALWAYS_PASS,
        limit=2,
        output_csv=None,
        label="test",
        token=False,
        curated_ids=["org/curated"],
        api=object(),
    )

    by_id = {r["model_id"]: r for r in rows}
    assert "org/curated" in by_id, "curated model lost to truncation"
    assert by_id["org/curated"]["curated"] is True
    assert by_id["org/curated"]["rank"] is None, "curated model given a download rank"
    assert sum(1 for r in rows if r["rank"] is not None) == 2


def test_build_catalog_unresolvable_curated_id_becomes_a_row(monkeypatch):
    """A typo'd or not-yet-released curated id must be visible, not absent."""
    _no_network(monkeypatch)
    monkeypatch.setattr(
        cat,
        "fetch_curated_model_infos",
        lambda ids, *, api, label: ([], ["org/typo"]),
    )

    rows = build_catalog(
        fetch_fn=lambda lim: [make_model("org/a")],
        gates=_ALWAYS_PASS,
        limit=10,
        output_csv=None,
        label="test",
        token=False,
        curated_ids=["org/typo"],
        api=object(),
    )

    failed = [r for r in rows if r["model_id"] == "org/typo"]
    assert len(failed) == 1, "unresolvable curated id produced no row"
    assert failed[0]["rejection_reason"] == REASON_CURATED_FETCH_FAILED
    assert failed[0]["curated"] is True
    assert failed[0]["is_supported"] is None
    assert failed[0]["is_moe"] is False
    assert failed[0]["model_info"] is None


def test_build_catalog_requires_api_for_curated_ids():
    with pytest.raises(ValueError, match="requires an api"):
        build_catalog(
            fetch_fn=lambda lim: [],
            gates=_ALWAYS_PASS,
            limit=1,
            output_csv=None,
            label="test",
            token=False,
            curated_ids=["org/x"],
            api=None,
        )


def test_build_catalog_writes_csv_with_new_columns(monkeypatch, tmp_path):
    _no_network(monkeypatch)
    out = tmp_path / "catalog.csv"
    build_catalog(
        fetch_fn=lambda lim: [
            make_model("org/a"),
            make_model("org/b", has_config=False),
        ],
        gates=((REASON_NO_CONFIG, lambda m, t: cat.has_config(m)),),
        limit=10,
        output_csv=out,
        label="test",
        token=False,
    )
    header = out.read_text().splitlines()[0]
    assert "rejection_reason" in header
    assert "curated" in header
    assert len(out.read_text().splitlines()) == 3  # header + 2 rows


# ---------------------------------------------------------------------------
# load_curated_model_ids()
# ---------------------------------------------------------------------------


def test_load_curated_model_ids_parses_comments_blanks_and_dupes(tmp_path):
    path = tmp_path / "curated.txt"
    path.write_text(
        "\n".join(
            [
                "# a leading comment",
                "",
                "org/first",
                "   org/second   # trailing comment",
                "  # indented comment",
                "org/first",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert load_curated_model_ids(path) == ["org/first", "org/second"]


def test_load_curated_model_ids_missing_file_returns_empty(tmp_path):
    """A deleted resource file must not take down the weekly scan."""
    assert load_curated_model_ids(tmp_path / "nope.txt") == []


def test_load_curated_model_ids_reads_the_real_resource_files():
    for path in (
        cat.CURATED_GENERATIVE_MODELS_FILE,
        cat.CURATED_EMBEDDING_MODELS_FILE,
    ):
        ids = load_curated_model_ids(path)
        assert ids, f"{path.name} parsed to an empty list"
        assert all("#" not in i and i == i.strip() for i in ids)
        assert all("/" in i for i in ids), "an entry is not an org/name model id"
        assert len(ids) == len(set(ids)), "duplicate ids survived"


def test_gate_tuples_keep_network_gates_last():
    """The cost invariant, asserted structurally on the real gate tuples."""
    from utils.fetch_top_embedding_models import EMBEDDING_GATES
    from utils.fetch_top_generative_models import GENERATIVE_GATES

    networked = {REASON_REMOTE_CODE, REASON_NO_LOADABLE_WEIGHTS}
    for gates in (GENERATIVE_GATES, EMBEDDING_GATES):
        names = [name for name, _ in gates]
        assert networked.issubset(names)
        first_networked = min(names.index(n) for n in networked)
        assert all(
            n in networked for n in names[first_networked:]
        ), f"a cheap gate runs after a networked one: {names}"
