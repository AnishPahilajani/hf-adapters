"""Shared building blocks for the Hugging Face model-catalog fetchers.

Both ``fetch_top_generative_models.py`` and ``fetch_top_embedding_models.py``
pull models from the Hub, enrich them with config/param metadata, and write a
ranked CSV. Everything they have in common lives here; each script only has to
supply how it *sources* candidates, how it *filters* them, and any *extra
columns* it wants on top of the shared schema.
"""

import csv
import logging
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.hf_api import HfApi, ModelInfo
from tqdm import tqdm
from transformers import AutoConfig

# Import the mapping to get supported config classes dynamically
from hf_adapters.auto_spyre_model import CONFIG_TO_ADAPTER_MODULE_MAPPING

logging.getLogger("transformers").setLevel(logging.ERROR)


# Get the resources directory (parent of resources/__init__.py)
RESOURCES_DIR: Path = Path(__file__).resolve().parent.parent / "resources"

# Manually-maintained model-id lists (one id per line; '#' comments and blank
# lines ignored). See load_curated_model_ids().
CURATED_GENERATIVE_MODELS_FILE: Path = RESOURCES_DIR / "generative_models_curated.txt"
CURATED_EMBEDDING_MODELS_FILE: Path = RESOURCES_DIR / "embedding_models_curated.txt"


def load_curated_model_ids(path: Path) -> list[str]:
    """Load Hugging Face model ids from a curated list file.

    The file holds one model id per line. Blank lines and lines whose first
    non-whitespace character is ``#`` are ignored, as is any inline ``#``
    comment following an id. Surrounding whitespace is stripped. Order is
    preserved and duplicates are dropped (first occurrence wins).

    A missing file yields an empty list rather than raising: these lists are
    an optional supplement to the Hub query, and a deleted or renamed resource
    file should not take down the weekly scan.
    """
    if not Path(path).exists():
        logging.warning("curated model list not found, skipping: %s", path)
        return []
    seen: set[str] = set()
    model_ids: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line in seen:
            continue
        seen.add(line)
        model_ids.append(line)
    return model_ids


# Metadata fields requested from list_models for every fetcher.
EXPAND_FIELDS: list[str] = [
    "config",
    "safetensors",
    "gated",
    "likes",
    "downloads",
    "createdAt",
    "library_name",
    "tags",
]

# HF-API gateway 5xx statuses. Anything outside this set (400/401/403/404/...)
# is a permanent failure and must not be retried.
_TRANSIENT_HTTP_STATUSES: frozenset[int] = frozenset({500, 502, 503, 504})
_MAX_FETCH_ATTEMPTS: int = 5
_MAX_BACKOFF_SECONDS: float = 60.0

_T = TypeVar("_T")


def with_transient_retry(
    call: Callable[[], Iterator[_T] | Iterable[_T]],
    description: str,
) -> list[_T]:
    """Materialize an HF paginated call, retrying transient 5xx failures.

    *call* is expected to return an iterable of ``ModelInfo`` (or similar)
    from a ``huggingface_hub`` API method. It is fully consumed into a list
    on each attempt because the pagination endpoint's failure mode is a
    mid-stream 504, and there is no way to resume — the whole traversal has
    to restart. Transient statuses (500/502/503/504) trigger up to
    ``_MAX_FETCH_ATTEMPTS`` retries with exponential backoff capped at
    ``_MAX_BACKOFF_SECONDS``; any other error propagates immediately.
    """
    last_error: HfHubHTTPError | None = None
    for attempt in range(1, _MAX_FETCH_ATTEMPTS + 1):
        try:
            return list(call())
        except HfHubHTTPError as e:
            status: int | None = (
                e.response.status_code if e.response is not None else None
            )
            if status not in _TRANSIENT_HTTP_STATUSES:
                raise
            last_error = e
            backoff: float = min(_MAX_BACKOFF_SECONDS, 2.0**attempt)
            print(
                f"    {description}: HF API returned {status} "
                f"(attempt {attempt}/{_MAX_FETCH_ATTEMPTS}); "
                f"retrying in {backoff:.0f}s..."
            )
            time.sleep(backoff)
    assert last_error is not None
    raise last_error


MOE_MODEL_TYPES: set[str] = {
    "mixtral",
    "qwen2_moe",
    "qwen3_moe",
    "dbrx",
    "jamba",
    "arctic",
    "olmoe",
    "gpt_oss",
}

MOE_MODEL_TYPE_PREFIXES: tuple[str, ...] = ("deepseek_v2", "deepseek_v3", "deepseek_v4")

MOE_ARCH_SUBSTRINGS: list[str] = [
    "mixtral",
    "moe",
    "dbrx",
    "jamba",
    "arctic",
    "olmoe",
    "deepseek",
    "gptoss",
]

# Get supported config class names dynamically from the mapping
SUPPORTED_CONFIG_CLASSES: set[str] = {
    cls.__name__ for cls in CONFIG_TO_ADAPTER_MODULE_MAPPING.keys()
}


def tags(model: ModelInfo) -> set[str]:
    """Lower-cased set of a model's tags (empty set if none)."""
    return {t.lower() for t in (getattr(model, "tags", None) or [])}


def is_supported_config(config_class_name: str | None) -> bool:
    """Check if the config class is supported by our adapter code."""
    if config_class_name is None:
        return False
    return config_class_name in SUPPORTED_CONFIG_CLASSES


def is_moe(model: ModelInfo) -> bool:
    if any("moe" in t for t in tags(model)):
        return True

    config: dict = model.config or {}
    model_type: str = (config.get("model_type") or "").lower()
    if model_type in MOE_MODEL_TYPES:
        return True
    if model_type.startswith(MOE_MODEL_TYPE_PREFIXES):
        return True

    architectures: list[str] = config.get("architectures") or []
    arch_lower: str = " ".join(architectures).lower()
    return any(sub in arch_lower for sub in MOE_ARCH_SUBSTRINGS)


def is_custom_code(model: ModelInfo) -> bool:
    if "custom_code" in tags(model):
        return True
    config: dict = model.config or {}
    return bool(config.get("auto_map"))


def is_nsfw(model: ModelInfo) -> bool:
    if "nsfw" in tags(model):
        return True
    return False


# Repo-id substrings marking non-native conversions (ONNX/GGUF/MLX), dropped.
NON_NATIVE_ID_SUBSTRINGS: tuple[str, ...] = ("onnx", "gguf", "mlx")


# ---------------------------------------------------------------------------
# Rejection reasons
#
# A candidate that fails a gate is no longer discarded — it becomes a row
# carrying the name of the gate it failed (see diagnose()), and that name is
# written verbatim into ClickHouse's failure_category column by
# weekly_test.py. So these strings are a STABLE WIRE CONTRACT: renaming one
# orphans every historical row that carries the old name.
#
# Recording rather than dropping exists because dropping hid a real incident:
# a PermissionError on HF_HOME made contains_remote_code() report "needs
# remote code" for every single model, filtering the entire list away with no
# trace (see .github/workflows/push-to-clickhouse.yaml, "Overrides the
# workflow-level HF_HOME"). The same failure now shows up as thousands of
# identical failure_category rows.
#
# Note there is deliberately NO "gated" reason. Gatedness is not a rejection:
# the Spyre pod has the HF token and license acceptances, so a gated model is
# passed through for the pod to judge (see the gated short-circuits below).
# ---------------------------------------------------------------------------
REASON_NO_CONFIG: str = "no_config"
REASON_NON_NATIVE_FORMAT: str = "non_native_format"
REASON_NSFW: str = "nsfw"
REASON_NOT_AN_EMBEDDER: str = "not_an_embedder"
REASON_RERANKER: str = "reranker"
REASON_REMOTE_CODE: str = "remote_code"
REASON_NO_LOADABLE_WEIGHTS: str = "no_loadable_weights"
REASON_CURATED_FETCH_FAILED: str = "curated_fetch_failed"


def has_config(model: ModelInfo) -> bool:
    """True if the Hub exposes a config for this repo.

    Config-less repos are abandoned uploads or non-transformers artifacts;
    without a config every downstream field (model_type, architectures,
    is_moe) is unknowable.
    """
    return bool(model.config)


def is_native_format(model: ModelInfo) -> bool:
    """False for ONNX/GGUF/MLX conversions, which AutoModel cannot consume."""
    if model.library_name in NON_NATIVE_ID_SUBSTRINGS:
        return False
    model_id_lower: str = model.id.lower()
    return not any(sub in model_id_lower for sub in NON_NATIVE_ID_SUBSTRINGS)


def is_not_nsfw(model: ModelInfo) -> bool:
    """False if the repo carries the nsfw tag."""
    return "nsfw" not in tags(model)


def is_baseline_keep(model: ModelInfo) -> bool:
    """Shared inclusion gate: drop config-less, and ONNX/GGUF/MLX id checkpoints.

    Kept as a composition of the three named predicates above so callers that
    only want a bool verdict (rather than diagnose()'s named cause) still have
    one.
    """
    return has_config(model) and is_native_format(model) and is_not_nsfw(model)


def contains_remote_code(model: ModelInfo, token: str | bool = False) -> bool:
    """Return True if the model requires trust_remote_code=True for its config.

    Gated repos short-circuit to False. Two reasons this matters:

    * Without an accepted license, AutoConfig raises OSError here regardless of
      whether the model actually needs remote code — so a gated model would be
      reported as needing it. Verified live: google/gemma-2-9b hits this,
      meta-llama/Llama-3.1-8B does not. Whether a given gated repo trips it
      depends on whether its config happens to be publicly readable, which
      makes the recorded reason look nondeterministic.
    * The Spyre pod holds the token and license acceptances, and
      resolve_adapter_module() re-reads the config there, so the pod is the
      environment entitled to make this call — not the ubuntu-latest job that
      builds the shards.

    ``token`` is now passed through to AutoConfig. Its absence was a latent
    bug: get_config_type() passed one, this function did not, so the two
    disagreed about the same repo.
    """
    if model.gated:
        return False
    try:
        AutoConfig.from_pretrained(model.id, token=token, trust_remote_code=False)
        return False
    except (ValueError, OSError):
        return True


# Session-scoped cache for _has_loadable_weights. Keyed by repo_id; values are
# the bool result. The fetchers run twice a week in a fresh process, and repo
# file lists rarely change within a single run, so a plain dict is enough — no
# TTL or on-disk persistence needed.
_LOADABLE_WEIGHTS_CACHE: dict[str, bool] = {}

# Filenames transformers' AutoModel.from_pretrained recognizes as native
# weights (single-file or sharded via the matching index.json).
_NATIVE_WEIGHT_FILES: frozenset[str] = frozenset(
    {
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    }
)


def has_loadable_weights(model: ModelInfo, token: str | bool) -> bool:
    """True if the repo ships weights AutoModel.from_pretrained can consume.

    Detects three unloadable classes without downloading any weight files
    — one ``list_repo_files`` call per repo:

    * adapter-only repos (LoRA/PEFT, `adapter_config.json` but no full model),
    * GGUF/MLX/ONNX-only repos that slipped past the id-substring filter,
    * abandoned uploads with a config but no weight files at all.

    Cached in-process by repo_id: transformers repos are effectively immutable
    within a fetcher run, and each fetcher process is short-lived.

    Gated repos short-circuit to True: list_repo_files 404s on a repo whose
    license has not been accepted, and the ``except Exception`` below cannot
    distinguish that from a genuinely weightless repo. Since the Spyre pod has
    the token and acceptances, the pod decides — same rationale as
    contains_remote_code(). Without this, removing the explicit `model.gated`
    filter from keep() would be pointless: gated models would simply be
    rejected here instead, under a misleading reason.
    """
    from huggingface_hub import HfApi

    if model.gated:
        return True

    cached = _LOADABLE_WEIGHTS_CACHE.get(model.id)
    if cached is not None:
        return cached

    api: HfApi = HfApi(token=token)
    try:
        files: list[str] = with_transient_retry(
            lambda: api.list_repo_files(model.id, token=token),
            description=f"list_repo_files[{model.id}]",
        )
    except Exception:
        # Any permanent failure (404, gated without token, ...) — treat as
        # not loadable rather than raising into the fetcher's filter path.
        _LOADABLE_WEIGHTS_CACHE[model.id] = False
        return False

    lower_files: set[str] = {f.lower() for f in files}

    # Adapter-only repos ship adapter_config.json + adapter_model.safetensors
    # and expect PeftModel.from_pretrained(base, ...), not AutoModel directly.
    if "adapter_config.json" in lower_files:
        _LOADABLE_WEIGHTS_CACHE[model.id] = False
        return False

    result: bool = any(name in lower_files for name in _NATIVE_WEIGHT_FILES)
    _LOADABLE_WEIGHTS_CACHE[model.id] = result
    return result


# A gate is (reason_name, predicate); predicate(model, token) -> True == PASS.
Gate = tuple[str, Callable[[ModelInfo, str | bool], bool]]


def diagnose(
    model: ModelInfo, gates: tuple[Gate, ...], token: str | bool
) -> str | None:
    """Return the name of the FIRST gate *model* fails, or None if it passes all.

    Short-circuits deliberately: one cause per model, the cheapest one that
    applies. Each fetcher declares its own ordered gate tuple, and that order
    is load-bearing — the two gates that touch the network
    (REASON_REMOTE_CODE does an AutoConfig download, REASON_NO_LOADABLE_WEIGHTS
    a list_repo_files call) must come last so a candidate rejected by a cheap
    metadata check never pays for an HTTP round-trip. At ~20k raw candidates
    per run that ordering is the difference between the fetch job finishing
    inside its timeout and not.

    A gate that *raises* is treated as a failure attributed to that gate, with
    a warning, rather than propagating or silently dropping the model. That is
    what makes an environment-wide breakage legible: when HF_HOME was
    unwritable, every AutoConfig call raised, and the result should be 20k rows
    all blaming REASON_REMOTE_CODE — an obvious signal — instead of an empty
    scan.
    """
    for reason, predicate in gates:
        try:
            if not predicate(model, token):
                return reason
        except Exception as e:
            logging.warning("gate %s raised for %s: %s", reason, model.id, e)
            return reason
    return None


def format_number_to_billions_smart(num: int | float) -> str:
    """Smart formatting that adjusts precision based on magnitude."""
    billions: float = num / 1_000_000_000

    if billions >= 10:
        # For numbers >= 10B, round to nearest integer
        result: int | float = round(billions)
        return f"{result}B"
    elif billions >= 1:
        # For 1B-10B, show 1 decimal place
        result = round(billions, 1)
        return f"{result}B" if result != int(result) else f"{int(result)}B"
    else:
        # For < 1B, show 1-2 decimal places
        result = round(billions, 2)
        return f"{result}B"


def parse_number_suffix(value: str) -> int:
    value = value.strip().upper()

    multipliers: dict[str, int] = {
        "K": 1_000,
        "M": 1_000_000,
        "B": 1_000_000_000,
        "T": 1_000_000_000_000,
    }

    suffix: str = value[-1]

    if suffix in multipliers:
        number: float = float(value[:-1])
        return int(number * multipliers[suffix])

    # No suffix → return as integer
    return int(float(value))


def extract_model_size_from_model_name(
    model_name: str, allow_millions: bool = False
) -> str | None:
    """Pull a parameter-size token (e.g. "7B", "33M") out of a model id.

    ``allow_millions`` also matches an ``M`` suffix — useful for embedding
    models, which are frequently sized in the tens/hundreds of millions.
    Returns the token only if exactly one match is found (avoids ambiguity).
    """
    units: str = "MBmb" if allow_millions else "Bb"
    pattern: str = rf"\b\d+(?:\.\d+)?[{units}]\b"
    matches: list[str] = re.findall(pattern, model_name)
    return matches[0] if len(matches) == 1 else None


def get_param_count(model: ModelInfo) -> int | None:
    if model.safetensors and model.safetensors.parameters:
        return sum(model.safetensors.parameters.values())
    return None


def get_config_type(model_id: str, token: str | bool) -> str | None:
    try:
        model_config = AutoConfig.from_pretrained(
            model_id, token=token, trust_remote_code=False
        )
        return type(model_config).__name__
    except Exception:
        return None


# Columns written for every catalog, in order. Each entry is
# (header, value_fn) where value_fn(model, config_class) -> cell value. The
# rank/config-class/param columns are interleaved by build_catalog because they
# depend on per-row computed state.
def _resolve_param_columns(
    model: ModelInfo, allow_millions: bool
) -> tuple[str | None, int | None]:
    """Return (param_str, param_int) for a model, name first then safetensors."""
    param_str: str | None = extract_model_size_from_model_name(model.id, allow_millions)
    param_int: int | None
    if param_str is None:
        param_int = get_param_count(model)
        if param_int is not None:
            param_str = format_number_to_billions_smart(param_int)
    else:
        param_int = parse_number_suffix(param_str)
    return param_str, param_int


def fetch_curated_model_infos(
    model_ids: list[str],
    *,
    api: HfApi,
    label: str,
) -> tuple[list[ModelInfo], list[str]]:
    """Resolve curated model ids to ModelInfo, one ``model_info`` call each.

    Requests the same ``EXPAND_FIELDS`` that ``list_models`` does, so a curated
    ModelInfo is metadata-identical to a fetched one and takes the exact same
    enrichment and gating path — curated-ness is provenance, never a behavioral
    shortcut.

    Returns ``(infos, failed_ids)``. An id that cannot be resolved (typo,
    renamed repo, private, or simply not released yet) is returned in
    *failed_ids* rather than raising: the curated files intentionally name
    unreleased models, so this is a normal steady state, and one stale line
    must not abort the weekly scan. Transient 5xx are retried by
    ``with_transient_retry``.
    """
    if not model_ids:
        return [], []

    def _one(model_id: str) -> tuple[str, ModelInfo | None]:
        try:
            # with_transient_retry materializes an iterable, so wrap the single
            # result to reuse its 5xx backoff policy unchanged.
            infos: list[ModelInfo] = with_transient_retry(
                # EXPAND_FIELDS is a plain list[str]; model_info's signature
                # wants a list of Literals. The values are identical to the ones
                # list_models() is already called with, so the cast is safe and
                # keeps one source of truth for the field list.
                lambda: [
                    api.model_info(model_id, expand=EXPAND_FIELDS)  # type: ignore[arg-type]
                ],
                description=f"model_info[{model_id}]",
            )
            return model_id, infos[0]
        except Exception as e:
            logging.warning("curated %s: model_info failed: %s", model_id, e)
            return model_id, None

    infos: list[ModelInfo] = []
    failed_ids: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for model_id, info in ex.map(_one, model_ids):
            if info is None:
                failed_ids.append(model_id)
            else:
                infos.append(info)

    print(
        f"curated {label}: {len(infos)} of {len(model_ids)} id(s) resolved"
        + (f", {len(failed_ids)} unresolved: {failed_ids}" if failed_ids else "")
    )
    return infos, failed_ids


def merge_curated(
    candidates: list[ModelInfo], curated: list[ModelInfo]
) -> tuple[list[ModelInfo], set[str]]:
    """Union fetched and curated candidates; return (merged, curated_id_set).

    Curated models are appended *after* the download-ranked candidates rather
    than prepended. Rank is assigned by list position, so prepending would give
    a curated model with a handful of downloads rank 1 and destroy the
    week-over-week comparability of the whole column. With limit in the
    thousands against a handful of curated ids, the truncation risk that
    prepending would guard against is negligible — and build_catalog warns if a
    curated model does land past the cut.

    Dedup is case-insensitive: Hub ids are case-preserving but resolve
    case-insensitively, and this repo already contains casing drift for the
    same model (see MODEL_PATH_TO_TORCH_DTYPE in hf_adapters/
    auto_spyre_model.py, which has both 'gemma-4-12B-it' and lowercase
    entries). A curated file spelled differently from the Hub's canonical
    casing must not produce a duplicate row. The fetched entry wins on a
    collision — the metadata is identical either way — but the id is still
    marked curated.
    """
    curated_id_set: set[str] = {m.id for m in curated}
    by_key: dict[str, ModelInfo] = {m.id.casefold(): m for m in candidates}
    for m in curated:
        by_key.setdefault(m.id.casefold(), m)
    merged: list[ModelInfo] = list(by_key.values())
    # Mark the canonical (fetched) spelling as curated too, so a curated model
    # already present in the top-K is flagged rather than appearing unmarked.
    curated_keys: set[str] = {mid.casefold() for mid in curated_id_set}
    curated_id_set |= {m.id for m in merged if m.id.casefold() in curated_keys}
    return merged, curated_id_set


def build_catalog(
    *,
    fetch_fn: Callable[[int], Iterable[ModelInfo]],
    gates: tuple[Gate, ...],
    limit: int,
    output_csv: Path | str | None,
    label: str,
    extra_columns: (
        list[tuple[str, Callable[[ModelInfo, str | None], object]]] | None
    ) = None,
    allow_millions: bool = False,
    token: str | bool,
    curated_ids: list[str] | None = None,
    api: HfApi | None = None,
) -> list[dict[str, object]]:
    """Fetch → classify → enrich → return a ranked model catalog as a list of dicts.

    Every candidate becomes a row. Models that fail a gate are NOT dropped:
    they come back with ``rejection_reason`` set to the name of the gate they
    failed, so the weekly scan can record *why* a model was never tested
    instead of leaving a hole. Rows that passed everything have
    ``rejection_reason`` of None and are the only ones ranked and counted
    against *limit*.

    Args:
        fetch_fn: callable(limit) -> list of raw model objects (over-fetched).
        gates: ordered (reason, predicate) tuple; see diagnose(). Order is
            load-bearing — cheap metadata checks must precede networked ones.
        limit: number of *passing* rows to keep. Rejected rows are additional
            (capped separately) so diagnostics never displace real test targets.
        output_csv: destination path, or None to skip writing.
        label: human-readable noun for log lines (e.g. "generative").
        extra_columns: optional list of (header, value_fn) where
            value_fn(model, config_class) -> cell. Inserted before config_class.
            NOTE: config_class is None for rejected rows (their config class is
            never resolved — see below), so a value_fn must tolerate None.
        allow_millions: pass-through to the size-name extractor.
        token: HF token (or True) for AutoConfig downloads.
        curated_ids: manually-maintained ids to merge in alongside the fetched
            candidates. Resolved via model_info; unresolvable ids still produce
            a row, flagged REASON_CURATED_FETCH_FAILED.
        api: HfApi used to resolve *curated_ids*. Required if curated_ids is set.

    Returns:
        List of dicts, one per model, keyed by column name. Passing rows first
        (rank 1..N), then rejected rows (rank None).
    """
    extra_columns = extra_columns or []

    candidates: list[ModelInfo] = list(fetch_fn(limit))
    print(f"Retrieved {len(candidates)} raw {label} candidates.")

    curated_id_set: set[str] = set()
    curated_failed_ids: list[str] = []
    if curated_ids:
        if api is None:
            raise ValueError("curated_ids requires an api instance")
        curated_infos, curated_failed_ids = fetch_curated_model_infos(
            curated_ids, api=api, label=label
        )
        before: int = len(candidates)
        candidates, curated_id_set = merge_curated(candidates, curated_infos)
        print(
            f"curated {label}: merged {len(candidates) - before} new id(s) "
            f"({len(curated_infos) - (len(candidates) - before)} already in the "
            f"fetched list)"
        )

    with ThreadPoolExecutor(max_workers=16) as ex:
        reasons: list[str | None] = list(
            tqdm(
                ex.map(lambda m: diagnose(m, gates, token), candidates),
                total=len(candidates),
                desc="Diagnosing candidates",
            )
        )

    passing: list[ModelInfo] = [m for m, r in zip(candidates, reasons) if r is None]
    rejected: list[tuple[ModelInfo, str]] = [
        (m, r) for m, r in zip(candidates, reasons) if r is not None
    ]
    print(
        f"{len(passing)} {label} models passed every gate; "
        f"{len(rejected)} rejected (recorded, not dropped)."
    )
    # Per-cause histogram. This is the observability payoff: a run where a
    # single cause accounts for nearly every candidate is an environment
    # failure (bad token, unwritable HF_HOME), not a property of the Hub.
    for rejection_cause, count in Counter(r for _, r in rejected).most_common():
        print(f"    {rejection_cause}: {count}")

    # Apply *limit* to the download-ranked models only, then re-append every
    # curated model. A curated id was named deliberately, so it must survive
    # truncation at any limit — otherwise a small --top-k silently tests none of
    # them. Rejected rows get their own cap: they are diagnostics and must never
    # push a real model out of the top-K, but the fetchers over-fetch 2x, so an
    # uncapped tail would emit roughly twice as many rows as asked for.
    def _split_curated(
        models: list[ModelInfo],
    ) -> tuple[list[ModelInfo], list[ModelInfo]]:
        ranked = [m for m in models if m.id not in curated_id_set]
        curated_only = [m for m in models if m.id in curated_id_set]
        return ranked, curated_only

    passing_ranked, passing_curated = _split_curated(passing)
    passing = passing_ranked[:limit] + passing_curated

    rejected_ranked = [(m, r) for m, r in rejected if m.id not in curated_id_set]
    rejected_curated = [(m, r) for m, r in rejected if m.id in curated_id_set]
    rejected = rejected_ranked[:limit] + rejected_curated

    if curated_id_set:
        print(
            f"    curated: {len(passing_curated)} passing, "
            f"{len(rejected_curated)} rejected, "
            f"{len(curated_failed_ids)} unresolved "
            f"(all retained regardless of limit={limit})"
        )

    # One ordered list of (model, reason, rank) drives both the CSV and the
    # returned dicts. Holding it in a single structure avoids the classic bug of
    # zipping the row list against a differently-ordered model list.
    #
    # rank means "position in the download ranking we are testing", so only
    # download-ranked passing models get a number. Curated-only models get None:
    # they were included by name, not by rank, and giving one a number would
    # imply a download position it does not hold (and would shift ranks
    # week-over-week as the curated file changes). Rejected rows are unranked
    # for the same reason. rank is CSV/human-facing only — it is not in
    # TABLE_COLUMNS, so nothing downstream depends on it.
    ordered: list[tuple[ModelInfo, str | None, int | None]] = (
        [(m, None, rank) for rank, m in enumerate(passing_ranked[:limit], start=1)]
        + [(m, None, None) for m in passing_curated]
        + [(m, reason, None) for m, reason in rejected]
    )

    base_head: list[str] = [
        "rank",
        "model_id",
        "downloads",
        "likes",
        "model_type",
        "architectures",
        "parameters (str)",
        "parameters",
        "library",
        # "is_gated",
        # "is_moe",
    ]
    extra_head: list[str] = [h for h, _ in extra_columns]
    tail_head: list[str] = [
        "is_custom_code",
        "config_class",
        "is_supported",
        "Year",
        "rejection_reason",
        "curated",
    ]
    header: list[str] = base_head + extra_head + tail_head

    # Resolve config classes ONLY for rows that passed every gate.
    # get_config_type is an AutoConfig.from_pretrained network call; spending
    # one on each rejected row would add thousands of round-trips to a job that
    # already needed its CI timeout raised to 120 minutes. It is also
    # pointless: for a REASON_REMOTE_CODE rejection, that same call is what
    # failed the gate, so the answer is definitionally None.
    passing_config_classes: list[str | None]
    with ThreadPoolExecutor(max_workers=16) as ex:
        passing_config_classes = list(
            tqdm(
                ex.map(lambda m: get_config_type(m.id, token), passing),
                total=len(passing),
                desc="Fetching config classes",
            )
        )
    config_class_by_key: dict[str, str | None] = {
        m.id: cc for m, cc in zip(passing, passing_config_classes)
    }

    rows: list[dict[str, object]] = []
    config_class: str | None
    for m, reason, rank in ordered:
        config_class = config_class_by_key.get(m.id)
        architectures: list[str] | None = (m.config or {}).get("architectures")
        arch_str: str | None = ";".join(architectures) if architectures else None
        param_str, param_int = _resolve_param_columns(m, allow_millions)
        extra_vals: list[object] = [fn(m, config_class) for _, fn in extra_columns]
        # is_supported is TRI-STATE. None means "not determined", which is not
        # the same as False ("determined: no adapter exists"). A rejected row
        # never had its config class resolved, and a gated row's AutoConfig call
        # may have been swallowed by get_config_type's bare except — in both
        # cases claiming False would make weekly_test.py write a terminal
        # not-implemented-adapter row and hide the real situation for 10 days.
        is_supported: bool | None = (
            None if config_class is None else is_supported_config(config_class)
        )
        rows.append(
            dict(
                zip(
                    header,
                    [
                        rank,
                        m.id,
                        m.downloads,
                        m.likes,
                        (m.config or {}).get("model_type"),
                        arch_str,
                        param_str,
                        param_int,
                        m.library_name,
                        # "is_gated",
                        # "is_moe",
                        *extra_vals,
                        is_custom_code(m),
                        config_class,
                        is_supported,
                        m.created_at.year if m.created_at else None,
                        reason,
                        m.id in curated_id_set,
                    ],
                )
            )
        )

    # Curated ids that could not be resolved at all have no ModelInfo, so they
    # cannot travel the enrichment path above. They still get a row — that is
    # the whole point of recording rather than dropping: a typo'd or
    # not-yet-released curated id must be visible, not silently absent.
    for model_id in curated_failed_ids:
        row: dict[str, object] = {h: None for h in header}
        row.update(
            {
                "model_id": model_id,
                "downloads": 0,
                "likes": 0,
                "rejection_reason": REASON_CURATED_FETCH_FAILED,
                "curated": True,
                "is_supported": None,
            }
        )
        rows.append(row)

    if output_csv is not None:
        print(f"Writing {len(rows)} row(s) to {output_csv}")
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)

    # Attach the source ModelInfo to each row AFTER the CSV write. It is a
    # runtime-only field (not serializable, and never part of the schema),
    # useful for callers that need metadata the row dict does not expose —
    # e.g. safetensors.parameters, gated, sha, siblings.
    #
    # is_moe is precomputed here (a pure function of data already fetched —
    # tags, config.model_type, config.architectures) so callers that need it
    # don't have to carry the non-serializable ModelInfo object forward.
    #
    # zip() against `ordered` stops at the shorter sequence, which correctly
    # leaves the synthetic curated-failure rows appended above untouched: they
    # have no ModelInfo, and their is_moe/model_info defaults are already set.
    for row, (m, _reason, _rank) in zip(rows, ordered):
        row["model_info"] = m
        row["is_moe"] = is_moe(m)
    for row in rows[len(ordered) :]:
        row["model_info"] = None
        row["is_moe"] = False

    return rows
