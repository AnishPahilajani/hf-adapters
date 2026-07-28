"""Fetch the top generative models from Hugging Face, ranked by downloads."""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.hf_api import ModelInfo

from utils.hf_model_catalog import (
    CURATED_GENERATIVE_MODELS_FILE,
    EXPAND_FIELDS,
    REASON_NO_CONFIG,
    REASON_NO_LOADABLE_WEIGHTS,
    REASON_NON_NATIVE_FORMAT,
    REASON_NSFW,
    REASON_REMOTE_CODE,
    RESOURCES_DIR,
    Gate,
    build_catalog,
    contains_remote_code,
    diagnose,
    has_config,
    has_loadable_weights,
    is_native_format,
    is_not_nsfw,
    load_curated_model_ids,
    with_transient_retry,
)
from utils.utilities import ts


def _fetch(api: HfApi, limit: int) -> list[ModelInfo]:
    """Top text-generation models by downloads (over-fetched to absorb the
    GGUF/MLX entries dropped by the filter).

    Retries transient 5xx gateway errors with exponential backoff via
    ``with_transient_retry``; permanent failures propagate.
    """
    print(f"{ts()} Fetching top {limit} text-generation models by downloads...")
    return with_transient_retry(
        lambda: api.list_models(
            pipeline_tag="text-generation",
            sort="downloads",
            limit=int(limit * 2),
            expand=EXPAND_FIELDS,
        ),
        description="list_models[text-generation]",
    )


# Ordered gates for the generative fetcher. THE ORDER IS LOAD-BEARING: every
# gate above the marker answers from metadata already in hand from
# list_models(expand=...), while the two below it each cost an HTTP round-trip.
# Since diagnose() short-circuits on the first failure, keeping the cheap ones
# first means a rejected candidate never pays network cost — the same cost
# profile the old keep() had, which matters at ~20k raw candidates per run.
#
# Note there is no `model.gated` gate. Gatedness is not a rejection: the Spyre
# pod holds the token and license acceptances, so gated models are passed
# through for the pod to judge (see contains_remote_code/has_loadable_weights,
# which short-circuit on gated for the same reason).
GENERATIVE_GATES: tuple[Gate, ...] = (
    (REASON_NO_CONFIG, lambda m, _t: has_config(m)),
    (REASON_NON_NATIVE_FORMAT, lambda m, _t: is_native_format(m)),
    (REASON_NSFW, lambda m, _t: is_not_nsfw(m)),
    # --- everything below costs one network call per candidate ---
    (REASON_REMOTE_CODE, lambda m, t: not contains_remote_code(m, t)),
    (REASON_NO_LOADABLE_WEIGHTS, lambda m, t: has_loadable_weights(m, t)),
)


def keep(model: ModelInfo, token: str | bool) -> bool:
    """True if *model* passes every gate. Thin shim over diagnose().

    Retained so callers that only want a bool verdict keep working; the
    fetchers themselves use diagnose() directly so they can record the cause.
    """
    return diagnose(model, GENERATIVE_GATES, token) is None


def fetch_top_generative_models(
    limit: int,
    output_csv: Path | str | None = None,
    curated_file: Path | None = CURATED_GENERATIVE_MODELS_FILE,
) -> list[dict[str, object]]:
    """Fetch the top-*limit* generative models plus the curated list.

    *curated_file* defaults to the maintained resource list; pass None to skip
    it (used by tests, and by any caller that wants the pure Hub ranking).
    """
    # Falls back to False (explicit anonymous access), not True: in
    # huggingface_hub, token=True means "use the locally cached login token,
    # and raise LocalTokenNotFoundError if none exists" — it does NOT mean
    # "anonymous is fine". A CI runner with no `hf auth login` would raise on
    # every call with that fallback. `or False` also covers GHA setting
    # HF_TOKEN to an empty string (rather than omitting it) when the secret
    # doesn't exist, which `.get(..., True)` alone would not catch.
    token: str | bool = os.environ.get("HF_TOKEN") or False
    api: HfApi = HfApi(token=token)
    curated_ids: list[str] = (
        load_curated_model_ids(curated_file) if curated_file else []
    )
    return build_catalog(
        fetch_fn=lambda lim: _fetch(api, lim),
        gates=GENERATIVE_GATES,
        limit=limit,
        output_csv=output_csv,
        label="generative",
        token=token,
        curated_ids=curated_ids,
        api=api,
    )


if __name__ == "__main__":
    limit_: int = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    fetch_top_generative_models(
        limit=limit_, output_csv=RESOURCES_DIR / "top_generative_models.csv"
    )
