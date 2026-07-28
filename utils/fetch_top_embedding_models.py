"""Fetch the top embedding models from Hugging Face, ranked by downloads.

Embedding (sentence/text-embedding) models on the Hub almost always carry
exactly one ``pipeline_tag`` — and for the most popular models (BGE, E5,
Qwen3-Embedding, mxbai, jina) that primary tag is ``feature-extraction``, NOT
``sentence-similarity``. To get full coverage we therefore query BOTH pipeline
tags and merge.

``feature-extraction`` is noisy: it also surfaces audio encoders (wav2vec2,
hubert, mimi, encodec, ...), vision encoders (clip, vit, ...) and rerankers.
We separate genuine text embedders from this noise by requiring a
"sentence-transformers signal" — the model is published with
``library_name == "sentence-transformers"`` or carries the
``sentence-similarity`` / ``sentence-transformers`` tag. This single filter
removes essentially all pure audio/vision/reranker entries while keeping
transformers-library embedders such as ``jinaai/jina-embeddings-v3``.

Rerankers (cross-encoders) are excluded. Multimodal embedders (jina-clip,
jina-embeddings-v5-omni, ...) are kept but flagged via ``is_multimodal``.
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.hf_api import ModelInfo

from utils.hf_model_catalog import (
    CURATED_EMBEDDING_MODELS_FILE,
    EXPAND_FIELDS,
    REASON_NO_CONFIG,
    REASON_NO_LOADABLE_WEIGHTS,
    REASON_NON_NATIVE_FORMAT,
    REASON_NOT_AN_EMBEDDER,
    REASON_NSFW,
    REASON_REMOTE_CODE,
    REASON_RERANKER,
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
    tags,
    with_transient_retry,
)
from utils.utilities import ts

# Pipeline tags that embedding models are filed under. They are mutually
# exclusive (one primary tag per model), so we query both and union.
EMBEDDING_PIPELINE_TAGS: tuple[str, ...] = ("feature-extraction", "sentence-similarity")

# Substrings (in model_type, architectures or tags) that mark a model as
# multimodal — i.e. it consumes images/audio/video in addition to (or instead
# of) text. These are kept but flagged, not dropped.
MULTIMODAL_SUBSTRINGS: list[str] = [
    "clip",
    "vision",
    "_vl",
    "vl_",
    "vit",
    "blip",
    "audio",
    "wav2vec",
    "hubert",
    "wavlm",
    "mimi",
    "encodec",
    "whisper",
    "speecht5",
    "sew",
    "unispeech",
    "data2vec-audio",
    "videoprism",
    "video",
    "owlvit",
    "groupvit",
    "_omni",
    "omni",
]

# Markers that identify rerankers / cross-encoders (excluded entirely — they
# are not bi-encoder embedders).
RERANKER_SUBSTRINGS: list[str] = ["rerank", "cross-encoder", "cross_encoder"]


def _has_embedding_signal(model: ModelInfo) -> bool:
    """True if the model looks like a sentence-transformers / embedding model.

    This is the core inclusion gate: it separates genuine text embedders from
    the audio/vision/reranker noise that shares the feature-extraction tag.
    """
    if model.library_name == "sentence-transformers":
        return True
    t: set[str] = tags(model)
    return "sentence-similarity" in t or "sentence-transformers" in t


def _is_reranker(model: ModelInfo) -> bool:
    """True if the model is a reranker / cross-encoder (excluded)."""
    if any(any(sub in t for sub in RERANKER_SUBSTRINGS) for t in tags(model)):
        return True
    return any(sub in model.id.lower() for sub in RERANKER_SUBSTRINGS)


def _is_multimodal(model: ModelInfo, _config_class: str | None = None) -> bool:
    """True if the embedder also handles images/audio/video (flagged, kept)."""
    config: dict = model.config or {}
    model_type: str = (config.get("model_type") or "").lower()
    if any(sub in model_type for sub in MULTIMODAL_SUBSTRINGS):
        return True

    architectures: list[str] = config.get("architectures") or []
    arch_lower: str = " ".join(architectures).lower()
    if any(sub in arch_lower for sub in MULTIMODAL_SUBSTRINGS):
        return True

    multimodal_tags: set[str] = {
        "image-feature-extraction",
        "multimodal",
        "vision",
        "audio",
    }
    return bool(tags(model) & multimodal_tags)


def _fetch(api: HfApi, limit: int) -> list[ModelInfo]:
    """Query both embedding pipeline tags and return a deduplicated list,
    sorted by downloads descending. Over-fetched (x2) to absorb the noise +
    rerankers + GGUF/MLX entries removed by the filter.

    Each per-tag call is wrapped in ``with_transient_retry`` so a mid-fetch
    504 from the HF gateway does not abort the run.
    """
    print(f"{ts()} Fetching top {limit} text-embedding models by downloads...")
    per_tag_limit: int = int(limit * 2)
    by_id: dict[str, ModelInfo] = {}
    for tag in EMBEDDING_PIPELINE_TAGS:
        print(f"Fetching up to {per_tag_limit} '{tag}' models by downloads...")
        models: list[ModelInfo] = with_transient_retry(
            lambda t=tag: api.list_models(
                pipeline_tag=t,
                sort="downloads",
                limit=per_tag_limit,
                expand=EXPAND_FIELDS,
            ),
            description=f"list_models[{tag}]",
        )
        for m in models:
            # First tag wins on dupes; they carry identical metadata anyway.
            by_id.setdefault(m.id, m)

    return sorted(by_id.values(), key=lambda m: (m.downloads or 0), reverse=True)


# Ordered gates for the embedding fetcher. THE ORDER IS LOAD-BEARING — see the
# equivalent note in fetch_top_generative_models.py. The two embedding-specific
# gates sit in the cheap block (both answer from tags/library_name/id alone),
# ahead of the two networked gates.
#
# REASON_NOT_AN_EMBEDDER is by far the largest bucket here (~32% of candidates,
# measured). It is a genuinely mixed set: real text embedders that simply lack a
# sentence-transformers signal (BAAI/bge-small-en, allenai/specter2_base) sit
# alongside the audio/vision encoders that share the feature-extraction pipeline
# tag (wav2vec2, encodec, clap) and placeholder repos. That mix is exactly why
# these are now recorded instead of dropped — the bucket is worth auditing.
EMBEDDING_GATES: tuple[Gate, ...] = (
    (REASON_NO_CONFIG, lambda m, _t: has_config(m)),
    (REASON_NON_NATIVE_FORMAT, lambda m, _t: is_native_format(m)),
    (REASON_NSFW, lambda m, _t: is_not_nsfw(m)),
    (REASON_NOT_AN_EMBEDDER, lambda m, _t: _has_embedding_signal(m)),
    (REASON_RERANKER, lambda m, _t: not _is_reranker(m)),
    # --- everything below costs one network call per candidate ---
    (REASON_REMOTE_CODE, lambda m, t: not contains_remote_code(m, t)),
    (REASON_NO_LOADABLE_WEIGHTS, lambda m, t: has_loadable_weights(m, t)),
)


def keep(model: ModelInfo, token: str | bool) -> bool:
    """True if *model* passes every gate. Thin shim over diagnose().

    Retained so callers that only want a bool verdict keep working; the
    fetchers themselves use diagnose() directly so they can record the cause.
    """
    return diagnose(model, EMBEDDING_GATES, token) is None


def fetch_top_embedding_models(
    limit: int,
    output_csv: Path | str | None = None,
    curated_file: Path | None = CURATED_EMBEDDING_MODELS_FILE,
) -> list[dict[str, object]]:
    """Fetch the top-*limit* embedding models plus the curated list.

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
        gates=EMBEDDING_GATES,
        limit=limit,
        output_csv=output_csv,
        label="embedding",
        extra_columns=[("is_multimodal", _is_multimodal)],
        allow_millions=True,
        token=token,
        curated_ids=curated_ids,
        api=api,
    )


if __name__ == "__main__":
    limit_: int = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    fetch_top_embedding_models(
        limit=limit_, output_csv=RESOURCES_DIR / "top_embedding_models.csv"
    )
