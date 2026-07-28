#!/usr/bin/env python3
"""
Fetch the weekly top-K generative and embedding model lists once, then split
each into fixed-size shards for parallel GitHub Actions jobs.

Mirrors the pattern in generate_test_matrix.py: this script does the one-time
fetch/compute step and emits a JSON matrix via GITHUB_OUTPUT for a downstream
`strategy.matrix.include` job. Each shard is written to its own JSON file
under --output-dir; downstream jobs load their shard via
`weekly_test.py --model-list-file`.

Rows come back in two flavours (see utils/hf_model_catalog.build_catalog):
testable models, and models the fetcher rejected with a named
``rejection_reason``. Both are sharded, but separately — a rejected shard is a
pure sink write with no model work, so packing those rows into a few large
shards keeps dozens of Spyre-pod jobs from being scheduled to do nothing.

Usage (called by the GHA workflow):
    python .github/scripts/generate_weekly_shards.py \
        --top-k 10000 \
        --shard-size-generative 250 \
        --shard-size-embedding 500 \
        --output-dir shards
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add the project root to the Python path so we can import from utils/
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.fetch_top_embedding_models import fetch_top_embedding_models  # noqa: E402
from utils.fetch_top_generative_models import fetch_top_generative_models  # noqa: E402

MODES = ("generative", "embedding")

# Rejected rows are recorded, not tested: weekly_test.py writes each one
# straight to the sink and never spawns a worker, so the per-row cost is a dict
# lookup plus a buffered insert. Sharding them at the testable sizes (250/500)
# would add ~30 Spyre-pod matrix jobs whose entire runtime is checkout +
# uv sync + a ClickHouse handshake, each occupying a contended accelerator to
# do no model work. Packing them into a few big shards keeps that overhead at
# ~3 jobs while leaving the testable shards' sizing — and therefore their
# per-shard runtime — completely unchanged.
DEFAULT_SHARD_SIZE_REJECTED: int = 5000


def _chunk(rows: list[dict], shard_size: int) -> list[list[dict]]:
    """Split *rows* into consecutive sub-lists of length *shard_size* (the
    last shard may be shorter). Empty input yields zero shards.
    """
    return [rows[i : i + shard_size] for i in range(0, len(rows), shard_size)]


def generate_shards(
    top_k: int,
    shard_size_generative: int,
    shard_size_embedding: int,
    output_dir: Path,
    shard_size_rejected: int = DEFAULT_SHARD_SIZE_REJECTED,
) -> list[dict]:
    """Fetch both mode's lists once, write shard JSON files, and return the
    combined matrix (list of {mode, kind, shard_index, shard_file} dicts).

    ``kind`` is "testable" or "rejected"; the workflow surfaces it in the job
    name so a run's matrix is legible at a glance.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fetchers = {
        "generative": (fetch_top_generative_models, shard_size_generative),
        "embedding": (fetch_top_embedding_models, shard_size_embedding),
    }

    matrix: list[dict] = []
    for mode, (fetch_fn, shard_size) in fetchers.items():
        rows: list[dict] = fetch_fn(limit=top_k)
        # model_info is a live huggingface_hub.ModelInfo object attached by
        # build_catalog — not JSON-serializable, and no longer needed since
        # is_moe is precomputed onto each row (see utils/hf_model_catalog.py).
        for row in rows:
            row.pop("model_info", None)

        testable: list[dict] = [r for r in rows if not r.get("rejection_reason")]
        rejected: list[dict] = [r for r in rows if r.get("rejection_reason")]
        curated_count: int = sum(1 for r in rows if r.get("curated"))

        print(
            f"{mode}: fetched {len(rows)} row(s) — {len(testable)} testable, "
            f"{len(rejected)} rejected, {curated_count} curated"
        )

        for kind, kind_rows, kind_size in (
            ("testable", testable, shard_size),
            ("rejected", rejected, shard_size_rejected),
        ):
            shards = _chunk(kind_rows, kind_size)
            if not shards:
                continue
            print(
                f"    {kind}: {len(kind_rows)} row(s) -> {len(shards)} shard(s) "
                f"of up to {kind_size} each"
            )
            for shard_index, shard_rows in enumerate(shards):
                shard_file = f"{mode}-{kind}-shard-{shard_index:03d}.json"
                (output_dir / shard_file).write_text(json.dumps(shard_rows))
                matrix.append(
                    {
                        "mode": mode,
                        "kind": kind,
                        "shard_index": shard_index,
                        "shard_file": shard_file,
                    }
                )

    return matrix


def write_github_output(outputs: dict[str, str]) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        print("Not running in GitHub Actions. Output would be:")
        for key, value in outputs.items():
            print(f"{key}={value}")
        return

    with open(github_output, "a") as f:
        for key, value in outputs.items():
            f.write(f"{key}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-k",
        type=int,
        default=10000,
        help="Number of top models to fetch per mode (by downloads).",
    )
    parser.add_argument(
        "--shard-size-generative",
        type=int,
        default=250,
        help="Models per generative shard.",
    )
    parser.add_argument(
        "--shard-size-embedding",
        type=int,
        default=500,
        help="Models per embedding shard.",
    )
    parser.add_argument(
        "--shard-size-rejected",
        type=int,
        default=DEFAULT_SHARD_SIZE_REJECTED,
        help=(
            "Models per rejected shard. Much larger than the testable sizes "
            "because these rows are recorded, not tested — no worker is "
            "spawned for them."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("shards"),
        help="Directory to write shard JSON files into.",
    )
    args = parser.parse_args()

    matrix = generate_shards(
        top_k=args.top_k,
        shard_size_generative=args.shard_size_generative,
        shard_size_embedding=args.shard_size_embedding,
        output_dir=args.output_dir,
        shard_size_rejected=args.shard_size_rejected,
    )

    print(f"\nTotal shards across both modes: {len(matrix)}")
    write_github_output({"matrix": json.dumps(matrix)})


if __name__ == "__main__":
    main()
