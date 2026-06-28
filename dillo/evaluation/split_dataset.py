#!/usr/bin/env python3
"""
Generate and save the validation split for each LIBERO suite.

This script replicates the EXACT same train/val split that was used during
training (seed=42, 90/10 split at the item level) and writes a JSON file
containing all validation items together with their metadata.

The JSON can then be consumed by libero_baselines.py and eval_trained_model.py
without having to re-derive the split.

Usage:
    # Single suite
    python -m dillo.evaluation.split_dataset \
        --suite 90 \
        --data_dir "data/libero_90_video_and_obs/*/*" \
        --chunk_size 20 \
        --output_dir val_splits

    # All suites at once (with defaults)
    python -m dillo.evaluation.split_dataset --all_suites

Output:
    val_splits/libero_{suite}_val.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dillo.evaluation.libero_val_dataset import build_libero_val_items, val_items_to_dicts

# ──────────────────────────────────────────────────────────────────────
# Default per-suite configurations (must match training configs)
# ──────────────────────────────────────────────────────────────────────
SUITE_DEFAULTS = {
    "spatial": {
        "data_dir": "data/libero_spatial_video_and_obs/*/*",
        "chunk_size": 20,
        "tokenizer": "google/gemma-3-1b-it",
    },
    "goal": {
        "data_dir": "data/libero_goal_video_and_obs/*/*",
        "chunk_size": 20,
        "tokenizer": "google/gemma-3-1b-it",
    },
    "object": {
        "data_dir": "data/libero_object_video_and_obs/*/*",
        "chunk_size": 20,
        "tokenizer": "google/gemma-3-1b-it",
    },
    "10": {
        "data_dir": "data/libero_10_video_and_obs/*/*",
        "chunk_size": 20,
        "tokenizer": "google/gemma-3-1b-it",
    },
    "90": {
        "data_dir": "data/libero_90_video_and_obs/*/*",
        "chunk_size": 20,
        "tokenizer": "google/gemma-3-1b-it",
    },
}


def generate_split(
    suite: str,
    data_dir: str,
    chunk_size: int,
    tokenizer_name: str,
    output_dir: str,
    split_seed: int = 42,
    val_fraction: float = 0.1,
    min_description_tokens: int = 5,
    force: bool = False,
):
    out_path = Path(output_dir) / f"libero_{suite}_val.json"

    if out_path.exists() and not force:
        print(f"[split_dataset] Val split already exists at {out_path}. "
              "Use --force to overwrite.")
        return

    print(f"\n{'='*60}")
    print(f"Suite: libero_{suite}")
    print(f"Data : {data_dir}")
    print(f"Seed : {split_seed}  |  val_fraction: {val_fraction}")
    print(f"{'='*60}")

    val_items = build_libero_val_items(
        data_dirs=data_dir,
        split_seed=split_seed,
        val_fraction=val_fraction,
        chunk_size=chunk_size,
        min_description_tokens=min_description_tokens,
        tokenizer_name=tokenizer_name,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = val_items_to_dicts(val_items)

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"[split_dataset] Saved {len(records)} val items → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate LIBERO validation splits matching training splits"
    )
    parser.add_argument(
        "--suite", type=str, default=None,
        choices=list(SUITE_DEFAULTS.keys()),
        help="Single suite to process. Omit and use --all_suites to process all."
    )
    parser.add_argument(
        "--all_suites", action="store_true",
        help="Process all configured suites."
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Glob pattern for episode folders (overrides default for --suite)."
    )
    parser.add_argument(
        "--chunk_size", type=int, default=None,
        help="Actions per chunk (overrides default for --suite)."
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="HuggingFace tokenizer name (overrides default for --suite)."
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="val_splits",
        help="Directory where val JSON files are saved."
    )
    parser.add_argument(
        "--split_seed", type=int, default=42,
        help="Random seed for the split (must match training; default 42)."
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.1,
        help="Fraction of examples held out as validation (default 0.1)."
    )
    parser.add_argument(
        "--min_description_tokens", type=int, default=5,
        help="Minimum token count for a description to be included."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing split files."
    )

    args = parser.parse_args()

    if args.all_suites:
        suites = list(SUITE_DEFAULTS.keys())
    elif args.suite is not None:
        suites = [args.suite]
    else:
        parser.error("Provide --suite <name> or --all_suites.")

    for suite in suites:
        defaults = SUITE_DEFAULTS[suite]
        generate_split(
            suite=suite,
            data_dir=args.data_dir or defaults["data_dir"],
            chunk_size=args.chunk_size or defaults["chunk_size"],
            tokenizer_name=args.tokenizer or defaults["tokenizer"],
            output_dir=args.output_dir,
            split_seed=args.split_seed,
            val_fraction=args.val_fraction,
            min_description_tokens=args.min_description_tokens,
            force=args.force,
        )


if __name__ == "__main__":
    main()
