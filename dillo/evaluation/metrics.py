#!/usr/bin/env python3
"""
Compute evaluation metrics on saved prediction outputs.

Reads per-item JSON prediction files produced by the DILLO validation or
baseline evaluators and computes:
  - Fidelity T2O (EEF Directional Fidelity): checks whether directional
    words in the predicted text match the actual EEF movement direction
    using the saved eef_pos_before / eef_pos_after coordinates.
  - Fidelity T2T: directional consistency between prediction and
    ground-truth description.
  - (Optional) BLEU, ROUGE, BERTScore when --slow is passed.

Usage:
    # Single output directory
    python -m dillo.evaluation.metrics \
        --predictions_dir outputs/validation/libero_spatial/gemma-3-1b-it/stage3_latest

    # Compare multiple directories and print a table
    python -m dillo.evaluation.metrics \
        --predictions_dir \
            outputs/validation/libero_spatial/gemma-3-1b-it/stage3_latest \
            outputs/baselines/libero_spatial/gemma-3-1b-it \
        --slow

    # Auto-discover ALL runs under a root folder and save all_metrics.csv
    python -m dillo.evaluation.metrics \
        --root_dir outputs/validation \
        --csv outputs/validation/all_metrics.csv

Output:
    Prints a table to stdout, saves metrics.json inside each predictions_dir,
    and (with --root_dir) saves all_metrics.csv in the root directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Import evaluation utilities ───────────────────────────────────────
from dillo.evaluation.eval_utils import (
    eef_fidelity_single,
    fidelity_text_to_text,
    compute_eef_fidelity,
    compute_fidelity_t2t,
    evaluate_text_generation,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def is_prediction_dir(path: Path) -> bool:
    """Return True if `path` directly contains numbered prediction JSONs."""
    return any(re.fullmatch(r"\d{6}\.json", f.name) for f in path.glob("*.json"))


def discover_prediction_dirs(root: Path) -> List[Path]:
    """
    Recursively find all directories under `root` that directly contain
    numbered prediction JSON files (000000.json, 000001.json, …).
    """
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        if is_prediction_dir(p):
            found.append(p)
            dirnames.clear()  # don't recurse into a prediction dir
    return sorted(found)


def parse_run_id(path: Path, root: Optional[Path] = None) -> Tuple[str, str, str]:
    """
    Parse (suite, model_tag, run_name) from a prediction directory path.

    Expected layouts (relative to root):
      libero_{suite}/{model_tag}/              # baseline
      libero_{suite}/{model_name}/{run_tag}/   # trained model
    """
    if root is not None:
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
    else:
        rel = path

    parts = rel.parts
    suite = parts[0] if parts else str(path)
    model_tag = "/".join(parts[1:]) if len(parts) > 1 else str(path)
    run_name = str(rel)
    return suite, model_tag, run_name


def load_records(predictions_dir: str) -> List[dict]:
    """Load all per-item JSON records from a predictions directory."""
    p = Path(predictions_dir)
    files = sorted(p.glob("*.json"))
    # Exclude summary.json and metrics.json
    files = [f for f in files if f.name not in ("summary.json", "metrics.json")]
    records = []
    for f in files:
        try:
            with open(f) as fh:
                records.append(json.load(fh))
        except Exception as e:
            print(f"  WARNING: could not read {f}: {e}")
    return records


def compute_metrics_for_records(
    records: List[dict],
    compute_slow: bool = False,
) -> Dict[str, float]:
    """
    Compute all metrics for a list of prediction records.

    Each record must have (both field naming conventions are supported):
        description_pred  OR  response        : str  (generated description)
        description_gt    OR  gt_description  : str  (ground-truth description)
        eef_pos_before    : list[float] length 3
        eef_pos_after     : list[float] length 3
        gripper_before    : list[float] length 2
        gripper_after     : list[float] length 2
    """
    # Support both field naming conventions:
    #   trained model   → description_pred / description_gt
    #   baselines       → response / gt_description
    predictions = [
        r.get("description_pred") or r.get("response", "") for r in records
    ]
    references = [
        r.get("description_gt") or r.get("gt_description", "") for r in records
    ]

    # Build obs array: (N, 2, 12)
    obs_list = []
    for r in records:
        eef_b   = np.array(r.get("eef_pos_before", [0, 0, 0]), dtype=np.float32)  # (3,)
        eef_a   = np.array(r.get("eef_pos_after",  [0, 0, 0]), dtype=np.float32)  # (3,)
        grip_b  = np.array(r.get("gripper_before", [0, 0]),    dtype=np.float32)  # (2,)
        grip_a  = np.array(r.get("gripper_after",  [0, 0]),    dtype=np.float32)  # (2,)
        # Reconstruct (2, 12): eef(3) + joint(7, zeros) + gripper(2)
        # joint positions are not in the records but are not needed for fidelity
        before = np.concatenate([eef_b, np.zeros(7, dtype=np.float32), grip_b])  # (12,)
        after  = np.concatenate([eef_a, np.zeros(7, dtype=np.float32), grip_a])  # (12,)
        obs_list.append(np.stack([before, after]))  # (2, 12)

    obs_np = np.stack(obs_list, axis=0) if obs_list else None  # (N, 2, 12)

    metrics = evaluate_text_generation(
        predictions=predictions,
        references=references,
        obs=obs_np,
        compute_slow_metrics=compute_slow,
    )

    # Rename to be explicit about what "fidelity_eef" means for the paper
    if "fidelity_eef" in metrics:
        metrics["fidelity_t2o"] = metrics.pop("fidelity_eef")

    # Add basic counts
    metrics["n_items"] = len(records)
    metrics["n_empty_pred"] = sum(1 for p in predictions if not p.strip())

    return metrics


def print_metrics_table(results: Dict[str, Dict[str, float]]):
    """Print a comparison table of metrics across multiple runs."""
    all_keys = set()
    for m in results.values():
        all_keys.update(m.keys())

    # Order columns
    order = ["fidelity_t2o", "fidelity_t2t", "bleu",
             "rouge_rouge1", "rouge_rouge2", "rouge_rougeL", "bert_f1",
             "n_items", "n_empty_pred"]
    display_keys = [k for k in order if k in all_keys]
    display_keys += sorted(all_keys - set(display_keys))

    header = f"{'Run':<55}" + "".join(f"{k:>16}" for k in display_keys)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for run_name, metrics in results.items():
        name_trunc = run_name[-55:].ljust(55)
        row = name_trunc
        for k in display_keys:
            val = metrics.get(k, float("nan"))
            if isinstance(val, int):
                row += f"{val:>16d}"
            else:
                row += f"{val:>16.4f}"
        print(row)

    print("=" * len(header) + "\n")


def save_csv(
    results: Dict[str, Dict],
    run_meta: Dict[str, Tuple[str, str]],  # run_name → (suite, model_tag)
    csv_path: Path,
):
    """Save all metrics to a CSV file with suite and model_tag columns."""
    all_metric_keys: List[str] = []
    seen = set()
    order = ["fidelity_t2o", "fidelity_t2t", "bleu",
             "rouge_rouge1", "rouge_rouge2", "rouge_rougeL", "bert_f1",
             "n_items", "n_empty_pred"]
    for k in order:
        if any(k in m for m in results.values()):
            all_metric_keys.append(k)
            seen.add(k)
    # Append any remaining keys in sorted order
    extra = sorted({k for m in results.values() for k in m} - seen)
    all_metric_keys.extend(extra)

    fieldnames = ["suite", "model_tag"] + all_metric_keys

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run_name, metrics in results.items():
            suite, model_tag = run_meta.get(run_name, ("", run_name))
            row: dict = {"suite": suite, "model_tag": model_tag}
            for k in all_metric_keys:
                val = metrics.get(k, "")
                row[k] = f"{val:.4f}" if isinstance(val, float) else val
            writer.writerow(row)

    print(f"[compute_metrics] CSV saved → {csv_path}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics on saved predictions"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predictions_dir", nargs="+",
                       help="One or more prediction directories to evaluate")
    group.add_argument("--root_dir",
                       help="Root folder (e.g. xagent_outputs): auto-discovers "
                            "all prediction sub-directories and saves all_metrics.csv")
    parser.add_argument("--slow", action="store_true", default=False,
                        help="Also compute BLEU, ROUGE, BERTScore (slow)")
    parser.add_argument("--no_save", action="store_true", default=False,
                        help="Do not save metrics.json files")
    parser.add_argument("--csv", default=None,
                        help="Path to save a CSV of all results "
                             "(auto-set to <root_dir>/all_metrics.csv when using --root_dir)")
    args = parser.parse_args()

    # ── Determine which directories to evaluate ───────────────────────
    root: Optional[Path] = None
    if args.root_dir:
        root = Path(args.root_dir)
        pred_dirs = discover_prediction_dirs(root)
        if not pred_dirs:
            print(f"[compute_metrics] No prediction directories found under {root}")
            return
        print(f"[compute_metrics] Discovered {len(pred_dirs)} prediction dirs under {root}")
        csv_out = Path(args.csv) if args.csv else root / "all_metrics.csv"
    else:
        pred_dirs = [Path(d) for d in args.predictions_dir]
        csv_out = Path(args.csv) if args.csv else None

    all_results: Dict[str, Dict] = {}
    run_meta: Dict[str, tuple] = {}   # run_name → (suite, model_tag)

    for pred_dir in pred_dirs:
        print(f"\n[compute_metrics] Loading: {pred_dir}")
        records = load_records(str(pred_dir))

        if not records:
            print(f"  WARNING: no records found in {pred_dir}")
            continue

        print(f"  {len(records)} records found")
        metrics = compute_metrics_for_records(records, compute_slow=args.slow)

        suite, model_tag, run_name = parse_run_id(pred_dir, root)
        all_results[run_name] = metrics
        run_meta[run_name] = (suite, model_tag)

        for k, v in metrics.items():
            if isinstance(v, int):
                print(f"  {k:30s}: {v}")
            else:
                print(f"  {k:30s}: {v:.4f}")

        if not args.no_save:
            out_path = pred_dir / "metrics.json"
            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  Metrics saved → {out_path}")

    # ── Print table ───────────────────────────────────────────────────
    if len(all_results) > 1:
        print_metrics_table(all_results)
    elif all_results:
        run_name, metrics = next(iter(all_results.items()))
        print(f"\nResults for {run_name}:")
        for k, v in metrics.items():
            if isinstance(v, int):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v:.4f}")

    # ── Save CSV ──────────────────────────────────────────────────────
    if csv_out and all_results:
        save_csv(all_results, run_meta, csv_out)


if __name__ == "__main__":
    main()
