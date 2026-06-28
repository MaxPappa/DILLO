#!/usr/bin/env python3
"""
LLM Fuzzy Matching for LIBERO prediction outputs.

For each per-item JSON produced by the DILLO validation or baseline evaluators,
asks an LLM to grade the predicted description against the ground-truth description
using atomic action decomposition (motion direction, gripper state, object behaviour).

Results are written back into each JSON file as:
  grading_score   : float in [0, 1]
  grading_output  : full structured LLM grading response

Handles both field naming conventions:
  baselines      → response       / gt_description
  trained model  → description_pred / description_gt

Usage:
    # All runs under a root folder (auto-discovers every prediction dir)
    python -m dillo.evaluation.llm_fuzzy --root_dir outputs/validation/libero_90

    # Single predictions directory
    python -m dillo.evaluation.llm_fuzzy \\
        --predictions_dir outputs/validation/libero_10/gemma-3-1b-it/stage3_latest

    # Custom LLM server
    python -m dillo.evaluation.llm_fuzzy --root_dir ... \\
        --host 127.0.0.1 --port 8000 --model Qwen/Qwen2.5-32B-Instruct

    # Re-grade (ignore already-graded files)
    python -m dillo.evaluation.llm_fuzzy --root_dir ... --no_resume
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI


# ──────────────────────────────────────────────────────────────────────
# Configuration (override via CLI flags)
# ──────────────────────────────────────────────────────────────────────

DEFAULT_HOST  = "127.0.0.1"
DEFAULT_PORT  = 8080
DEFAULT_MODEL = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_API_KEY = "password"
DEFAULT_CONCURRENCY = 64


# ──────────────────────────────────────────────────────────────────────
# Grading prompt
# ──────────────────────────────────────────────────────────────────────

GRADING_PROMPT = """\
You are a meticulous grader. Compare a REFERENCE description against a STUDENT \
description and score how well the STUDENT matches the REFERENCE at the level of \
atomic actions.

## What to evaluate
Only evaluate actions explicitly present in the REFERENCE. Do not invent extra actions \
and do not penalize the student for adding extra details; simply ignore details that are \
not required by the REFERENCE. The denominator is the number of REFERENCE actions you extract.

## Action extraction (from REFERENCE)
Break the REFERENCE into atomic actions, each expressing a single property. Typical categories:
- Lateral motion: left / right / none
- Depth motion: forward / backward / none
- Vertical motion: up / down / maintain height / none
- Gripper state: opening / closing / maintain openness; with optional modifier: slightly / moderately / greatly
- Object state: stationary / moving (+ direction if present)
- Intent/interaction: approaching / preparing for interaction / grasping / withdrawing, etc.

Normalize synonyms (e.g., "leftwards" = left; "advance" = forward; "going down" = down; \
"tightening" = closing). Treat negations carefully (e.g., "no change in height" = maintain height).

## Scoring each REFERENCE action
- 1.0 (correct): Same action/value.
- 0.5 (partial): Same general action but mismatch in magnitude/specificity; or imprecise near-match.
- 0.0 (incorrect): Contradictory or missing.

## Output format (JSON only — no prose before or after)
{{
  "frame_id": "{frame_id}",
  "reference_actions": [
    {{
      "action": "<short normalized action, e.g. 'move: left'>",
      "student_evidence": "<exact student phrase or 'missing'>",
      "judgement": "correct | partially correct | incorrect",
      "score": 1.0 | 0.5 | 0.0,
      "reason": "<one-sentence justification>"
    }}
  ],
  "summary": {{
    "num_reference_actions": <int>,
    "final_score": <float to 3 decimals>
  }}
}}

## Now grade this pair

FRAME_ID: {frame_id}

REFERENCE:
{reference}

STUDENT:
{student}"""


# ──────────────────────────────────────────────────────────────────────
# Discovery helpers  (same logic as compute_metrics.py)
# ──────────────────────────────────────────────────────────────────────

def is_prediction_dir(path: Path) -> bool:
    return any(re.fullmatch(r"\d{6}\.json", f.name) for f in path.glob("*.json"))


def discover_prediction_dirs(root: Path) -> List[Path]:
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        p = Path(dirpath)
        if is_prediction_dir(p):
            found.append(p)
            dirnames.clear()
    return sorted(found)


def parse_run_id(path: Path, root: Optional[Path]) -> Tuple[str, str, str]:
    """Return (suite, model_tag, run_name) relative to root."""
    if root is not None:
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
    else:
        rel = path
    parts = rel.parts
    suite     = parts[0] if parts else str(path)
    model_tag = "/".join(parts[1:]) if len(parts) > 1 else str(path)
    return suite, model_tag, str(rel)


# ──────────────────────────────────────────────────────────────────────
# JSON helpers
# ──────────────────────────────────────────────────────────────────────

def get_pred_ref(record: dict) -> Tuple[str, str]:
    pred = (record.get("description_pred") or record.get("response") or "").strip()
    ref  = (record.get("description_gt")   or record.get("gt_description") or "").strip()
    return pred, ref


def safe_parse_json(text: str, frame_id) -> Optional[dict]:
    text = text.strip()
    for marker in ("```json", "```"):
        if marker in text:
            text = text.split(marker, 1)[-1]
    text = text.replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        print(f"  [WARN] frame {frame_id}: could not parse JSON response")
        return None


# ──────────────────────────────────────────────────────────────────────
# Async grading
# ──────────────────────────────────────────────────────────────────────

async def grade_one(
    file_path: Path,
    client: AsyncOpenAI,
    model: str,
    sem: asyncio.Semaphore,
    resume: bool,
) -> Optional[float]:
    """Grade a single JSON file, writing result back in-place. Returns score or None."""
    try:
        with open(file_path) as f:
            record = json.load(f)
    except Exception as e:
        print(f"  [WARN] could not read {file_path}: {e}")
        return None

    # Resume: skip already-graded
    if resume and "grading_score" in record:
        return float(record["grading_score"])

    frame_id = record.get("index", file_path.stem)
    pred, ref = get_pred_ref(record)

    if not ref:
        return None
    if not pred:
        record["grading_score"] = 0.0
        record["grading_output"] = None
        with open(file_path, "w") as f:
            json.dump(record, f, indent=2)
        return 0.0

    prompt = GRADING_PROMPT.format(frame_id=frame_id, reference=ref, student=pred)

    async with sem:
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "You are a meticulous grader. Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            text = completion.choices[0].message.content
        except Exception as e:
            print(f"  [ERROR] frame {frame_id}: {e}")
            return None

    result = safe_parse_json(text, frame_id)
    if result is None:
        return None

    score = float(result.get("summary", {}).get("final_score", 0.0))
    record["grading_score"]  = score
    record["grading_output"] = result

    with open(file_path, "w") as f:
        json.dump(record, f, indent=2)

    return score


async def grade_dir(
    pred_dir: Path,
    client: AsyncOpenAI,
    model: str,
    sem: asyncio.Semaphore,
    resume: bool,
) -> List[float]:
    """Grade all prediction files in a directory. Returns list of scores."""
    files = sorted(
        f for f in pred_dir.glob("*.json")
        if f.name not in ("summary.json", "metrics.json")
    )
    tasks = [grade_one(f, client, model, sem, resume) for f in files]
    results = await tqdm_asyncio.gather(*tasks, desc=pred_dir.name, leave=False)
    return [r for r in results if r is not None]


# ──────────────────────────────────────────────────────────────────────
# Summary CSV
# ──────────────────────────────────────────────────────────────────────

def save_csv(rows: List[dict], csv_path: Path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[llm_fuzzy] CSV saved → {csv_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

async def run(args):
    client = AsyncOpenAI(
        base_url=f"http://{args.host}:{args.port}/v1",
        api_key=args.api_key,
    )
    sem = asyncio.Semaphore(args.concurrency)
    resume = not args.no_resume

    # ── Discover dirs ──────────────────────────────────────────────
    root: Optional[Path] = None
    if args.root_dir:
        root = Path(args.root_dir)
        pred_dirs = discover_prediction_dirs(root)
        csv_out = Path(args.csv) if args.csv else root / "llm_fuzzy_scores.csv"
        print(f"[llm_fuzzy] Discovered {len(pred_dirs)} prediction dirs under {root}")
    else:
        pred_dirs = [Path(d) for d in args.predictions_dir]
        csv_out = Path(args.csv) if args.csv else None

    summary_rows: List[dict] = []

    for pred_dir in pred_dirs:
        print(f"\n[llm_fuzzy] Grading: {pred_dir}")
        scores = await grade_dir(pred_dir, client, args.model, sem, resume)

        if not scores:
            print(f"  WARNING: no scores for {pred_dir}")
            continue

        avg = sum(scores) / len(scores)
        n_graded = len(scores)
        print(f"  avg_llm_fuzzy = {avg:.4f}  ({n_graded} items)")

        suite, model_tag, run_name = parse_run_id(pred_dir, root)
        summary_rows.append({
            "suite":        suite,
            "model_tag":    model_tag,
            "run_name":     run_name,
            "avg_llm_fuzzy": f"{avg:.4f}",
            "n_graded":     n_graded,
        })

        # Also persist per-dir summary
        dir_summary_path = pred_dir / "llm_fuzzy_summary.json"
        with open(dir_summary_path, "w") as f:
            json.dump({"avg_llm_fuzzy": avg, "n_graded": n_graded}, f, indent=2)

    if summary_rows:
        summary_rows.sort(key=lambda r: (r["suite"], r["model_tag"]))
        print("\n[llm_fuzzy] Summary:")
        for r in summary_rows:
            print(f"  {r['run_name']:60s}  avg={r['avg_llm_fuzzy']}  n={r['n_graded']}")
        if csv_out:
            save_csv(summary_rows, csv_out)


def main():
    parser = argparse.ArgumentParser(
        description="LLM Fuzzy Matching for LIBERO prediction outputs"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--root_dir",
                       help="Root folder (discovers all prediction sub-dirs)")
    group.add_argument("--predictions_dir", nargs="+",
                       help="One or more specific prediction directories")

    parser.add_argument("--host",        default=DEFAULT_HOST)
    parser.add_argument("--port",        type=int, default=DEFAULT_PORT)
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--api_key",     default=DEFAULT_API_KEY)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="Max parallel LLM calls (default: 64)")
    parser.add_argument("--no_resume",   action="store_true", default=False,
                        help="Re-grade all files even if already graded")
    parser.add_argument("--csv",         default=None,
                        help="Path for output CSV (default: <root_dir>/llm_fuzzy_scores.csv)")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
