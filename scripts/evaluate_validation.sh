#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a DILLO checkpoint directory}"

SUITE="${SUITE:-goal}"
SUITE_SHORT="${SUITE#libero_}"
MODEL_NAME="${MODEL_NAME:-google/gemma-3-1b-it}"
STAGE="${STAGE:-stage3}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
DEVICE="${DEVICE:-cuda}"
VAL_SPLITS_DIR="${VAL_SPLITS_DIR:-val_splits}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/validation}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
USE_VERDICT_HEAD="${USE_VERDICT_HEAD:-1}"
USE_RAW_OBS="${USE_RAW_OBS:-0}"
ACT_CHECKPOINT="${ACT_CHECKPOINT:-}"
RESUME="${RESUME:-1}"

OBS_TYPE="latentobs"
if [[ "$USE_RAW_OBS" == "1" ]]; then
  OBS_TYPE="rawobs"
fi
CKPT_TAG="$(basename "$CHECKPOINT_DIR")"
if [[ "$CKPT_TAG" == "latentobs" || "$CKPT_TAG" == "rawobs" || "$CKPT_TAG" == "imageobs" ]]; then
  CKPT_TAG="$(basename "$(dirname "$CHECKPOINT_DIR")")"
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$OUTPUT_ROOT/libero_${SUITE_SHORT}/${MODEL_NAME##*/}/${STAGE}_${OBS_TYPE}_${CKPT_TAG}"
fi

ARGS=(
  --suite "$SUITE_SHORT"
  --stage "$STAGE"
  --checkpoint_dir "$CHECKPOINT_DIR"
  --model_name "$MODEL_NAME"
  --chunk_size "$CHUNK_SIZE"
  --device "$DEVICE"
  --val_splits_dir "$VAL_SPLITS_DIR"
  --output_root "$OUTPUT_ROOT"
  --output_dir "$OUTPUT_DIR"
)

if [[ "$USE_VERDICT_HEAD" == "1" ]]; then
  ARGS+=(--use_verdict_head)
fi
if [[ "$RESUME" == "1" ]]; then
  ARGS+=(--resume)
fi
if [[ "$USE_RAW_OBS" == "1" ]]; then
  ARGS+=(--use_raw_obs)
elif [[ -n "$ACT_CHECKPOINT" ]]; then
  ARGS+=(--act_checkpoint "$ACT_CHECKPOINT")
fi

python -m dillo.evaluation.validate "${ARGS[@]}" "$@"

python -m dillo.evaluation.metrics --predictions_dir "$OUTPUT_DIR"
