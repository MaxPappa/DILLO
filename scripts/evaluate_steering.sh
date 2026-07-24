#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

: "${ACT_CHECKPOINT:?Set ACT_CHECKPOINT to the ACT .pth checkpoint}"

SUITE="${SUITE:-libero_goal}"
EXPLAINER_CKPTDIR="${EXPLAINER_CKPTDIR:-}"
MLP_CKPT="${MLP_CKPT:-}"
STEERING_MODEL_TYPE="${STEERING_MODEL_TYPE:-explainer}"
MODEL_NAME="${MODEL_NAME:-google/gemma-3-1b-it}"
OUTDIR="${OUTDIR:-outputs/steering/${SUITE}}"
N_EPISODES="${N_EPISODES:-20}"
PROPOSAL_BATCH_SIZE="${PROPOSAL_BATCH_SIZE:-8}"
MAX_REFUSAL_ATTEMPTS="${MAX_REFUSAL_ATTEMPTS:-8}"
NUM_PARALLEL="${NUM_PARALLEL:-1}"
PRINT_DECISION_LOGS="${PRINT_DECISION_LOGS:-0}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

ARGS=(
  --suite "$SUITE"
  --act-checkpoint "$ACT_CHECKPOINT"
  --steering-model-type "$STEERING_MODEL_TYPE"
  --n-episodes "$N_EPISODES"
  --proposal-batch-size "$PROPOSAL_BATCH_SIZE"
  --max-refusal-attempts "$MAX_REFUSAL_ATTEMPTS"
  --num-parallel "$NUM_PARALLEL"
  --explainer-model-name "$MODEL_NAME"
  --device "$DEVICE"
  --seed "$SEED"
  --outdir "$OUTDIR"
)

if [[ "$STEERING_MODEL_TYPE" == "explainer" ]]; then
  : "${EXPLAINER_CKPTDIR:?Set EXPLAINER_CKPTDIR for explainer steering}"
  ARGS+=(--explainer-ckptdir "$EXPLAINER_CKPTDIR")
else
  : "${MLP_CKPT:?Set MLP_CKPT for MLP steering}"
  ARGS+=(--mlp-ckpt "$MLP_CKPT")
fi

if [[ "$PRINT_DECISION_LOGS" == "1" ]]; then
  ARGS+=(--print-decision-logs)
else
  ARGS+=(--no-print-decision-logs)
fi

python -m dillo.evaluation.steering "${ARGS[@]}" "$@"
