#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

MODEL_NAME="${MODEL_NAME:-google/gemma-3-1b-it}"
MODEL_SHORT="${MODEL_NAME##*/}"
RELEASE_MODEL_DIR="${MODEL_SHORT//-/_}"
STAGE="${STAGE:-stage3}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/release/dillo_${RELEASE_MODEL_DIR}}"
ACT_ROOT="${ACT_ROOT:-checkpoints/release/act_policies}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/validation/${MODEL_SHORT}}"
VAL_SPLITS_DIR="${VAL_SPLITS_DIR:-val_splits}"
SUITES="${SUITES:-10 spatial goal object}"
USE_VERDICT_HEAD="${USE_VERDICT_HEAD:-1}"
RESUME="${RESUME:-1}"

suite_benchmark() {
  case "$1" in
    10) echo "LIBERO_10" ;;
    spatial) echo "LIBERO_SPATIAL" ;;
    goal) echo "LIBERO_GOAL" ;;
    object) echo "LIBERO_OBJECT" ;;
    90) echo "LIBERO_90" ;;
    *) echo "[ERROR] Unknown suite: $1" >&2; exit 1 ;;
  esac
}

checkpoint_dir() {
  local suite="$1"
  local benchmark
  benchmark="$(suite_benchmark "$suite")"
  local candidates=(
    "${CHECKPOINT_ROOT}/${benchmark}"
    "${CHECKPOINT_ROOT}/libero_${suite}"
    "${CHECKPOINT_ROOT}/dillo_${suite}_${MODEL_SHORT}/stage3/latentobs/latest"
    "${CHECKPOINT_ROOT}/dillo_${suite}_${MODEL_SHORT}/stage3/latentobs"
    "${CHECKPOINT_ROOT}/dillo_libero_${suite}_${MODEL_SHORT}/stage3/latentobs/latest"
    "${CHECKPOINT_ROOT}/dillo_libero_${suite}_${MODEL_SHORT}/stage3/latentobs"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if compgen -G "${candidate}/e=*_policy_explainer.pth" >/dev/null; then
      echo "$candidate"
      return
    fi
  done
  echo "[ERROR] Could not find checkpoint for libero_${suite} under ${CHECKPOINT_ROOT}" >&2
  exit 1
}

act_checkpoint() {
  local benchmark="$1"
  local candidates=(
    "${ACT_ROOT}/${benchmark}/best_model.pth"
    "${ACT_ROOT}/${benchmark}/ACT_chunk20_seed42/run_001/best_model.pth"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  done
  echo "[ERROR] Missing ACT checkpoint under ${ACT_ROOT}/${benchmark}" >&2
  exit 1
}

for suite in $SUITES; do
  benchmark="$(suite_benchmark "$suite")"
  CHECKPOINT_DIR="$(checkpoint_dir "$suite")"
  ACT_CHECKPOINT="$(act_checkpoint "$benchmark")"
  OUTPUT_DIR="${OUTPUT_ROOT}/libero_${suite}"

  echo
  echo "============================================================"
  echo "[Val outputs] model=${MODEL_NAME} suite=libero_${suite}"
  echo "  ckpt:       ${CHECKPOINT_DIR}"
  echo "  act ckpt:   ${ACT_CHECKPOINT}"
  echo "  output_dir: ${OUTPUT_DIR}"
  echo "============================================================"

  SUITE="$suite" \
  MODEL_NAME="$MODEL_NAME" \
  STAGE="$STAGE" \
  CHUNK_SIZE="$CHUNK_SIZE" \
  DEVICE="$DEVICE" \
  CHECKPOINT_DIR="$CHECKPOINT_DIR" \
  ACT_CHECKPOINT="$ACT_CHECKPOINT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  VAL_SPLITS_DIR="$VAL_SPLITS_DIR" \
  USE_VERDICT_HEAD="$USE_VERDICT_HEAD" \
  RESUME="$RESUME" \
    scripts/evaluate_validation.sh
done

echo
echo "[Metrics] Computing Fidelity T2O/T2T under ${OUTPUT_ROOT}"
python -m dillo.evaluation.metrics \
  --root_dir "$OUTPUT_ROOT" \
  --csv "$OUTPUT_ROOT/all_metrics.csv"

echo
echo "[Done] Validation outputs and metrics saved under ${OUTPUT_ROOT}"
