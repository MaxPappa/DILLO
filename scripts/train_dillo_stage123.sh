#!/usr/bin/env bash
set -euo pipefail

: "${ACT_CHECKPOINT:?Set ACT_CHECKPOINT to the ACT .pth checkpoint}"

SUITE="${SUITE:-goal}"
SUITE_SHORT="${SUITE#libero_}"
SUITE_FULL="libero_${SUITE_SHORT}"
MODEL_NAME="${MODEL_NAME:-google/gemma-3-1b-it}"
MODEL_TAG="${MODEL_NAME##*/}"
TRAIN_DATA="${TRAIN_DATA:-data/${SUITE_FULL}_video_and_obs/*/*}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/dillo_${SUITE_SHORT}_${MODEL_TAG}}"
CONFIG="${CONFIG:-configs/dillo_stage1.yaml}"
DEVICE="${DEVICE:-cuda}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
NUM_CHUNKS="${NUM_CHUNKS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-0.0002}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-3}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-3}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-3}"
WANDB_PROJECT="${WANDB_PROJECT:-DILLO}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

COMMON_ARGS=(
  --config "$CONFIG"
  --model_name "$MODEL_NAME"
  --suite "$SUITE_SHORT"
  --train_data "$TRAIN_DATA"
  --act_checkpoint "$ACT_CHECKPOINT"
  --chunk_size "$CHUNK_SIZE"
  --num_chunks "$NUM_CHUNKS"
  --batch_size "$BATCH_SIZE"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --lr "$LR"
  --device "$DEVICE"
  --wandb_project "$WANDB_PROJECT"
  --no_timestamp_subdir
)

if [[ -n "$WANDB_ENTITY" ]]; then
  COMMON_ARGS+=(--wandb_entity "$WANDB_ENTITY")
fi

python -m dillo.training.train_dillo \
  "${COMMON_ARGS[@]}" \
  --stage stage1 \
  --epochs "$STAGE1_EPOCHS" \
  --output_dir "$RUN_DIR/stage1/latentobs"

python -m dillo.training.train_dillo \
  "${COMMON_ARGS[@]}" \
  --stage stage2 \
  --epochs "$STAGE2_EPOCHS" \
  --output_dir "$RUN_DIR/stage2/latentobs"

python -m dillo.training.train_dillo \
  "${COMMON_ARGS[@]}" \
  --stage stage3 \
  --use_verdict_head \
  --epochs "$STAGE3_EPOCHS" \
  --output_dir "$RUN_DIR/stage3/latentobs"

