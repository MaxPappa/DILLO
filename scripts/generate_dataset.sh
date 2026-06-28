#!/usr/bin/env bash
set -euo pipefail

: "${ACT_CHECKPOINT:?Set ACT_CHECKPOINT to the ACT .pth checkpoint}"
: "${VLM_MODEL:?Set VLM_MODEL to the model served by vLLM}"

SUITE="${SUITE:-libero_goal}"
SAVE_DIR="${SAVE_DIR:-data/${SUITE}_video_and_obs}"
DEVICE="${DEVICE:-cuda}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
CAMERA_SIZE="${CAMERA_SIZE:-512}"
VLM_MAX_FRAMES="${VLM_MAX_FRAMES:-8}"
MAX_CONCURRENT_VLM="${MAX_CONCURRENT_VLM:-10}"

python -m dillo.data_generation.collect_dataset \
  --checkpoint "$ACT_CHECKPOINT" \
  --suite "$SUITE" \
  --save_dir "$SAVE_DIR" \
  --vlm_model "$VLM_MODEL" \
  --vlm_host "$VLLM_HOST" \
  --vlm_port "$VLLM_PORT" \
  --device "$DEVICE" \
  --episodes_per_task "$EPISODES_PER_TASK" \
  --camera_size "$CAMERA_SIZE" \
  --send_mode timestamped_frames \
  --vlm_max_frames "$VLM_MAX_FRAMES" \
  --max_concurrent_vlm "$MAX_CONCURRENT_VLM" \
  --resume \
  "$@"

