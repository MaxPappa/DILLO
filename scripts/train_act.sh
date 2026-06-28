#!/usr/bin/env bash
set -euo pipefail

SUITE="${SUITE:-libero_goal}"
EXP_DIR="${EXP_DIR:-experiments_act}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"

python -m dillo.policy.train_act \
  --suite "$SUITE" \
  --exp_dir "$EXP_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --chunk_size "$CHUNK_SIZE" \
  --seed "$SEED" \
  --device "$DEVICE" \
  "$@"

