#!/usr/bin/env bash
set -euo pipefail

: "${ROOT_DIR:?Set ROOT_DIR to the directory containing prediction JSONs}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
CONCURRENCY="${CONCURRENCY:-64}"

python -m dillo.evaluation.llm_fuzzy \
  --root_dir "$ROOT_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --concurrency "$CONCURRENCY" \
  "$@"

