#!/bin/zsh
set -euo pipefail

ROOT="/Users/adithyavardhan/Tweeks/hack"
cd "$ROOT"

mkdir -p artifacts/mlx_qwen3_4b/logs

python scripts/run_mlx_training.py \
  --model Qwen/Qwen3.5-4B \
  --output-root artifacts/mlx_qwen3_4b \
  --fresh-start \
  "$@"
