#!/bin/zsh
set -euo pipefail

ROOT="/Users/adithyavardhan/Tweeks/hack"
cd "$ROOT"

python scripts/prepare_mlx_data.py --force
mkdir -p artifacts/mlx_qwen3_4b/logs artifacts/mlx_qwen3_4b/metrics artifacts/mlx_qwen3_4b/adapters

python -m mlx_lm lora \
  --model Qwen/Qwen3.5-4B \
  --train \
  --data "$ROOT/artifacts/mlx_qwen3_4b/data" \
  --mask-prompt \
  --num-layers 8 \
  --batch-size 1 \
  --iters 10 \
  --val-batches 2 \
  --learning-rate 5e-5 \
  --steps-per-report 1 \
  --steps-per-eval 1000 \
  --save-every 10 \
  --grad-accumulation-steps 8 \
  --grad-checkpoint \
  --adapter-path "$ROOT/artifacts/mlx_qwen3_4b/adapters" \
  --max-seq-length 1024 \
  > "$ROOT/artifacts/mlx_qwen3_4b/logs/mlx_lora_benchmark.log" 2>&1

python scripts/save_mlx_speed.py
