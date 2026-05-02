#!/usr/bin/env bash
set -euo pipefail

# Run the requested SFT experiments on Qwen2.5-Math-1.5B.
# Designed for a single T4 GPU runtime.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TRAIN_JSONL="${TRAIN_JSONL:-/data/a5-alignment/MATH/sft.jsonl}"
VALID_JSONL="${VALID_JSONL:-/data/a5-alignment/MATH/validation.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft_math_sweep}"

if [[ ! -f "$TRAIN_JSONL" ]]; then
  echo "Missing train file: $TRAIN_JSONL"
  echo "Set TRAIN_JSONL to your /data/a5-alignment/MATH/sft.jsonl path."
  exit 1
fi

if [[ ! -f "$VALID_JSONL" ]]; then
  echo "Missing validation file: $VALID_JSONL"
  echo "Set VALID_JSONL to your /data/a5-alignment/MATH/validation.jsonl path."
  exit 1
fi

# Keep run length bounded for ~30 minutes on T4.
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-120}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-20}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-128}"
MAX_LENGTH="${MAX_LENGTH:-768}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUM_STEPS="${GRADIENT_ACCUM_STEPS:-16}"

# The tuning grid is intentionally small for T4 budget.
TUNE_LRS="${TUNE_LRS:-5e-6,1e-5,2e-5}"
TUNE_ACCUM="${TUNE_ACCUM:-8,16}"

mkdir -p "$OUTPUT_DIR"

uv run python -m cs336_alignment.sft_experiment \
  --model-name-or-path "Qwen/Qwen2.5-Math-1.5B" \
  --train-jsonl-path "$TRAIN_JSONL" \
  --validation-jsonl-path "$VALID_JSONL" \
  --prompt-template-path "cs336_alignment/prompts/r1_zero.prompt" \
  --output-dir "$OUTPUT_DIR" \
  --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUM_STEPS" \
  --max-length "$MAX_LENGTH" \
  --max-train-steps "$MAX_TRAIN_STEPS" \
  --eval-every-steps "$EVAL_EVERY_STEPS" \
  --max-eval-examples "$MAX_EVAL_EXAMPLES" \
  --run-size-sweep \
  --sweep-sizes "128,256,512,1024" \
  --run-full-tuning \
  --tune-learning-rates "$TUNE_LRS" \
  --tune-accum-steps "$TUNE_ACCUM" \
  --run-filtered-experiment \
  --use-vllm-eval \
  --policy-device cuda:0 \
  --vllm-device cuda:0 \
  --vllm-gpu-memory-utilization 0.80 \
  --disable-wandb \
  --run-name "qwen15b_math_sft_t4"

echo "Done. Report written to: $OUTPUT_DIR/sweep_report.json"
