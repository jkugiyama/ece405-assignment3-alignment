#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TRAIN_JSONL="${TRAIN_JSONL:-/data/a5-alignment/MATH/train.jsonl}"
VALID_JSONL="${VALID_JSONL:-/data/a5-alignment/MATH/validation.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ei_math_sweep}"

if [[ ! -f "$TRAIN_JSONL" ]]; then
  echo "Missing train file: $TRAIN_JSONL"
  exit 1
fi
if [[ ! -f "$VALID_JSONL" ]]; then
  echo "Missing validation file: $VALID_JSONL"
  exit 1
fi

N_EI_STEPS="${N_EI_STEPS:-5}"
ROLLOUT_COUNTS="${ROLLOUT_COUNTS:-4,8}"
SFT_EPOCHS_LIST="${SFT_EPOCHS_LIST:-1,2}"
DB_SIZES="${DB_SIZES:-512,1024,2048}"
MAX_CONFIGS="${MAX_CONFIGS:-4}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUM_STEPS="${GRADIENT_ACCUM_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda:0}"
VLLM_DEVICE="${VLLM_DEVICE:-cuda:0}"

if ! uv run python - <<'PY'
import torch
import importlib.util
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Run this on a GPU runtime (e.g., Colab T4).")
if importlib.util.find_spec("vllm") is None:
    raise SystemExit("vLLM is not installed in this environment.")
print("Runtime check passed: CUDA + vLLM available")
PY
then
  exit 1
fi

uv run python scripts/ei_math_sweep.py \
  --model-name-or-path "Qwen/Qwen2.5-Math-1.5B" \
  --train-jsonl-path "$TRAIN_JSONL" \
  --validation-jsonl-path "$VALID_JSONL" \
  --prompt-template-path "cs336_alignment/prompts/r1_zero.prompt" \
  --output-dir "$OUTPUT_DIR" \
  --n-ei-steps "$N_EI_STEPS" \
  --rollout-counts "$ROLLOUT_COUNTS" \
  --sft-epochs-list "$SFT_EPOCHS_LIST" \
  --db-sizes "$DB_SIZES" \
  --max-configs "$MAX_CONFIGS" \
  --learning-rate "$LEARNING_RATE" \
  --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUM_STEPS" \
  --policy-device "$POLICY_DEVICE" \
  --vllm-device "$VLLM_DEVICE"

echo "EI sweep done: $OUTPUT_DIR/sweep_summary.json"
