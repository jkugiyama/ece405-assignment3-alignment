#!/usr/bin/env bash
# Run GRPO training on MATH using a T4 GPU (Colab / hosted runtime).
# Requires: CUDA, vLLM, the A5 MATH dataset at /data/a5-alignment/MATH/.

set -euo pipefail

# ---- runtime preflight ----
python - <<'EOF'
import sys
import torch
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available. This script requires a GPU.", file=sys.stderr)
    sys.exit(1)
try:
    import vllm
except ImportError:
    print("ERROR: vllm is not installed. Run: pip install vllm", file=sys.stderr)
    sys.exit(1)
print(f"CUDA OK: {torch.cuda.get_device_name(0)}")
print(f"vllm OK: {vllm.__version__}")
EOF

# ---- configurable defaults ----
MODEL="${MODEL:-Qwen/Qwen2.5-Math-1.5B}"
TRAIN_JSONL="${TRAIN_JSONL:-/data/a5-alignment/MATH/train.jsonl}"
VALID_JSONL="${VALID_JSONL:-/data/a5-alignment/MATH/validation.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo_math}"
LOSS_TYPE="${LOSS_TYPE:-grpo_clip}"

ROLLOUT_G="${ROLLOUT_G:-8}"                # responses per question
QUESTIONS_PER_STEP="${QUESTIONS_PER_STEP:-16}"
ROLLOUT_TEMP="${ROLLOUT_TEMP:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
CLIPRANGE="${CLIPRANGE:-0.2}"
LR="${LR:-5e-6}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200}"
EVAL_EVERY="${EVAL_EVERY:-20}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-256}"
VLLM_MEM="${VLLM_MEM:-0.80}"
SEED="${SEED:-42}"

# ---- dataset existence checks ----
for F in "$TRAIN_JSONL" "$VALID_JSONL"; do
    if [ ! -f "$F" ]; then
        echo "ERROR: Required file not found: $F" >&2
        exit 1
    fi
done

echo "=== GRPO Training ==="
echo "  model:             $MODEL"
echo "  train:             $TRAIN_JSONL"
echo "  valid:             $VALID_JSONL"
echo "  loss_type:         $LOSS_TYPE"
echo "  rollout_g:         $ROLLOUT_G"
echo "  questions/step:    $QUESTIONS_PER_STEP"
echo "  max_train_steps:   $MAX_TRAIN_STEPS"
echo "  output_dir:        $OUTPUT_DIR"

uv run python cs336_alignment/grpo_train.py \
    --model-name-or-path "$MODEL" \
    --train-jsonl-path "$TRAIN_JSONL" \
    --validation-jsonl-path "$VALID_JSONL" \
    --output-dir "$OUTPUT_DIR" \
    --loss-type "$LOSS_TYPE" \
    --rollout-g "$ROLLOUT_G" \
    --questions-per-step "$QUESTIONS_PER_STEP" \
    --rollout-temperature "$ROLLOUT_TEMP" \
    --rollout-max-new-tokens "$MAX_NEW_TOKENS" \
    --cliprange "$CLIPRANGE" \
    --learning-rate "$LR" \
    --max-grad-norm "$MAX_GRAD_NORM" \
    --per-device-batch-size "$PER_DEVICE_BS" \
    --max-train-steps "$MAX_TRAIN_STEPS" \
    --eval-every-steps "$EVAL_EVERY" \
    --max-eval-examples "$MAX_EVAL_EXAMPLES" \
    --vllm-gpu-memory-utilization "$VLLM_MEM" \
    --seed "$SEED"

echo ""
echo "=== Plotting GRPO curves ==="
uv run python scripts/plot_grpo_curves.py --output-dir "$OUTPUT_DIR"

echo ""
echo "Done. Outputs in $OUTPUT_DIR/"
