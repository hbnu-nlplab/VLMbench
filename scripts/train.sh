#!/usr/bin/env bash
# Train Qwen2-VL with LoRA. Override knobs via env vars or pass extra flags.
#
# Examples:
#   bash scripts/train.sh
#   MODEL=Qwen/Qwen2.5-VL-3B-Instruct bash scripts/train.sh
#   bash scripts/train.sh --no-task-b --use-cf
#   bash scripts/train.sh --resume-from-checkpoint ./outputs/.../checkpoint-60000

set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2-VL-2B-Instruct}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/${MODEL}_LRGR_bbox}
NUM_EPOCHS=${NUM_EPOCHS:-3}
LR=${LR:-2e-4}
SAVE_STEPS=${SAVE_STEPS:-10000}

cd "$(dirname "$0")/.."

python run_qwen2_vl.py train \
    --model-name-or-path "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --num-epochs "$NUM_EPOCHS" \
    --lr "$LR" \
    --save-steps "$SAVE_STEPS" \
    --include-bbox \
    --task-b \
    "$@"
