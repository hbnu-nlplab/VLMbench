#!/usr/bin/env bash
# Evaluate Qwen2-VL on LACE-Bench. Override knobs via env vars or pass extra flags.
#
# Examples:
#   bash scripts/eval.sh
#   ADAPTER=./outputs/.../checkpoint-50000 bash scripts/eval.sh
#   bash scripts/eval.sh --knowledge-edit --use-cf-ke
#   bash scripts/eval.sh --qual-anal

set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2-VL-2B-Instruct}
ADAPTER=${ADAPTER:-}

cd "$(dirname "$0")/.."

ADAPTER_FLAG=()
if [[ -n "$ADAPTER" ]]; then
    ADAPTER_FLAG=(--adapter-path "$ADAPTER")
fi

python run_qwen2_vl.py eval \
    --model-name-or-path "$MODEL" \
    "${ADAPTER_FLAG[@]}" \
    --include-bbox \
    --task-b \
    "$@"
