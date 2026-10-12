#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/core${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${CONFIG:-configs/supplement_server_10fold.yaml}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/autodl-tmp/SAFF_ws/SAFF/outputs_p0}"
CHECKPOINT_EXPERIMENT="${CHECKPOINT_EXPERIMENT:-table4_baselines}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/SAFF_ws/SAFF_revision/candidate_sampling}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"
COUNTS="${COUNTS:-1200 2400 0}"
REPEATS="${REPEATS:-3}"

python -m saff_p0.evaluate_checkpoint --config "$CONFIG" \
  --checkpoint-root "$CHECKPOINT_ROOT" --checkpoint-experiment "$CHECKPOINT_EXPERIMENT" \
  --model saff_noduledet --variant full --folds $FOLDS --counts $COUNTS \
  --repeats "$REPEATS" --output-root "$OUT_ROOT"

