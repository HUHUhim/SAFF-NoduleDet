#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/core${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${CONFIG:-configs/supplement_server_10fold.yaml}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/SAFF_ws/SAFF_revision/outputs_revision}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"
EPOCHS="${EPOCHS:-80}"

run_condition() {
  local name="$1"; shift
  python -m saff_p0.run_experiment --config "$CONFIG" --output-root "$OUT_ROOT" \
    --experiment "$name" --models saff_noduledet --variants full \
    --folds $FOLDS --epochs "$EPOCHS" "$@"
}

for r in 3 5 7; do
  run_condition "region_r${r}" --regional-regions "$r" --distill-weight 0.10 --smooth-weight 0.005
done
for w in 0.05 0.10 0.20; do
  run_condition "distill_w_${w}" --regional-regions 5 --distill-weight "$w" --smooth-weight 0.005
done
for w in 0 0.005 0.010; do
  run_condition "smooth_w_${w}" --regional-regions 5 --distill-weight 0.10 --smooth-weight "$w"
done

for exp in region_r3 region_r5 region_r7 distill_w_0.05 distill_w_0.10 distill_w_0.20 smooth_w_0 smooth_w_0.005 smooth_w_0.010; do
  python -m saff_p0.summarize --config "$CONFIG" --experiment "$exp" --output-root "$OUT_ROOT"
done

python experiments/table7/summarize_table7.py \
  --results-root "$OUT_ROOT" \
  --output-dir "$OUT_ROOT/summaries/table7"
