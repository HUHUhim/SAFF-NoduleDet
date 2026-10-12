#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/core${PYTHONPATH:+:$PYTHONPATH}"

RESULTS_ROOT="${RESULTS_ROOT:-/root/autodl-tmp/SAFF_ws/SAFF_revision/outputs_revision}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/root/autodl-tmp/SAFF_ws/SAFF/outputs_p0}"
SAMPLING_CSV="${SAMPLING_CSV:-/root/autodl-tmp/SAFF_ws/SAFF_revision/candidate_sampling/candidate_sampling_metrics.csv}"
OUT_DIR="${OUT_DIR:-$RESULTS_ROOT/analysis}"
EXPERIMENTS="${EXPERIMENTS:-region_r3 region_r5 region_r7 distill_w_0.05 distill_w_0.10 distill_w_0.20 smooth_w_0 smooth_w_0.005 smooth_w_0.010}"

python -m saff_p0.analyze_revision --results-root "$RESULTS_ROOT" \
  --experiments $EXPERIMENTS --reference-root "$REFERENCE_ROOT" \
  --output-dir "$OUT_DIR" --sampling-csv "$SAMPLING_CSV"

