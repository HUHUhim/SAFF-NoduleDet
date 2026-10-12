#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/core${PYTHONPATH:+:$PYTHONPATH}"
RESULTS_ROOT="${RESULTS_ROOT:-/root/autodl-tmp/SAFF_ws/SAFF_revision/outputs_revision}"
OUT_DIR="${OUT_DIR:-$RESULTS_ROOT/figures}"
python figures/make_manuscript_figures.py --results-root "$RESULTS_ROOT" --out-dir "$OUT_DIR"
