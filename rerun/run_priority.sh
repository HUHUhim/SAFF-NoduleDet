#!/usr/bin/env bash
# Priority-tiered rerun. Folds that are already done are skipped automatically (run_experiment.py checks final_metrics.json),
# so the run can be interrupted and resumed at any time without paying twice.
#
# Usage:
#   TIER=1 bash rerun/run_priority.sh                 # tier 1 only (Table 4 SAFF+CoTr3D + the 4.3 tests)
#   TIER=2 bash rerun/run_priority.sh                 # tiers 1 and 2 (completing Table 4)
#   TIER=3 bash rerun/run_priority.sh                 # plus the Table 6 ablations
#   TIER=4 bash rerun/run_priority.sh                 # plus the Table 7 sensitivity runs
#   FOLDS="0 1" EPOCHS=2 TIER=1 bash rerun/run_priority.sh   # smoke: only 2 folds, 2 epochs, to calibrate speed
#   FORCE=1 ...                                       # overwrite existing results and rerun (use with care)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/core${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${CONFIG:-configs/rerun_4090_final.yaml}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/SAFF_ws/SAFF_rerun/outputs_rerun}"
TIER="${TIER:-1}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"
EPOCHS="${EPOCHS:-80}"
FORCE_FLAG=""
[ "${FORCE:-0}" = "1" ] && FORCE_FLAG="--force"

# Interpreter resolution: non-interactive bash does not read .bashrc, so a bare `python` is often not on PATH
# (observed in practice: launching via nohup reported `rerun/run_priority.sh: line 30: python: command not found`)
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ] || ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=""
  for c in /root/miniconda3/bin/python /opt/conda/bin/python /usr/bin/python3 python3 python; do
    if command -v "$c" >/dev/null 2>&1; then PYTHON="$c"; break; fi
  done
fi
if [ -z "$PYTHON" ]; then
  echo "Error: no python interpreter found; specify one explicitly, e.g. PYTHON=/root/miniconda3/bin/python"
  exit 127
fi
export PATH="$(dirname "$PYTHON"):$PATH"

run() {  # run <experiment> <models...> [--extra args]
  local exp="$1"; shift
  echo ""
  echo "================ $(date '+%F %T')  experiment=$exp  $(echo "$@" | tr '\n' ' ') ================"
  "$PYTHON" -m saff_p0.run_experiment \
    --config "$CONFIG" --experiment "$exp" --output-root "$OUT_ROOT" \
    --folds $FOLDS --epochs "$EPOCHS" $FORCE_FLAG "$@"
}

echo "Interpreter : $PYTHON"
echo "Config      : $CONFIG"
echo "Output root : $OUT_ROOT"
echo "Tier        : $TIER  folds: $FOLDS  epochs: $EPOCHS"

# ---------------- Tier 1: SAFF and CoTr3D of Table 4 (the minimal set for the 4.3 paired tests) ----------------
if [ "$TIER" -ge 1 ]; then
  run p0_main          --models saff_noduledet --variants full
  run table4_baselines --models cotr3d
fi

# ---------------- Tier 2: the remaining Table 4 baselines ----------------
if [ "$TIER" -ge 2 ]; then
  run table4_baselines --models unet3d convnext3d swin_unetr
fi

# ---------------- Tier 3: Table 6 ablations (full is already covered by tier 1) ----------------
if [ "$TIER" -ge 3 ]; then
  run table6_ablation --models saff_noduledet \
    --variants full no_distill no_morph_gate fixed_fusion global_norm cnn_only transformer_only full_attention
fi

# ---------------- Tier 4: Table 7 sensitivity (one separate experiment directory per setting, so they cannot overwrite each other) ----------------
if [ "$TIER" -ge 4 ]; then
  run region_r3        --models saff_noduledet --variants full --regional-regions 3
  run region_r5        --models saff_noduledet --variants full --regional-regions 5
  run region_r7        --models saff_noduledet --variants full --regional-regions 7
  run distill_w_0.05   --models saff_noduledet --variants full --distill-weight 0.05
  run distill_w_0.10   --models saff_noduledet --variants full --distill-weight 0.10
  run distill_w_0.20   --models saff_noduledet --variants full --distill-weight 0.20
  run smooth_w_0       --models saff_noduledet --variants full --smooth-weight 0.0
  run smooth_w_0.005   --models saff_noduledet --variants full --smooth-weight 0.005
  run smooth_w_0.010   --models saff_noduledet --variants full --smooth-weight 0.010
fi

echo ""
echo "All done: $(date '+%F %T')"
echo "Next: python analysis/make_results_package.py --runs-root $OUT_ROOT --out-dir <results package directory>"
