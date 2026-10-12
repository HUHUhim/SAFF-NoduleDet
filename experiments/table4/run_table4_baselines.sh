#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/core${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${CONFIG:-configs/p0_server.yaml}"
FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"
EPOCHS="${EPOCHS:-80}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/SAFF_ws/SAFF_revision/outputs_revision}"

python -m saff_p0.check_data --config "$CONFIG"
python -m saff_p0.run_experiment \
  --config "$CONFIG" \
  --experiment table4_baselines \
  --models saff_noduledet unet3d swin_unetr convnext3d cotr3d \
  --folds $FOLDS \
  --epochs "$EPOCHS" --output-root "$OUT_ROOT"
python -m saff_p0.summarize --config "$CONFIG" --experiment table4_baselines --output-root "$OUT_ROOT"
