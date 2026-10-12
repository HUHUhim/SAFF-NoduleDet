#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/core${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${CONFIG:-configs/p0_server.yaml}"

python -m saff_p0.run_experiment \
  --config "$CONFIG" \
  --experiment smoke \
  --models saff_noduledet unet3d \
  --folds 0 \
  --epochs 1 \
  --train-negatives-per-positive 1 \
  --eval-negatives-per-scan 20 \
  --max-train-samples 24 \
  --max-eval-scans 4 \
  --force

python -m saff_p0.summarize --config "$CONFIG" --experiment smoke
