#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

bash experiments/table4/run_table4_baselines.sh
bash experiments/table6/run_table6_ablation.sh
bash experiments/table7/run_table7_sensitivity.sh

echo "Training experiments finished. Candidate-sampling evaluation requires completed main checkpoints."

