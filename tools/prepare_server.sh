#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/core${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${CONFIG:-configs/p0_server.yaml}"

mkdir -p /root/autodl-tmp/SAFF_ws/SAFF

python - <<'PY'
import sys
print("python", sys.version)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_device", torch.cuda.get_device_name(0))
        print("cuda_runtime", torch.version.cuda)
except Exception as exc:
    print("torch_import_error", exc)
    raise
PY

python -m pip install -e .
python -m pip install -r requirements.txt

python -m saff_p0.check_data --config "$CONFIG"
