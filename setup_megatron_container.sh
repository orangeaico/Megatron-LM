#!/usr/bin/env bash
set -euo pipefail

# Always use your CUDA 12.9 ptxas (needed for sm_120)
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# Install Triton 3.3.x if not already
python - <<'PY' || pip install --no-deps --force-reinstall "triton==3.3.*"
import importlib.metadata as m
v = m.version("triton")
assert v.split(".")[0:2] >= ["3","3"]
print("triton ok:", v)
PY

pip install transformers

# Clear old JIT caches (optional)
rm -rf ~/.cache/torch/inductor ~/.triton || true

