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

# Install Transformers if missing
python - <<'PY' || pip install -U "transformers"
import importlib
import transformers  # noqa
try:
    from transformers import __version__ as v
except Exception:
    v = "unknown"
print("transformers ok:", v)
PY

# Clear old JIT caches (optional)
rm -rf ~/.cache/torch/inductor ~/.triton || true

unset PIP_CONSTRAINT
pip install --upgrade --no-cache-dir   "dill<0.3.9,>=0.3.0"   "datasets>=2.20.0"   "fsspec>=2024.6.1"   "huggingface_hub>=0.24.0"   "pyarrow>=12"
pip install jsonlines
pip install simpy
