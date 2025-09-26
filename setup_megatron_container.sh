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

# Install CCE if missing
python - <<'PY' || pip install --no-deps "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"
import importlib
import cut_cross_entropy  # noqa
print("cce ok")
PY

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

# Patch transformer_engine to use flash_attn_interface instead of flash_attn_3.flash_attn_interface
python - <<'PY'
import pathlib, re, sys

FILEPATH = "/usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/attention.py"

p = pathlib.Path(FILEPATH)
if not p.exists():
    print(f"[patch] ERROR: file not found: {FILEPATH}", file=sys.stderr)
    sys.exit(2)

src = p.read_text(encoding="utf-8")

# Replace only leading import statements that reference flash_attn_3.flash_attn_interface
pat = re.compile(r'(?m)^(?P<i>\s*)(?P<kw>from|import)\s+flash_attn_3\.flash_attn_interface\b')
dst, n = pat.subn(r'\g<i>\g<kw> flash_attn_interface', src)

if n == 0:
    print(f"[patch] Nothing changed (already patched or different source): {FILEPATH}")
    sys.exit(0)

p.write_text(dst, encoding="utf-8")
print(f"[patch] Patched {n} line(s) in {FILEPATH}")
PY

# Install flash_attn_3
# pip install --no-index --no-deps https://huggingface.co/datasets/himanshu-livup/wheels/resolve/main/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl