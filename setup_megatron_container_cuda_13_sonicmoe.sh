#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import torch
print("torch before setup:", torch.__version__)
try:
    import transformer_engine.pytorch  # noqa: F401
    print("transformer_engine before setup: ok")
except Exception as exc:
    raise SystemExit(
        "transformer_engine is broken before Megatron setup starts. "
        "Use a fresh NGC container or reinstall a Transformer Engine build that matches "
        f"the active torch ({torch.__version__}). Original error: {type(exc).__name__}: {exc}"
    ) from exc
PY

# Install CCE if missing
python - <<'PY' || pip install --no-deps "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"
import importlib
import cut_cross_entropy  # noqa
print("cce ok")
PY

# Patch CCE to add temperature scaling instead of softcapping
python - <<'PY'
import ast
from pathlib import Path
from typing import Dict

TARGET_FILES: Dict[Path, Dict[str, str]] = {
    Path("tl_utils.py"): {
        "tl_softcapping": "return v / softcap",
        "tl_softcapping_grad": "return dv / softcap",
    },
    Path("utils.py"): {
        "softcapping": "return logits / softcap",
    },
}

def locate_target(relative: Path) -> Path:
    try:
        import cut_cross_entropy
    except ImportError as exc:
        raise SystemExit(f"Failed to import cut_cross_entropy: {exc}") from exc
    module_path = Path(cut_cross_entropy.__file__).resolve()
    candidate = module_path.parent / relative
    if candidate.exists():
        return candidate
    raise SystemExit(f"Unable to locate {relative}")


def apply_replacements(target_path: Path, replacements: Dict[str, str]) -> bool:
    source_text = target_path.read_text()
    tree = ast.parse(source_text)
    lines = source_text.splitlines()
    updated = False

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in replacements:
            if not node.body:
                continue
            start = node.body[0].lineno - 1
            end = node.body[-1].end_lineno - 1
            indent = " " * node.body[0].col_offset
            replacement_line = f"{indent}{replacements[node.name]}"
            if lines[start : end + 1] != [replacement_line]:
                lines[start : end + 1] = [replacement_line]
                updated = True

    if updated:
        target_path.write_text("\n".join(lines) + "\n")
    return updated


for relative_path, replacements in TARGET_FILES.items():
    target_file = locate_target(relative_path)
    if apply_replacements(target_file, replacements):
        print(f"Updated {target_file}")
    else:
        print(f"No updates required for {target_file}")
PY

if [[ "${INSTALL_SONICMOE:-1}" == "1" ]]; then
  python - <<'PY' || {
from sonicmoe.enums import ActivationType  # noqa: F401
from sonicmoe.functional import moe_general_routing_inputs  # noqa: F401
print("sonic-moe ok")
PY
    # Do not let sonic-moe resolve its torch dependency inside the NGC image.
    # Replacing torch without rebuilding Transformer Engine causes ABI errors such as:
    # undefined symbol: c10::cuda::CUDAStream::query().
    pip install --no-cache-dir --upgrade --no-deps "sonic-moe"
    pip install --no-cache-dir --upgrade --no-deps "quack-kernels[cu13]>=0.4.0"
    pip install --no-cache-dir --upgrade --no-deps \
      "nvidia-cutlass-dsl>=4.4.0" \
      "nvidia-cutlass-dsl-libs-base>=4.4.0"
  }
  python - <<'PY'
from sonicmoe.enums import ActivationType  # noqa: F401
from sonicmoe.functional import moe_general_routing_inputs  # noqa: F401
print("sonic-moe ok")
PY
fi

pip install --no-cache-dir --upgrade --no-deps omegaconf==2.3.0 transformers==5.7.0
python - <<'PY' || pip install --no-cache-dir --upgrade "antlr4-python3-runtime==4.9.3"
import antlr4  # noqa: F401
print("antlr4 runtime ok")
PY
python - <<'PY' || pip install --no-cache-dir --ignore-installed --no-deps "PyYAML>=5.1.0"
import yaml  # noqa: F401
print("PyYAML ok")
PY

python - <<'PY'
import omegaconf  # noqa: F401
print("omegaconf ok")
PY

python - <<'PY'
import torch
import transformer_engine.pytorch  # noqa: F401
from sonicmoe.enums import ActivationType  # noqa: F401
from sonicmoe.functional import _DownProjection, _UpProjection  # noqa: F401
print("torch after setup:", torch.__version__)
print("transformer_engine after setup: ok")
print("sonic-moe direct kernels: ok")
PY
