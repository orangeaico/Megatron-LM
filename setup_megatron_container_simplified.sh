#!/usr/bin/env bash
set -euo pipefail

# Install CCE if missing
python - <<'PY' || pip install --no-deps "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"
import importlib
import cut_cross_entropy  # noqa
print("cce ok")
PY

# Patch CCE to add temperature scaling instead of softcapping 
python - <<'PY'
import ast
import os
from pathlib import Path
from typing import Dict, Optional

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