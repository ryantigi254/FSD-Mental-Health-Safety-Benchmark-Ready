from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = RUNTIME_ROOT / "hf-local-scripts"
SCRIPT_PATH = SCRIPT_DIR / "run_invariance_generate_only.py"


def _load_module(module_name: str, path: Path):
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_variant_bundle_root_resolves_to_study_specific_child(tmp_path: Path) -> None:
    module = _load_module("run_invariance_generate_only_test_module", SCRIPT_PATH)

    bundle_root = tmp_path / "v5_invariance_variants"
    child_root = bundle_root / "study_a_lexical"
    child_root.mkdir(parents=True)
    (child_root / "study_a_test.json").write_text("[]", encoding="utf-8")

    resolved = module._resolve_data_dir("study_a_invariance", str(bundle_root))

    assert resolved == child_root
