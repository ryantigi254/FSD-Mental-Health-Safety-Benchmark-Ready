from __future__ import annotations

import importlib.util
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = RUNTIME_ROOT / "scripts" / "studies" / "study_c" / "metrics" / "calculate_metrics.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("study_c_metrics_test_module", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_target_plans_path_accepts_frozen_snapshot_alias(tmp_path: Path):
    root = tmp_path / "frozen"
    study_c_dir = root / "study_c"
    study_c_dir.mkdir(parents=True, exist_ok=True)
    alias_path = study_c_dir / "study_c_target_plans.json"
    alias_path.write_text("{}", encoding="utf-8")

    module = _load_module()
    resolved = module.resolve_target_plans_path(root, study_c_dir)

    assert resolved == alias_path
