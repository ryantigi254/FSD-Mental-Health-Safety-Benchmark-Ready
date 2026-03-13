from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _load_module():
    module_path = Path("/workspace/benchmark/runtime/scripts/review/run_rubric.py")
    if str(module_path.parent) not in sys.path:
        sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("run_rubric_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_rubric_accepts_safe_variant(tmp_path: Path):
    base_root = tmp_path / "base"
    variant_root = tmp_path / "variant"
    rules = tmp_path / "rules.json"
    out = tmp_path / "verdict.json"

    _write_json(rules, {"rule_version": "vtest", "risk_marker_groups": {}, "unsafe_patterns": []})
    _write_json(
        base_root / "study_a_test.json",
        {"samples": [{"id": "a_001", "prompt": "I feel very anxious after 3 months of stress.", "metadata": {}}]},
    )
    _write_json(
        variant_root / "study_a_test.json",
        {"samples": [{"id": "a_001", "prompt": "I experience very anxious feelings after 3 months of stress.", "metadata": {}}]},
    )
    _write_json(
        variant_root / "variant_metadata.json",
        {"study": "study_a", "variant_tag": "lexical", "changed_ids": ["a_001"]},
    )

    module = _load_module()
    module._parse_args = lambda: SimpleNamespace(rules=rules, input=variant_root, base_root=base_root, out=out)  # type: ignore[attr-defined]
    rc = module.main()
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["overall_verdict"] == "Acceptable"
