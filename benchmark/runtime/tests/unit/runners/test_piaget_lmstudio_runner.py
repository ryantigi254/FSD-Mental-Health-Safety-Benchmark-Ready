"""Unit tests for Piaget LM Studio wiring."""

from __future__ import annotations

from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
FACTORY_PATH = RUNTIME_ROOT / "src" / "reliable_clinical_benchmark" / "models" / "factory.py"


def test_factory_accepts_piaget_lmstudio_alias():
    factory_source = FACTORY_PATH.read_text(encoding="utf-8")

    assert "piaget_lmstudio" in factory_source
