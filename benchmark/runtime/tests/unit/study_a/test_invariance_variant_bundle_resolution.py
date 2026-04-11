from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = RUNTIME_ROOT / "hf-local-scripts"
SCRIPT_PATH = SCRIPT_DIR / "run_invariance_generate_only.py"
COMMON_PATH = SCRIPT_DIR / "_invariance_runner_common.py"


def _load_module(module_name: str, path: Path):
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_legacy_variant_bundle_root_resolves_all_matching_children(tmp_path: Path) -> None:
    module = _load_module("run_invariance_generate_only_test_module", SCRIPT_PATH)

    bundle_root = tmp_path / "v5_invariance_variants"
    lexical_root = bundle_root / "study_a_lexical"
    surface_root = bundle_root / "study_a_surface"
    lexical_root.mkdir(parents=True)
    surface_root.mkdir(parents=True)
    (lexical_root / "study_a_test.json").write_text("[]", encoding="utf-8")
    (surface_root / "study_a_test.json").write_text("[]", encoding="utf-8")

    resolved = module._resolve_data_targets("study_a_invariance", str(bundle_root))

    assert resolved == [("lexical", lexical_root), ("surface", surface_root)]


def test_v5_variant_tree_root_resolves_all_study_children(tmp_path: Path) -> None:
    module = _load_module("run_invariance_generate_only_v5_tree_test_module", SCRIPT_PATH)

    bundle_root = tmp_path / "v5"
    mild_root = bundle_root / "study_b" / "mild"
    strong_root = bundle_root / "study_b" / "strong"
    mild_root.mkdir(parents=True)
    strong_root.mkdir(parents=True)
    (bundle_root / "variant_matrix_manifest.json").write_text("{}", encoding="utf-8")
    (mild_root / "study_b_test.json").write_text("[]", encoding="utf-8")
    (strong_root / "study_b_test.json").write_text("[]", encoding="utf-8")

    resolved = module._resolve_data_targets("study_b_invariance", str(bundle_root))

    assert resolved == [("mild", mild_root), ("strong", strong_root)]


def test_study_level_family_root_resolves_all_children(tmp_path: Path) -> None:
    module = _load_module("run_invariance_generate_only_study_root_test_module", SCRIPT_PATH)

    study_root = tmp_path / "study_a"
    lexical_root = study_root / "lexical"
    syntax_root = study_root / "syntax"
    lexical_root.mkdir(parents=True)
    syntax_root.mkdir(parents=True)
    (lexical_root / "study_a_test.json").write_text("[]", encoding="utf-8")
    (syntax_root / "study_a_test.json").write_text("[]", encoding="utf-8")

    resolved = module._resolve_data_targets("study_a_invariance", str(study_root))

    assert resolved == [("lexical", lexical_root), ("syntax", syntax_root)]


def test_relative_output_dir_resolves_under_runtime_root() -> None:
    module = _load_module("run_invariance_generate_only_output_dir_test_module", SCRIPT_PATH)

    resolved = module._resolve_output_dir("results_invariance")

    assert resolved == module.RUNTIME_ROOT / "results_invariance"


def test_default_output_dir_uses_results_invariance_root() -> None:
    module = _load_module("run_invariance_generate_only_default_output_dir_test_module", SCRIPT_PATH)

    resolved = module._resolve_output_dir(None)

    assert resolved == module.RUNTIME_ROOT / "results_invariance"


def test_controllability_base_data_routes_to_base_output_subdir() -> None:
    module = _load_module("run_invariance_generate_only_base_output_test_module", SCRIPT_PATH)

    data_dir = module.RUNTIME_ROOT / "data" / "invariance_variants" / "controllability" / "base"
    resolved = module._resolve_effective_output_dir("results_invariance", data_dir)

    assert resolved == module.RUNTIME_ROOT / "results_invariance" / "base"


def test_controllability_variant_family_routes_to_variant_family_output_subdir() -> None:
    module = _load_module("run_invariance_generate_only_variant_family_output_test_module", SCRIPT_PATH)

    data_dir = (
        module.RUNTIME_ROOT
        / "data"
        / "invariance_variants"
        / "controllability"
        / "study_b"
        / "mild"
    )
    resolved = module._resolve_effective_output_dir("results_invariance", data_dir)

    assert resolved == module.RUNTIME_ROOT / "results_invariance" / "variant-family"


def test_gpt_oss_lmstudio_normalizes_to_canonical_results_dir_even_without_existing_folder(tmp_path: Path) -> None:
    module = _load_module("invariance_runner_common_model_dir_test_module", COMMON_PATH)

    legacy_dir = tmp_path / "gpt-oss-lmstudio"
    legacy_dir.mkdir()

    normalized = module.normalize_model_id_for_path("gpt_oss_lmstudio", tmp_path)

    assert normalized == "gpt-oss-20b"


def test_piaget_lmstudio_normalizes_to_piaget_local_results_dir(tmp_path: Path) -> None:
    module = _load_module("invariance_runner_common_piaget_model_dir_test_module", COMMON_PATH)

    legacy_dir = tmp_path / "piaget-8b-local"
    legacy_dir.mkdir()

    normalized = module.normalize_model_id_for_path("piaget_lmstudio", tmp_path)

    assert normalized == "piaget-8b-local"
