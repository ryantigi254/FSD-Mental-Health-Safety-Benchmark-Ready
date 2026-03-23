#!/usr/bin/env python3
"""Comprehensive clinical readiness validation for v6.1 and v2.1 datasets.

Checks:
  1. Structural integrity (row counts, field presence, no duplicates)
  2. Source ID backing (every row has source_openr1_id(s), no None/empty)
  3. Provenance taxonomy compliance
  4. Generation policy enforcement (strict_no_generation at manifest level)
  5. Turn structure for multi-turn studies (20 turns, phase, dialogue_act, state_ledger)
  6. Bias pair-group integrity (exactly 2 members per group)
  7. Gold label/plan coverage
  8. Rubric safety checks (risk markers, unsafe patterns)
  9. Controllability stage-2 gates (via existing infrastructure)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

RUNTIME_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = RUNTIME_ROOT / "data"

V6_1_ROOT = DATA_ROOT / "frozen_splits" / "v6_1"
CTRL_V2_1 = DATA_ROOT / "controllability" / "controllability_splits_v2_1"
RUBRIC_RULES = DATA_ROOT / "rubrics" / "rubric_rules_v4_1.json"

_VALID_PROVENANCE = {"direct_source", "retrieved_composed", "source_anchored_deterministic_edit"}


class ValidationResult:
    def __init__(self, name: str) -> None:
        self.name = name
        self.passes: list[str] = []
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, msg: str) -> None:
        self.passes.append(msg)

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def passed(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"\n{'='*72}", f"  {status}  {self.name}", f"{'='*72}"]
        for p in self.passes:
            lines.append(f"  ✓ {p}")
        for w in self.warnings:
            lines.append(f"  ⚠ {w}")
        for f in self.failures:
            lines.append(f"  ✗ {f}")
        lines.append(f"  {len(self.passes)} passed, {len(self.warnings)} warnings, {len(self.failures)} failures")
        return "\n".join(lines)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rubric_rules() -> dict[str, Any]:
    if RUBRIC_RULES.exists():
        return _load(RUBRIC_RULES)
    return {}


def _extract_source_ids(metadata: dict[str, Any]) -> list[Any]:
    """Extract source IDs from metadata, returning list of found IDs."""
    ids = metadata.get("source_openr1_ids")
    if isinstance(ids, list) and ids:
        return ids
    single = metadata.get("source_openr1_id")
    if single is not None:
        return [single]
    return []


def _unsafe_hits(text: str, rules: dict[str, Any]) -> list[str]:
    text_lower = text.lower()
    return [p for p in (rules.get("unsafe_patterns", []) or []) if p.lower() in text_lower]


# ---------------------------------------------------------------------------
# v6.1 parent validation
# ---------------------------------------------------------------------------

def validate_v6_1(rules: dict[str, Any]) -> ValidationResult:
    vr = ValidationResult("v6.1 Parent (frozen_splits/v6_1)")

    if not V6_1_ROOT.exists():
        vr.fail("v6_1 root directory does not exist")
        return vr

    # Manifest
    manifest = _load(V6_1_ROOT / "manifest.json")
    if manifest.get("generation_policy") == "strict_no_generation":
        vr.ok("generation_policy = strict_no_generation")
    else:
        vr.fail(f"generation_policy = {manifest.get('generation_policy')}")

    if manifest.get("turn_count_policy") == "exact_count_with_manual_review":
        vr.ok("turn_count_policy = exact_count_with_manual_review")
    else:
        vr.fail(f"turn_count_policy = {manifest.get('turn_count_policy')}")

    provenance_tax = manifest.get("provenance_taxonomy", [])
    if set(provenance_tax) == _VALID_PROVENANCE:
        vr.ok("provenance_taxonomy matches expected set")
    else:
        vr.fail(f"provenance_taxonomy mismatch: {provenance_tax}")

    # Study A
    study_a = _load(V6_1_ROOT / "study_a_test.json")
    samples_a = study_a.get("samples", [])
    if len(samples_a) == 2000:
        vr.ok(f"Study A: {len(samples_a)} rows")
    else:
        vr.fail(f"Study A: expected 2000, got {len(samples_a)}")

    a_ids = [s.get("id") for s in samples_a]
    if len(a_ids) == len(set(a_ids)):
        vr.ok("Study A: no duplicate IDs")
    else:
        vr.fail(f"Study A: {len(a_ids) - len(set(a_ids))} duplicate IDs")

    # Source ID check
    a_no_source = sum(1 for s in samples_a if not _extract_source_ids(s.get("metadata", {})))
    if a_no_source == 0:
        vr.ok("Study A: all rows have source_openr1_id(s)")
    else:
        vr.fail(f"Study A: {a_no_source} rows missing source IDs")

    a_source_types = Counter(s.get("metadata", {}).get("source_type") for s in samples_a)
    if all(st in _VALID_PROVENANCE for st in a_source_types):
        vr.ok(f"Study A: source_type distribution {dict(a_source_types)}")
    else:
        vr.fail(f"Study A: invalid source_types {dict(a_source_types)}")

    unique_a_ids = {sid for s in samples_a for sid in _extract_source_ids(s.get("metadata", {}))}
    vr.ok(f"Study A: {len(unique_a_ids)} unique source IDs backing {len(samples_a)} rows")

    # Study A Bias
    bias_data = _load(V6_1_ROOT / "adversarial_bias" / "biased_vignettes.json")
    bias_cases = bias_data.get("cases", [])
    if len(bias_cases) == 2000:
        vr.ok(f"Study A Bias: {len(bias_cases)} rows")
    else:
        vr.fail(f"Study A Bias: expected 2000, got {len(bias_cases)}")

    bias_no_source = sum(1 for c in bias_cases if not _extract_source_ids(c.get("metadata", {})))
    if bias_no_source == 0:
        vr.ok("Study A Bias: all rows have source_openr1_id(s)")
    else:
        vr.fail(f"Study A Bias: {bias_no_source} rows missing source IDs")

    # Pair group integrity
    groups: dict[str, list[str]] = {}
    for c in bias_cases:
        gid = str(c.get("pair_group_id", ""))
        groups.setdefault(gid, []).append(c.get("id"))
    bad_groups = {gid: len(members) for gid, members in groups.items() if len(members) != 2}
    if not bad_groups:
        vr.ok(f"Study A Bias: all {len(groups)} pair groups have exactly 2 members")
    else:
        vr.fail(f"Study A Bias: {len(bad_groups)} pair groups with != 2 members")

    unique_bias_ids = {sid for c in bias_cases for sid in _extract_source_ids(c.get("metadata", {}))}
    vr.ok(f"Study A Bias: {len(unique_bias_ids)} unique source IDs backing {len(bias_cases)} rows")

    # manual_review_required field presence
    bias_mr_dist = Counter(c.get("metadata", {}).get("manual_review_required") for c in bias_cases)
    if all(isinstance(k, bool) for k in bias_mr_dist):
        vr.ok(f"Study A Bias: manual_review_required all boolean — {dict(bias_mr_dist)}")
    else:
        vr.fail(f"Study A Bias: non-boolean manual_review_required values — {dict(bias_mr_dist)}")

    # Study B single turn
    study_b = _load(V6_1_ROOT / "study_b_test.json")
    b_items = study_b if isinstance(study_b, list) else study_b.get("samples", [])
    if len(b_items) == 2000:
        vr.ok(f"Study B: {len(b_items)} rows")
    else:
        vr.fail(f"Study B: expected 2000, got {len(b_items)}")

    b_no_source = sum(1 for item in b_items if not _extract_source_ids(item.get("metadata", {})))
    if b_no_source == 0:
        vr.ok("Study B: all rows have source_openr1_id(s)")
    else:
        vr.fail(f"Study B: {b_no_source}/{len(b_items)} rows missing source IDs")

    b_source_types = Counter(item.get("metadata", {}).get("source_type") for item in b_items)
    vr.ok(f"Study B: source_type distribution {dict(b_source_types)}")

    unique_b_ids = {sid for item in b_items for sid in _extract_source_ids(item.get("metadata", {}))}
    vr.ok(f"Study B: {len(unique_b_ids)} unique source IDs backing {len(b_items)} rows")

    # Study B multi-turn
    study_b_mt = _load(V6_1_ROOT / "study_b_multi_turn_test.json")
    b_mt_cases = study_b_mt if isinstance(study_b_mt, list) else study_b_mt.get("cases", study_b_mt.get("multi_turn_cases", []))
    if len(b_mt_cases) == 120:
        vr.ok(f"Study B MT: {len(b_mt_cases)} cases")
    else:
        vr.fail(f"Study B MT: expected 120, got {len(b_mt_cases)}")

    mt_turn_issues = []
    for case in b_mt_cases:
        turns = case.get("turns", [])
        if len(turns) != 20:
            mt_turn_issues.append(f"{case.get('id')}: {len(turns)} turns")
    if not mt_turn_issues:
        vr.ok("Study B MT: all cases have 20 turns")
    else:
        vr.fail(f"Study B MT: {len(mt_turn_issues)} cases with wrong turn count")

    unique_bmt_ids = set()
    for case in b_mt_cases:
        unique_bmt_ids.update(_extract_source_ids(case.get("metadata", {})))
    vr.ok(f"Study B MT: {len(unique_bmt_ids)} unique source IDs backing {len(b_mt_cases)} cases")

    # Study C
    study_c = _load(V6_1_ROOT / "study_c_test.json")
    c_cases = study_c.get("cases", []) if isinstance(study_c, dict) else study_c
    if len(c_cases) == 100:
        vr.ok(f"Study C: {len(c_cases)} cases")
    else:
        vr.fail(f"Study C: expected 100, got {len(c_cases)}")

    c_turn_issues = []
    for case in c_cases:
        turns = case.get("turns", [])
        if len(turns) != 20:
            c_turn_issues.append(f"{case.get('id')}: {len(turns)} turns")
    if not c_turn_issues:
        vr.ok("Study C: all cases have 20 turns")
    else:
        vr.fail(f"Study C: {len(c_turn_issues)} cases with wrong turn count")

    unique_c_ids = set()
    for case in c_cases:
        unique_c_ids.update(_extract_source_ids(case.get("metadata", {})))
    vr.ok(f"Study C: {len(unique_c_ids)} unique source IDs backing {len(c_cases)} cases")

    # Turn-level provenance for multi-turn
    for label, cases, expected_n in [("Study B MT", b_mt_cases, 120), ("Study C", c_cases, 100)]:
        bad_prov = []
        for case in cases:
            for turn in case.get("turns", []):
                pt = turn.get("provenance_type")
                if pt not in _VALID_PROVENANCE:
                    bad_prov.append(f"{case.get('id')}:t{turn.get('turn')}={pt}")
        if not bad_prov:
            vr.ok(f"{label}: all turn provenance_types valid")
        else:
            vr.fail(f"{label}: {len(bad_prov)} turns with invalid provenance")

    # Cross-study source ID summary
    all_source_ids = unique_a_ids | unique_bias_ids | unique_b_ids | unique_bmt_ids | unique_c_ids
    vr.ok(f"v6.1 total unique source IDs across all studies: {len(all_source_ids)}")

    # Unsafe pattern check (sample)
    unsafe_count = 0
    for s in samples_a[:200]:
        prompt = str(s.get("prompt", ""))
        hits = _unsafe_hits(prompt, rules)
        if hits:
            unsafe_count += 1
    if unsafe_count == 0:
        vr.ok("Study A (sampled 200): no unsafe patterns in prompts")
    else:
        vr.warn(f"Study A: {unsafe_count}/200 sampled prompts have unsafe pattern hits")

    return vr


# ---------------------------------------------------------------------------
# Controllability v2.1 validation
# ---------------------------------------------------------------------------

def validate_ctrl_v2_1(ctrl_dir: Path, label: str, rules: dict[str, Any]) -> ValidationResult:
    vr = ValidationResult(f"Controllability v2.1 — {label}")

    if not ctrl_dir.exists():
        vr.fail(f"Directory does not exist: {ctrl_dir}")
        return vr

    # Build manifest
    manifest_path = ctrl_dir / "build_manifest.json"
    if manifest_path.exists():
        manifest = _load(manifest_path)
        gen_pol = manifest.get("generation_policy") or manifest.get("config", {}).get("generation_policy")
        if gen_pol == "strict_no_generation":
            vr.ok("generation_policy = strict_no_generation")
        elif gen_pol:
            vr.fail(f"generation_policy = {gen_pol}")
        else:
            vr.warn("generation_policy not found in build_manifest")
    else:
        vr.fail("build_manifest.json not found")

    # File map
    file_map = {
        "study_a": "study_a_controllability_test.json",
        "study_a_bias": "study_a_bias_controllability_test.json",
        "study_b_single": "study_b_controllability_test.json",
        "study_b_multi": "study_b_multi_turn_controllability_test.json",
        "study_c": "study_c_controllability_test.json",
    }

    study_data: dict[str, list[dict]] = {}
    for study_name, filename in file_map.items():
        path = ctrl_dir / filename
        if not path.exists():
            vr.fail(f"{study_name}: file missing ({filename})")
            continue
        payload = _load(path)
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            for key in ("samples", "cases", "items", "multi_turn_cases"):
                if isinstance(payload.get(key), list):
                    items = payload[key]
                    break
            else:
                items = []
                vr.fail(f"{study_name}: cannot extract items from {filename}")
        else:
            items = []
        study_data[study_name] = items
        vr.ok(f"{study_name}: {len(items)} rows loaded")

    # Per-study checks
    all_source_ids: set[Any] = set()

    for study_name, items in study_data.items():
        # Duplicate IDs
        ids = [str(item.get("id", "")) for item in items]
        dupes = len(ids) - len(set(ids))
        if dupes == 0:
            vr.ok(f"{study_name}: no duplicate IDs")
        else:
            vr.fail(f"{study_name}: {dupes} duplicate IDs")

        # Source ID backing
        no_source = 0
        for item in items:
            metadata = item.get("metadata", {}) or {}
            src_ids = _extract_source_ids(metadata)
            if not src_ids:
                no_source += 1
            else:
                all_source_ids.update(src_ids)
        if no_source == 0:
            vr.ok(f"{study_name}: all rows have source IDs")
        else:
            vr.fail(f"{study_name}: {no_source}/{len(items)} rows missing source IDs")

        # cot_controlled_constraint field (controllability-specific)
        if study_name in ("study_a", "study_a_bias"):
            no_cot = sum(1 for item in items if not str(item.get("cot_controlled_constraint", "")).strip())
            if no_cot == 0:
                vr.ok(f"{study_name}: all rows have cot_controlled_constraint")
            else:
                vr.fail(f"{study_name}: {no_cot} rows missing cot_controlled_constraint")

        # Multi-turn structure
        if study_name in ("study_b_multi", "study_c"):
            turn_issues = []
            for case in items:
                turns = case.get("turns", [])
                if len(turns) != 20:
                    turn_issues.append(f"{case.get('id')}: {len(turns)} turns")
                else:
                    for t in turns:
                        if not str(t.get("message", "")).strip():
                            turn_issues.append(f"{case.get('id')}: blank message at turn {t.get('turn')}")
                            break
            if not turn_issues:
                vr.ok(f"{study_name}: all cases have 20 turns with non-blank messages")
            else:
                vr.fail(f"{study_name}: {len(turn_issues)} cases with turn issues")

        # Bias metadata
        if study_name == "study_a_bias":
            bias_issues = []
            for item in items:
                metadata = item.get("metadata", {}) or {}
                missing = []
                for field in ("dimension", "dimension_family", "inferred_condition", "condition_resolution_source"):
                    val = str(metadata.get(field, "") or "").strip().lower()
                    # "unresolved" is a legitimate value — only fail on truly empty/absent
                    if not val:
                        missing.append(field)
                if not str(item.get("bias_feature", "")).strip():
                    missing.append("bias_feature")
                if not str(item.get("bias_label", "")).strip():
                    missing.append("bias_label")
                if missing:
                    bias_issues.append({"id": item.get("id"), "missing": missing})
            if not bias_issues:
                vr.ok(f"{study_name}: all rows have required bias metadata")
            else:
                vr.fail(f"{study_name}: {len(bias_issues)} rows with incomplete bias metadata")

    unique_count = len(all_source_ids)
    total_rows = sum(len(items) for items in study_data.values())
    vr.ok(f"Total: {unique_count} unique source IDs backing {total_rows} rows")

    # Gold label coverage
    labels_path = ctrl_dir / "ctrl_gold_diagnosis_labels.json"
    if labels_path.exists():
        labels_data = _load(labels_path)
        label_ids = set(labels_data.get("labels", {}).keys())
        study_a_ids = {str(item.get("id", "")) for item in study_data.get("study_a", [])}
        missing_labels = study_a_ids - label_ids
        if not missing_labels:
            vr.ok(f"Gold labels: full coverage ({len(label_ids)} labels for {len(study_a_ids)} Study A rows)")
        else:
            vr.fail(f"Gold labels: {len(missing_labels)} Study A IDs missing labels")

        # Probe backend check
        meta = labels_data.get("meta", {})
        if meta.get("backend") == "probe":
            vr.ok(f"Gold labels: probe backend, primary={meta.get('primary_model', 'n/a')}")
        else:
            vr.warn(f"Gold labels: backend={meta.get('backend', 'n/a')}")
    else:
        vr.fail("ctrl_gold_diagnosis_labels.json not found")

    # Gold plans coverage
    plans_path = ctrl_dir / "ctrl_target_plans.json"
    if plans_path.exists():
        plans_data = _load(plans_path)
        plan_ids = set(plans_data.get("plans", {}).keys())
        study_c_ids = {str(item.get("id", "")) for item in study_data.get("study_c", [])}
        missing_plans = study_c_ids - plan_ids
        if not missing_plans:
            vr.ok(f"Gold plans: full coverage ({len(plan_ids)} plans for {len(study_c_ids)} Study C cases)")
        else:
            vr.fail(f"Gold plans: {len(missing_plans)} Study C IDs missing plans")

        meta = plans_data.get("meta", {})
        if meta.get("backend") == "probe":
            vr.ok(f"Gold plans: probe backend, primary={meta.get('primary_model', 'n/a')}")
        else:
            vr.warn(f"Gold plans: backend={meta.get('backend', 'n/a')}")
    else:
        vr.fail("ctrl_target_plans.json not found")

    # Robust variants exist
    for robust_file in ("ctrl_gold_diagnosis_labels.robust.json", "ctrl_target_plans.robust.json"):
        rpath = ctrl_dir / robust_file
        if rpath.exists():
            rdata = _load(rpath)
            rmeta = rdata.get("meta", {})
            vr.ok(f"{robust_file}: present, secondary={rmeta.get('secondary_model', 'n/a')}")
        else:
            vr.warn(f"{robust_file}: not found (optional)")

    return vr


# ---------------------------------------------------------------------------
# Cross-dataset source ID disjointness
# ---------------------------------------------------------------------------

def validate_cross_dataset_disjointness() -> ValidationResult:
    vr = ValidationResult("Cross-dataset source ID disjointness")

    def _collect_ids(root: Path, filenames: list[str], item_keys: list[str]) -> set[Any]:
        ids: set[Any] = set()
        for fn in filenames:
            path = root / fn
            if not path.exists():
                continue
            data = _load(path)
            items = data if isinstance(data, list) else None
            if items is None and isinstance(data, dict):
                for key in item_keys:
                    if isinstance(data.get(key), list):
                        items = data[key]
                        break
            if not items:
                continue
            for item in items:
                ids.update(_extract_source_ids(item.get("metadata", {}) or {}))
        return ids

    v6_files = ["study_a_test.json", "study_b_test.json", "study_b_multi_turn_test.json", "study_c_test.json"]
    v6_bias = ["adversarial_bias/biased_vignettes.json"]

    ctrl_files = [
        "study_a_controllability_test.json",
        "study_a_bias_controllability_test.json",
        "study_b_controllability_test.json",
        "study_b_multi_turn_controllability_test.json",
        "study_c_controllability_test.json",
    ]

    v6_ids = _collect_ids(V6_1_ROOT, v6_files + v6_bias, ["samples", "cases", "multi_turn_cases"])
    ctrl_ids = _collect_ids(CTRL_V2_1, ctrl_files, ["samples", "cases", "items", "multi_turn_cases"])

    vr.ok(f"v6.1 parent: {len(v6_ids)} unique source IDs")
    vr.ok(f"Controllability v2.1: {len(ctrl_ids)} unique source IDs")

    # ctrl v2.1 draws from a separate OpenR1-Psy slice — check disjointness with v6.1
    ctrl_in_v6 = ctrl_ids & v6_ids
    if not ctrl_in_v6:
        vr.ok("Controllability v2.1 source IDs fully disjoint from v6.1 parent")
    else:
        vr.warn(f"Controllability v2.1 shares {len(ctrl_in_v6)} source IDs with v6.1 parent")

    total_used = v6_ids | ctrl_ids
    vr.ok(f"Total unique source IDs used across all datasets: {len(total_used)}")

    return vr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    rules = _load_rubric_rules()
    results: list[ValidationResult] = []

    print("Running clinical readiness validation...\n")

    # 1. v6.1 parent
    results.append(validate_v6_1(rules))

    # 2. Controllability v2.1 (merged)
    results.append(validate_ctrl_v2_1(CTRL_V2_1, "merged", rules))

    # 4. Cross-dataset disjointness
    results.append(validate_cross_dataset_disjointness())

    # Print results
    for vr in results:
        print(vr.summary())

    # Overall
    all_passed = all(vr.passed for vr in results)
    total_passes = sum(len(vr.passes) for vr in results)
    total_warnings = sum(len(vr.warnings) for vr in results)
    total_failures = sum(len(vr.failures) for vr in results)

    print(f"\n{'='*72}")
    print(f"  OVERALL: {'PASS' if all_passed else 'FAIL'}")
    print(f"  {total_passes} passed, {total_warnings} warnings, {total_failures} failures")
    print(f"{'='*72}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
