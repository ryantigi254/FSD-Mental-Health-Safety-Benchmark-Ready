#!/usr/bin/env python3
"""Deterministically restructure Study A adversarial bias data for v3.1 quality."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = RUNTIME_ROOT / "data" / "adversarial_bias"
SOURCE_PATH = DATA_DIR / "biased_vignettes.json"
LEGACY_PATH = DATA_DIR / "biased_vignettes_legacy_2016.json"
REPORT_PATH = (
    RUNTIME_ROOT
    / "docs"
    / "reports"
    / "clinician_package"
    / "v0.3"
    / "adversarial_bias_structure_report.json"
)

STRUCTURE_VERSION = "v3.1"
WHITESPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^a-z0-9]+")

# Minimal normalisation aliases for common demographic lexical variants.
FEATURE_ALIASES = {
    "black": {"black", "african american", "african-american", "african"},
    "white": {"white", "caucasian", "european"},
    "asian": {"asian", "south asian", "east asian"},
    "male": {"male", "man"},
    "female": {"female", "woman"},
    "transgender": {"transgender", "trans"},
    "non-binary": {"non-binary", "nonbinary", "non binary"},
    "middle eastern": {"middle eastern", "middle-eastern"},
    "hispanic": {"hispanic", "latino", "latina"},
}


def _norm(text: object) -> str:
    return WHITESPACE_RE.sub(" ", str(text or "").strip().lower())


def _tokenise_feature(feature: str) -> set[str]:
    base = _norm(feature)
    aliases = set(FEATURE_ALIASES.get(base, set()))
    aliases.add(base)
    aliases.add(base.replace("-", " "))
    aliases.add(base.replace(" ", "-"))
    return {a for a in aliases if a}


def _normalise_prompt_signature(prompt: str, bias_feature: str) -> str:
    text = _norm(prompt)
    variants = sorted(_tokenise_feature(bias_feature), key=len, reverse=True)
    for v in variants:
        if v:
            text = text.replace(v, "<bias_feature>")
    # final lightweight symbol normalisation
    text = NON_WORD_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def _canonical_key(case: dict) -> str:
    return "||".join(
        [
            _norm(case.get("prompt", "")),
            _norm(case.get("bias_feature", "")),
            _norm(case.get("bias_label", "")),
        ]
    )


def _ensure_legacy_copy() -> None:
    if LEGACY_PATH.exists():
        return
    LEGACY_PATH.write_text(SOURCE_PATH.read_text(encoding="utf-8"), encoding="utf-8")


def _load_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError(f"Expected list at {path}: cases")
    return cases


def _build_canonical_cases(source_cases: list[dict]) -> tuple[list[dict], dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for case in source_cases:
        grouped[_canonical_key(case)].append(case)

    canonical_rows: list[dict] = []
    dimensions = Counter()
    signature_counter = Counter()

    for index, key in enumerate(sorted(grouped), start=1):
        items = sorted(grouped[key], key=lambda x: str(x.get("id", "")))
        lead = items[0]
        prompt = str(lead.get("prompt", "")).strip()
        bias_feature = str(lead.get("bias_feature", "")).strip()
        bias_label = str(lead.get("bias_label", "")).strip()

        dim_counts = Counter(
            _norm((i.get("metadata", {}) or {}).get("dimension", "")) for i in items
        )
        dimension = dim_counts.most_common(1)[0][0] if dim_counts else ""
        dimensions[dimension] += 1

        template_signature = _normalise_prompt_signature(prompt, bias_feature)
        pair_group_id = hashlib.sha1(template_signature.encode("utf-8")).hexdigest()[:12]
        signature_counter[template_signature] += 1

        canonical_rows.append(
            {
                "id": f"abias_{index:04d}",
                "prompt": prompt,
                "bias_feature": bias_feature,
                "bias_label": bias_label,
                "metadata": {"dimension": dimension},
                "pair_group_id": pair_group_id,
                "template_signature": template_signature,
                "structure_version": STRUCTURE_VERSION,
                "source_variant_count": len(items),
            }
        )

    report = {
        "structure_version": STRUCTURE_VERSION,
        "old_count": len(source_cases),
        "new_count": len(canonical_rows),
        "effective_duplicates_removed": len(source_cases) - len(canonical_rows),
        "dimensions": dict(dimensions),
        "pair_group_count": len({row["pair_group_id"] for row in canonical_rows}),
        "top_signatures": [
            {"template_signature": sig, "count": n}
            for sig, n in signature_counter.most_common(20)
        ],
    }
    return canonical_rows, report


def main() -> int:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing source dataset: {SOURCE_PATH}")

    _ensure_legacy_copy()
    source_cases = _load_cases(LEGACY_PATH if LEGACY_PATH.exists() else SOURCE_PATH)
    canonical_rows, report = _build_canonical_cases(source_cases)

    SOURCE_PATH.write_text(
        json.dumps({"cases": canonical_rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"Restructured adversarial bias dataset: {report['old_count']} -> {report['new_count']} "
        f"(removed {report['effective_duplicates_removed']} duplicates)"
    )
    print(f"Wrote canonical dataset: {SOURCE_PATH}")
    print(f"Wrote legacy archive: {LEGACY_PATH}")
    print(f"Wrote structure report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
