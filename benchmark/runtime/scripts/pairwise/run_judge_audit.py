#!/usr/bin/env python3
"""Compute judge-audit metrics from parsed pairwise records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RUNTIME_ROOT / "src"))

from reliable_clinical_benchmark.pairwise.judge_audit import (  # noqa: E402
    compute_judge_audit_report,
    load_judge_audit_manifest,
    write_judge_audit_report,
)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    manifest = load_judge_audit_manifest(args.manifest)
    records = _load_records(args.parsed_jsonl, args.parsed_dir)
    report = compute_judge_audit_report(records=records, manifest=manifest)
    output_path = write_judge_audit_report(
        output_root=args.output_root,
        report_name=args.report_name,
        report=report,
    )
    print(
        json.dumps(
            {
                "audit_items": len(manifest.items),
                "records": len(records),
                "report_path": str(output_path),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the pairwise judge-audit summary.")
    parser.add_argument("--manifest", required=True, help="Path to the judge-audit manifest JSON.")
    parser.add_argument(
        "--parsed-jsonl",
        action="append",
        default=[],
        help="One or more parsed pairwise JSONL files. May be supplied multiple times.",
    )
    parser.add_argument(
        "--parsed-dir",
        action="append",
        default=[],
        help="One or more directories to scan recursively for parsed pairwise JSONL files.",
    )
    parser.add_argument(
        "--output-root",
        default=str(RUNTIME_ROOT / "metric-results" / "pairwise" / "judge_audit" / "reports"),
        help="Directory for the judge-audit report JSON.",
    )
    parser.add_argument(
        "--report-name",
        default="judge_audit_summary",
        help="Stem name for the written report file.",
    )
    return parser


def _load_records(parsed_jsonl: list[str], parsed_dirs: list[str]) -> list[dict]:
    records: list[dict] = []
    seen: set[Path] = set()
    for entry in parsed_jsonl:
        path = Path(entry).resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        records.extend(_read_jsonl(path))
    for directory in parsed_dirs:
        root = Path(directory).resolve()
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.jsonl")):
            if path in seen:
                continue
            seen.add(path)
            records.extend(_read_jsonl(path))
    return records


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
