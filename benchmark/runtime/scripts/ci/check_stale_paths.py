#!/usr/bin/env python3
"""Fail if legacy path tokens appear in tracked benchmark text artefacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

LEGACY_TOKENS = ("Assignment" + " 2", "Uni-" + "setup", "Mac-" + "setup")
TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".py",
    ".ipynb",
    ".html",
    ".sh",
    ".yml",
    ".yaml",
    ".json",
    ".ini",
    ".toml",
    ".csv",
}


def _tracked_files(root: Path) -> list[Path]:
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "benchmark"],
        check=True,
        capture_output=True,
        text=True,
    )
    files: list[Path] = []
    for rel in proc.stdout.splitlines():
        p = root / rel
        if rel == "benchmark/runtime/scripts/ci/check_stale_paths.py":
            continue
        if p.suffix.lower() in TEXT_SUFFIXES and p.is_file():
            files.append(p)
    return files


def _scan_file(path: Path, root: Path) -> list[str]:
    rel = path.relative_to(root)
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    hits: list[str] = []
    for token in LEGACY_TOKENS:
        if token in content:
            hits.append(f"{rel}: contains '{token}'")
    return hits


def main() -> int:
    runtime_root = Path(__file__).resolve().parents[2]
    repo_root = runtime_root.parents[1]

    violations: list[str] = []
    for file_path in _tracked_files(repo_root):
        violations.extend(_scan_file(file_path, repo_root))

    if violations:
        print("Stale-path guard failed. Remove legacy path tokens:")
        for hit in sorted(set(violations)):
            print(f"- {hit}")
        return 1

    print("Stale-path guard passed: no legacy path tokens found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
