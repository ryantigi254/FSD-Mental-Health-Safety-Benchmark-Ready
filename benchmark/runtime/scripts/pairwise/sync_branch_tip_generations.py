#!/usr/bin/env python3
"""Sync generation caches from selected remote branch tips.

This script reads generation files directly from remote branch tip trees via
`git ls-tree` and `git cat-file --filters`, writes them into the current runtime checkout,
normalises known cross-branch model aliases, and emits a compact coverage
summary for raw / controllability / invariance / ctrl-invariance lanes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


BRANCH_ORDER: List[str] = [
    "origin/codex/worker-system-all-studies-port",
    "origin/metric-invariance",
    "origin/controllability-v2.1",
    "origin/v6.1-parent-data",
]

ROOT_ALIASES: Dict[str, str] = {
    "results": "results",
    "results_invariance": "results_invariance",
    "results_ctrl_invariance": "results_ctrl_invariance",
    "results_invariance_ctrl": "results_ctrl_invariance",
    "results_ctrl_v2_1": "results",
}

MODEL_ALIASES: Dict[str, str] = {
    "piaget-lmstudio": "piaget-8b-local",
    "piaget-local": "piaget-8b-local",
    "psyche-r1-lmstudio": "psyche-r1-local",
    "psyche_r1_lmstudio": "psyche-r1-local",
    "psych-qwen-32b": "psych-qwen-32b-local",
    "psyllm-gml-local": "psyllm-lmstudio",
    "medgemma_lmstudio": "medgemma-lmstudio",
    "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8-0": "qwen3.5-27b-distilled",
}

RAW_TABLE_FILES: Dict[str, str] = {
    "study_a_generations.jsonl": "study_a",
    "study_a_bias_generations.jsonl": "study_a_bias",
    "study_b_generations.jsonl": "study_b",
    "study_b_multi_turn_generations.jsonl": "study_b_mt",
    "study_c_generations.jsonl": "study_c",
}

CTRL_TABLE_FILES: Dict[str, str] = {
    "ctrl_study_a_generations.jsonl": "ctrl_a",
    "ctrl_study_a_bias_generations.jsonl": "ctrl_a_bias",
    "ctrl_study_b_generations.jsonl": "ctrl_b",
    "ctrl_study_b_multi_turn_generations.jsonl": "ctrl_b_mt",
    "ctrl_study_c_generations.jsonl": "ctrl_c",
}

INVARIANCE_TABLE_FILES: Dict[str, str] = {
    "study_a_invariance_generations.jsonl": "study_a",
    "study_a_bias_invariance_generations.jsonl": "study_a_bias",
    "study_b_invariance_generations.jsonl": "study_b",
    "study_b_multi_turn_invariance_generations.jsonl": "study_b_mt",
    "study_c_invariance_generations.jsonl": "study_c",
}

TRACKED_MODELS: List[str] = [
    "deepseek-r1-lmstudio",
    "gpt-oss-20b",
    "piaget-8b-local",
    "psych-qwen-32b-local",
    "psyche-r1-local",
    "qwen3-lmstudio",
    "qwq",
    "medgemma-lmstudio",
    "qwen3.5-27b-distilled",
    "psyllm-lmstudio",
]


@dataclass(frozen=True)
class BranchTip:
    ref: str
    commit: str
    commit_ts: int
    commit_iso: str
    order: int


@dataclass
class Candidate:
    branch: BranchTip
    source_path: str
    dest_rel: Path


def _git(runtime_root: Path, *args: str, decode: bool = True) -> str | bytes:
    cmd = ["git", *args]
    result = subprocess.run(
        cmd,
        cwd=runtime_root.parent.parent,
        check=True,
        capture_output=True,
    )
    if decode:
        return result.stdout.decode("utf-8")
    return result.stdout


def _load_branch_tips(runtime_root: Path, refs: Iterable[str]) -> List[BranchTip]:
    tips: List[BranchTip] = []
    for index, ref in enumerate(refs):
        commit = _git(runtime_root, "rev-parse", ref).strip()
        commit_ts = int(_git(runtime_root, "log", "-1", "--format=%ct", ref).strip())
        commit_iso = _git(runtime_root, "log", "-1", "--format=%cI", ref).strip()
        tips.append(
            BranchTip(
                ref=ref,
                commit=commit,
                commit_ts=commit_ts,
                commit_iso=commit_iso,
                order=index,
            )
        )
    return tips


def _normalise_model_dir(model_dir: str) -> str:
    return MODEL_ALIASES.get(model_dir, model_dir)


def _list_generation_paths(runtime_root: Path, tip: BranchTip) -> List[str]:
    paths: List[str] = []
    for root in ROOT_ALIASES:
        stdout = _git(
            runtime_root,
            "ls-tree",
            "-r",
            "--name-only",
            tip.ref,
            "--",
            f"benchmark/runtime/{root}",
        )
        for raw_path in stdout.splitlines():
            raw_path = raw_path.strip()
            if raw_path.endswith(".jsonl"):
                paths.append(raw_path)
    return paths


def _dest_from_source(source_path: str) -> Optional[Path]:
    parts = Path(source_path).parts
    if len(parts) < 5:
        return None
    if parts[0:2] != ("benchmark", "runtime"):
        return None
    root_name = parts[2]
    model_dir = parts[3]
    filename = parts[-1]
    root_alias = ROOT_ALIASES.get(root_name)
    if root_alias is None:
        return None
    return Path("benchmark") / "runtime" / root_alias / _normalise_model_dir(model_dir) / filename


def _choose_candidate(current: Optional[Candidate], new: Candidate) -> Candidate:
    if current is None:
        return new
    current_key = (current.branch.commit_ts, -current.branch.order)
    new_key = (new.branch.commit_ts, -new.branch.order)
    return new if new_key > current_key else current


def _sync_files(runtime_root: Path, tips: List[BranchTip]) -> Dict[str, Candidate]:
    chosen: Dict[str, Candidate] = {}
    for tip in tips:
        for source_path in _list_generation_paths(runtime_root, tip):
            dest_rel = _dest_from_source(source_path)
            if dest_rel is None:
                continue
            candidate = Candidate(branch=tip, source_path=source_path, dest_rel=dest_rel)
            chosen[str(dest_rel)] = _choose_candidate(chosen.get(str(dest_rel)), candidate)

    unavailable: List[Candidate] = []
    for candidate in list(chosen.values()):
        try:
            blob = _git(
                runtime_root,
                "cat-file",
                "--filters",
                f"{candidate.branch.ref}:{candidate.source_path}",
                decode=False,
            )
        except subprocess.CalledProcessError:
            unavailable.append(candidate)
            chosen.pop(str(candidate.dest_rel), None)
            continue
        dest_path = runtime_root.parent.parent / candidate.dest_rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(blob)

    if unavailable:
        print(
            json.dumps(
                {
                    "warning": "skipped_unavailable_generation_blobs",
                    "count": len(unavailable),
                    "paths": [
                        {
                            "branch": item.branch.ref,
                            "source_path": item.source_path,
                            "destination": str(item.dest_rel),
                        }
                        for item in unavailable[:20]
                    ],
                },
                indent=2,
            )
        )

    return chosen


def _needs_backfill(source_path: Path, dest_path: Path) -> bool:
    if not dest_path.exists():
        return True
    source_size = source_path.stat().st_size
    dest_size = dest_path.stat().st_size
    if dest_size == 0:
        return True
    if dest_size <= 200 < source_size:
        return True
    if dest_size < source_size:
        return True
    return False


def _backfill_from_local_aliases(runtime_root: Path) -> List[Dict[str, str]]:
    workspace_root = runtime_root.parent.parent
    backfills: List[Dict[str, str]] = []
    local_roots = sorted(set(ROOT_ALIASES.values()))

    for root_name in local_roots:
        root_path = runtime_root / root_name
        if not root_path.exists():
            continue
        for alias, canonical in MODEL_ALIASES.items():
            alias_dir = root_path / alias
            if not alias_dir.exists():
                continue
            canonical_dir = root_path / canonical
            for alias_file in sorted(p for p in alias_dir.rglob("*") if p.is_file()):
                relative_path = alias_file.relative_to(alias_dir)
                canonical_file = canonical_dir / relative_path
                if not _needs_backfill(alias_file, canonical_file):
                    continue
                canonical_file.parent.mkdir(parents=True, exist_ok=True)
                canonical_file.write_bytes(alias_file.read_bytes())
                backfills.append(
                    {
                        "destination": str(canonical_file.relative_to(workspace_root)),
                        "source_path": str(alias_file.relative_to(workspace_root)),
                    }
                )
    return backfills


def _build_table(models: List[str], columns: List[str], values: Dict[str, Dict[str, bool]]) -> str:
    lines = ["| Model | " + " | ".join(columns) + " |", "|" + "---|" * (len(columns) + 1)]
    for model in models:
        cells = ["✅" if values.get(model, {}).get(column, False) else "❌" for column in columns]
        lines.append("| " + " | ".join([model, *cells]) + " |")
    return "\n".join(lines)


def _coverage_summary(runtime_root: Path) -> Dict[str, Dict[str, Dict[str, bool]]]:
    raw: Dict[str, Dict[str, bool]] = {}
    ctrl: Dict[str, Dict[str, bool]] = {}
    invariance: Dict[str, Dict[str, bool]] = {}
    ctrl_invariance: Dict[str, Dict[str, bool]] = {}

    root_names = ["results", "results_invariance", "results_ctrl_invariance"]
    for root_name in root_names:
        root_path = runtime_root / root_name
        if not root_path.exists():
            continue
        for file_path in root_path.rglob("*.jsonl"):
            try:
                model = file_path.relative_to(root_path).parts[0]
            except IndexError:
                continue
            filename = file_path.name

            if root_name == "results":
                if filename in RAW_TABLE_FILES:
                    raw.setdefault(model, {})[RAW_TABLE_FILES[filename]] = True
                if filename in CTRL_TABLE_FILES:
                    ctrl.setdefault(model, {})[CTRL_TABLE_FILES[filename]] = True
            elif root_name == "results_invariance" and filename in INVARIANCE_TABLE_FILES:
                invariance.setdefault(model, {})[INVARIANCE_TABLE_FILES[filename]] = True
            elif root_name == "results_ctrl_invariance":
                if filename in INVARIANCE_TABLE_FILES:
                    ctrl_invariance.setdefault(model, {})[INVARIANCE_TABLE_FILES[filename]] = True
                if filename in CTRL_TABLE_FILES:
                    ctrl.setdefault(model, {})[CTRL_TABLE_FILES[filename]] = True

    return {
        "raw": raw,
        "controllability": ctrl,
        "invariance": invariance,
        "ctrl_invariance": ctrl_invariance,
    }


def _write_summary(
    runtime_root: Path,
    tips: List[BranchTip],
    chosen: Dict[str, Candidate],
    local_backfills: List[Dict[str, str]],
) -> None:
    coverage = _coverage_summary(runtime_root)

    summary_json = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "branch_tips": [
            {
                "ref": tip.ref,
                "commit": tip.commit,
                "commit_iso": tip.commit_iso,
            }
            for tip in tips
        ],
        "copied_files": [
            {
                "destination": str(candidate.dest_rel),
                "source_ref": candidate.branch.ref,
                "source_commit": candidate.branch.commit,
                "source_path": candidate.source_path,
            }
            for candidate in sorted(chosen.values(), key=lambda item: str(item.dest_rel))
        ],
        "local_alias_backfills": local_backfills,
        "coverage": coverage,
    }

    summary_md_lines = [
        f"Fetched and rebuilt from these remote branch tips on **{datetime.now(timezone.utc).date().isoformat()}**:",
    ]
    for tip in tips:
        commit_day = tip.commit_iso.split("T", 1)[0]
        summary_md_lines.append(f"- `{tip.ref}` at `{tip.commit[:8]}` from **{commit_day}**")
    if local_backfills:
        summary_md_lines.append(
            f"- Local alias backfills: **{len(local_backfills)}** file(s) copied into canonical model names."
        )
    summary_md_lines.extend(
        [
            "",
            "### Table 1 - Main raw",
            "",
            _build_table(
                TRACKED_MODELS,
                ["study_a", "study_a_bias", "study_b", "study_b_mt", "study_c"],
                coverage["raw"],
            ),
            "",
            "### Table 2 - Controlability",
            "",
            _build_table(
                TRACKED_MODELS,
                ["ctrl_a", "ctrl_a_bias", "ctrl_b", "ctrl_b_mt", "ctrl_c"],
                coverage["controllability"],
            ),
            "",
            "### Table 3 - Invariance",
            "",
            _build_table(
                TRACKED_MODELS,
                ["study_a", "study_a_bias", "study_b", "study_b_mt", "study_c"],
                coverage["invariance"],
            ),
            "",
            "### Table 4 - Ctrl-invariance",
            "",
            _build_table(
                TRACKED_MODELS,
                ["study_a", "study_a_bias", "study_b", "study_b_mt", "study_c"],
                coverage["ctrl_invariance"],
            ),
            "",
            "### Table 5 - Reverse",
            "",
            _build_table(
                TRACKED_MODELS,
                ["study_a", "study_a_bias", "study_b", "study_b_mt", "study_c"],
                {},
            ),
            "",
            "Two small notes:",
            "- `ctrl-invariance` normalises both `results_ctrl_invariance/` and `results_invariance_ctrl/` into one local root.",
            "- `reverse` stays empty because none of the four branch tips expose a dedicated reverse-results lane.",
        ]
    )

    output_dir = runtime_root / "metric-results" / "pairwise"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "branch_tip_sync_summary.json").write_text(
        json.dumps(summary_json, indent=2),
        encoding="utf-8",
    )
    (output_dir / "branch_tip_sync_summary.md").write_text(
        "\n".join(summary_md_lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync generation caches from remote branch tips.")
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Path to benchmark/runtime",
    )
    args = parser.parse_args()

    runtime_root = args.runtime_root.resolve()
    tips = _load_branch_tips(runtime_root, BRANCH_ORDER)
    chosen = _sync_files(runtime_root, tips)
    local_backfills = _backfill_from_local_aliases(runtime_root)
    _write_summary(runtime_root, tips, chosen, local_backfills)

    print(f"Synced {len(chosen)} generation files from {len(tips)} branch tips.")
    print(f"Local alias backfills applied: {len(local_backfills)}")
    for tip in tips:
        print(f"- {tip.ref} @ {tip.commit[:8]} ({tip.commit_iso})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
