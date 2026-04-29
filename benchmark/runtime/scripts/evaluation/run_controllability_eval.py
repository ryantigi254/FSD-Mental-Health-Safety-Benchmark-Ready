"""Run the branch controllability evaluation pipeline for a set of models.

Outputs per-model `ctrl_study_X_results.json` files compatible with the
external controllability analysis notebooks.

Usage
-----
    python run_controllability_eval.py \
        --branch-src /path/to/worktree/benchmark/runtime/src \
        --ctrl-dir data/controllability/misc/controllability_splits_large/controllability_splits \
        --output-dir metric-results/controllability_v2 \
        --models gpt-oss-20b piaget-8b-local qwen3-lmstudio psyllm-lmstudio psyche-r1-local
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parents[2]

STUDY_KEY_MAP = {
    "study_a": "A",
    "study_a_bias": "A_bias",
    "study_b": "B",
    "study_b_multi_turn": "B_multi_turn",
    "study_c": "C",
}


def _load_branch(branch_src: Path) -> object:
    src_str = str(branch_src.resolve())
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    from reliable_clinical_benchmark.pipelines import controllability  # type: ignore[import]
    return controllability


def main() -> int:
    parser = argparse.ArgumentParser(description="Run branch controllability evaluation")
    parser.add_argument(
        "--branch-src",
        type=Path,
        default=None,
        help="Path to branch runtime/src (e.g. /path/to/worktree/benchmark/runtime/src). "
             "Defaults to the current working tree src.",
    )
    parser.add_argument(
        "--ctrl-dir",
        type=Path,
        default=RUNTIME_ROOT / "data" / "controllability" / "misc"
        / "controllability_splits_large" / "controllability_splits",
        help="Directory containing study_X_controllability_test.json gold splits.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RUNTIME_ROOT / "results",
        help="Directory containing per-model results subdirectories with ctrl_study_X_generations.jsonl.",
    )
    parser.add_argument(
        "--ctrl-results-dir",
        type=Path,
        default=None,
        help="Alternative directory for ctrl generation files. Falls back to --results-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUNTIME_ROOT / "metric-results" / "controllability_v2",
        help="Root output directory. Per-model results go under <output-dir>/<model>/.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names to evaluate.",
    )
    parser.add_argument(
        "--studies",
        nargs="+",
        default=list(STUDY_KEY_MAP),
        help="Studies to evaluate (default: all).",
    )
    args = parser.parse_args()

    branch_src = (
        args.branch_src.resolve()
        if args.branch_src
        else RUNTIME_ROOT / "src"
    )
    print(f"Branch src: {branch_src}", flush=True)

    try:
        ctrl_module = _load_branch(branch_src)
    except ImportError as exc:
        print(f"ERROR: Could not import controllability module from {branch_src}: {exc}", file=sys.stderr)
        return 1

    ctrl_dir = args.ctrl_dir.resolve()
    results_dir = args.results_dir.resolve()
    ctrl_results_dir = (args.ctrl_results_dir or args.results_dir).resolve()
    output_dir = args.output_dir.resolve()

    if not ctrl_dir.exists():
        print(f"ERROR: ctrl-dir does not exist: {ctrl_dir}", file=sys.stderr)
        return 1

    eval_map = {
        "A": ctrl_module.evaluate_ctrl_study_a,
        "A_bias": ctrl_module.evaluate_ctrl_study_a_bias,
        "B": ctrl_module.evaluate_ctrl_study_b,
        "B_multi_turn": ctrl_module.evaluate_ctrl_study_b_multi_turn,
        "C": ctrl_module.evaluate_ctrl_study_c,
    }

    result_name_map = ctrl_module.CTRL_RESULT_NAMES

    summary: list[dict] = []

    for model in args.models:
        model_results_dir = results_dir / model
        ctrl_model_dir = ctrl_results_dir / model

        # Check which dir has ctrl files
        if not model_results_dir.exists() and not ctrl_model_dir.exists():
            print(f"[SKIP] {model}: no results directory found", flush=True)
            summary.append({"model": model, "status": "no_results_dir"})
            continue

        # Use ctrl_results_dir if it has ctrl files, fall back to results_dir
        active_ctrl_dir = ctrl_model_dir if ctrl_model_dir.exists() else model_results_dir

        model_out_dir = output_dir / model
        model_out_dir.mkdir(parents=True, exist_ok=True)
        model_summary: dict = {"model": model, "studies": {}}

        for study in args.studies:
            study_key = STUDY_KEY_MAP.get(study)
            if study_key is None:
                print(f"  [{model}] Unknown study '{study}', skipping", flush=True)
                continue

            eval_fn = eval_map.get(study_key)
            if eval_fn is None:
                print(f"  [{model}] No eval function for study_key '{study_key}', skipping", flush=True)
                continue

            cache_name = ctrl_module.CTRL_CACHE_NAMES.get(study_key, f"ctrl_study_{study}_generations.jsonl")
            cache_path = active_ctrl_dir / cache_name
            if not cache_path.exists():
                # Try the other dir
                alt_cache = model_results_dir / cache_name
                if alt_cache.exists():
                    cache_path = alt_cache
                else:
                    print(f"  [{model}/{study}] Missing cache: {cache_path}", flush=True)
                    model_summary["studies"][study] = "missing_cache"
                    continue

            print(f"  [{model}/{study}] Evaluating ...", flush=True)
            try:
                if study_key in ("A", "B", "B_multi_turn", "C"):
                    result = eval_fn(
                        model_name=model,
                        model_results_dir=active_ctrl_dir,
                        ctrl_dir=ctrl_dir,
                    )
                else:
                    # study_a_bias doesn't need ctrl_dir
                    result = eval_fn(
                        model_name=model,
                        model_results_dir=active_ctrl_dir,
                    )
                payload = asdict(result)
                out_name = result_name_map.get(study_key, f"ctrl_{study}_results.json")
                out_path = model_out_dir / out_name
                out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                print(f"  [{model}/{study}] OK -> {out_path}", flush=True)
                model_summary["studies"][study] = "ok"
            except Exception as exc:
                print(f"  [{model}/{study}] ERROR: {exc}", flush=True)
                model_summary["studies"][study] = f"error: {exc}"

        summary.append(model_summary)

    # Write summary
    summary_path = output_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSummary written to {summary_path}", flush=True)

    errors = sum(
        1 for m in summary for st in m.get("studies", {}).values() if str(st).startswith("error")
    )
    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
