#!/usr/bin/env python3
"""Benchmark MLX batch sizes for Study A bias generation cutoffs."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _ensure_src_on_path(runtime_root: Path) -> None:
    src_dir = runtime_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _resolve_local_model_path(runtime_root: Path, explicit_model: str | None) -> Path:
    if explicit_model:
        model_path = Path(explicit_model).expanduser()
        if model_path.exists():
            return model_path
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    model_path = runtime_root / "models" / "Psych_Qwen_32B_MLX_4bit"
    if model_path.exists():
        return model_path
    raise FileNotFoundError(
        "Default MLX model path not found. Expected "
        f"{model_path}. Pass --model with an explicit MLX model directory."
    )


def _format_bias_prompt(vignette: str) -> str:
    return (
        f"{vignette}\n\n"
        "Think step-by-step about the patient's presentation and provide the most likely diagnosis. "
        "Explain your reasoning clearly."
    )


def _canonical_model_output_dir(model_id: str) -> str:
    canonical_names = {
        "psych_qwen_local": "psych-qwen-32b-local",
        "psych_qwen_mlx": "psych-qwen-32b-local",
        "psych-qwen-32b-local": "psych-qwen-32b-local",
    }
    return canonical_names.get(model_id.lower(), model_id)


@dataclass
class SweepResult:
    batch_size: int
    status: str
    total_cases: int
    wall_seconds: float
    cases_per_second: float
    avg_prompt_tps: float
    avg_generation_tps: float
    avg_peak_memory_gb: float
    error: str = ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MLX batch sizes for Study A bias prompts."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="psych_qwen_local",
        help="Logical model id used for results folder naming.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to an MLX-converted model directory.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to biased_vignettes.json. Defaults to runtime data path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output dir. Defaults to runtime/results.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=20,
        help="Number of Study A bias cases to benchmark.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max tokens generated per case in benchmark mode.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Batch sizes to test in ascending order.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Trials per batch size. Results are averaged per batch size.",
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.10,
        help="Minimum relative wall-time improvement to keep scaling batch size.",
    )
    return parser.parse_args()


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _recommend_batch(results: list[SweepResult], threshold: float) -> int | None:
    ok = [r for r in results if r.status == "ok"]
    if not ok:
        return None
    ok = sorted(ok, key=lambda r: r.batch_size)
    if len(ok) == 1:
        return ok[0].batch_size
    for idx in range(1, len(ok)):
        prev = ok[idx - 1]
        cur = ok[idx]
        if prev.wall_seconds <= 0:
            continue
        improvement = (prev.wall_seconds - cur.wall_seconds) / prev.wall_seconds
        if improvement < threshold:
            return cur.batch_size
    fastest = min(ok, key=lambda r: r.wall_seconds)
    return fastest.batch_size


def _fastest_batch(results: list[SweepResult]) -> int | None:
    ok = [r for r in results if r.status == "ok"]
    if not ok:
        return None
    return min(ok, key=lambda r: r.wall_seconds).batch_size


def main() -> None:
    args = _parse_args()
    runtime_root = Path(__file__).resolve().parents[3]
    _ensure_src_on_path(runtime_root)

    from reliable_clinical_benchmark.data.adversarial_loader import load_adversarial_bias_cases
    from mlx_lm import batch_generate, load

    model_path = _resolve_local_model_path(runtime_root, args.model)
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = runtime_root / "data" / "adversarial_bias" / "biased_vignettes.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Study A bias data not found: {data_path}")

    raw_cases = load_adversarial_bias_cases(str(data_path))
    if not raw_cases:
        raise ValueError(f"No bias cases loaded from {data_path}")
    selected_cases = raw_cases[: max(1, int(args.max_cases))]

    prompts: list[dict[str, Any]] = []
    for case in selected_cases:
        prompts.append(
            {
                "id": case.get("id", ""),
                "bias_feature": case.get("bias_feature", ""),
                "bias_label": case.get("bias_label", ""),
                "prompt": _format_bias_prompt(case.get("prompt", "")),
            }
        )

    output_dir = Path(args.output_dir) if args.output_dir else runtime_root / "results"
    model_output_dir = output_dir / _canonical_model_output_dir(args.model_id)
    benchmark_root = model_output_dir / "misc" / "mlx_batch_benchmark"
    temp_root = benchmark_root / "temp_generations"
    reports_root = benchmark_root / "reports"
    temp_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    run_date = datetime.now().strftime("%Y%m%d")
    run_stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    json_report_path = reports_root / f"study_a_mlx_batch_cutoff_{run_stamp}.json"
    md_report_path = reports_root / f"study_a_mlx_batch_cutoff_{run_stamp}.md"
    json_report_latest_path = reports_root / f"study_a_mlx_batch_cutoff_{run_date}.json"
    md_report_latest_path = reports_root / f"study_a_mlx_batch_cutoff_{run_date}.md"

    print(f"Loading MLX model: {model_path}")
    model, tokenizer = load(str(model_path))
    print(f"Loaded {len(prompts)} Study A bias prompts for benchmarking.")

    # Warmup to avoid first-run compile skew dominating all results.
    _ = batch_generate(
        model,
        tokenizer,
        [tokenizer.encode(prompts[0]["prompt"])],
        max_tokens=min(8, args.max_tokens),
        verbose=False,
    )

    sweep_results: list[SweepResult] = []

    for batch_size in sorted(set(int(v) for v in args.batch_sizes if int(v) > 0)):
        print(f"\nBenchmarking batch_size={batch_size}")
        batch_temp_dir = temp_root / f"batch_{batch_size}"
        batch_temp_dir.mkdir(parents=True, exist_ok=True)
        batch_temp_file = (
            batch_temp_dir
            / f"study_a_bias_generations_temp_batch{batch_size}_s{len(prompts)}_{run_stamp}.jsonl"
        )

        trial_wall_times: list[float] = []
        trial_prompt_tps: list[float] = []
        trial_gen_tps: list[float] = []
        trial_peak_mem: list[float] = []

        error_message = ""
        status = "ok"
        total_cases_written = 0

        for trial_idx in range(1, max(1, int(args.trials)) + 1):
            batch_lines: list[str] = []
            prompt_tps_values: list[float] = []
            gen_tps_values: list[float] = []
            peak_mem_values: list[float] = []

            t0 = time.perf_counter()
            try:
                for start in range(0, len(prompts), batch_size):
                    block = prompts[start : start + batch_size]
                    encoded = [tokenizer.encode(item["prompt"]) for item in block]
                    response = batch_generate(
                        model,
                        tokenizer,
                        encoded,
                        max_tokens=args.max_tokens,
                        verbose=False,
                    )
                    prompt_tps_values.append(float(response.stats.prompt_tps))
                    gen_tps_values.append(float(response.stats.generation_tps))
                    peak_mem_values.append(float(response.stats.peak_memory))
                    for item, text in zip(block, response.texts):
                        out_entry = {
                            "id": item["id"],
                            "bias_feature": item["bias_feature"],
                            "bias_label": item["bias_label"],
                            "prompt": item["prompt"],
                            "output_text": text,
                            "status": "ok" if str(text).strip() else "error",
                            "error_message": "" if str(text).strip() else "Empty generation output",
                            "timestamp": _now_utc(),
                            "model_name": args.model_id,
                            "meta": {
                                "batch_size": batch_size,
                                "trial": trial_idx,
                            },
                        }
                        batch_lines.append(json.dumps(out_entry, ensure_ascii=False))
            except Exception as exc:  # pragma: no cover - runtime dependent
                status = "error"
                error_message = str(exc)

            wall_seconds = time.perf_counter() - t0
            trial_wall_times.append(wall_seconds)

            if status == "ok":
                trial_prompt_tps.append(_safe_mean(prompt_tps_values))
                trial_gen_tps.append(_safe_mean(gen_tps_values))
                trial_peak_mem.append(max(peak_mem_values) if peak_mem_values else 0.0)
                total_cases_written = len(batch_lines)
                with open(batch_temp_file, "a", encoding="utf-8") as f:
                    for line in batch_lines:
                        f.write(line + "\n")
            else:
                break

        mean_wall = _safe_mean(trial_wall_times)
        cases_per_sec = (len(prompts) / mean_wall) if mean_wall > 0 else 0.0
        result = SweepResult(
            batch_size=batch_size,
            status=status,
            total_cases=total_cases_written if status == "ok" else 0,
            wall_seconds=mean_wall,
            cases_per_second=cases_per_sec if status == "ok" else 0.0,
            avg_prompt_tps=_safe_mean(trial_prompt_tps) if status == "ok" else 0.0,
            avg_generation_tps=_safe_mean(trial_gen_tps) if status == "ok" else 0.0,
            avg_peak_memory_gb=_safe_mean(trial_peak_mem) if status == "ok" else 0.0,
            error=error_message,
        )
        sweep_results.append(result)
        if status == "ok":
            print(
                f"batch={batch_size} wall={result.wall_seconds:.2f}s "
                f"cases/s={result.cases_per_second:.3f} "
                f"gen_tps={result.avg_generation_tps:.1f} "
                f"peak_mem={result.avg_peak_memory_gb:.2f}GB"
            )
        else:
            print(f"batch={batch_size} failed: {error_message}")

    recommended_batch = _recommend_batch(sweep_results, args.improvement_threshold)
    fastest_batch = _fastest_batch(sweep_results)

    report_payload = {
        "timestamp_utc": _now_utc(),
        "runtime_root": str(runtime_root),
        "model_id": args.model_id,
        "model_path": str(model_path),
        "data_path": str(data_path),
        "max_cases": len(prompts),
        "max_tokens": int(args.max_tokens),
        "batch_sizes": sorted(set(int(v) for v in args.batch_sizes if int(v) > 0)),
        "trials": max(1, int(args.trials)),
        "improvement_threshold": float(args.improvement_threshold),
        "recommended_batch_size": recommended_batch,
        "fastest_batch_size": fastest_batch,
        "results": [r.__dict__ for r in sweep_results],
    }
    report_json = json.dumps(report_payload, indent=2)
    json_report_path.write_text(report_json, encoding="utf-8")
    json_report_latest_path.write_text(report_json, encoding="utf-8")

    ok_rows = sorted([row for row in sweep_results if row.status == "ok"], key=lambda r: r.batch_size)
    wall_clock_chain = ", ".join(f"{row.batch_size}\u2192{row.wall_seconds:.2f}" for row in ok_rows)
    speedup_vs_batch1 = []
    if ok_rows:
        base_wall = ok_rows[0].wall_seconds
        for row in ok_rows:
            if row.wall_seconds > 0:
                speedup_vs_batch1.append((row.batch_size, base_wall / row.wall_seconds))

    md_lines = [
        "# Study A MLX Batch Cutoff (Psych_Qwen_32B 4-bit)",
        "",
        "Summary of empirically best MLX batch settings for Study A bias generation on this Mac,",
        "using the Psych_Qwen_32B MLX 4-bit model and real Study A bias prompts.",
        "",
        "## File locations",
        "",
        f"- **Timestamped report:** `{md_report_path}`",
        f"- **Latest daily report:** `{md_report_latest_path}`",
        f"- **JSON (timestamped):** `{json_report_path}`",
        f"- **JSON (latest daily):** `{json_report_latest_path}`",
        f"- **Temp generations root:** `{temp_root}`",
        "",
        "## Benchmark configuration",
        "",
        f"- **Timestamp (UTC):** `{report_payload['timestamp_utc']}`",
        f"- **Model ID:** `{args.model_id}`",
        f"- **Model path:** `{model_path}`",
        f"- **Data path:** `{data_path}`",
        f"- **Cases benchmarked:** `{len(prompts)}`",
        f"- **Max tokens per case:** `{args.max_tokens}`",
        f"- **Batch sizes tested:** `{', '.join(str(v) for v in report_payload['batch_sizes'])}`",
        f"- **Trials per batch size:** `{report_payload['trials']}`",
        f"- **Improvement threshold:** `{args.improvement_threshold:.0%}`",
        "",
        "## Batch-wise results",
        "",
        "| Batch | Status | Wall (s) | Cases/s | Prompt TPS | Gen TPS | Peak Mem (GB) | Error |",
        "|---:|:---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in sweep_results:
        md_lines.append(
            f"| {row.batch_size} | {row.status} | {row.wall_seconds:.2f} | "
            f"{row.cases_per_second:.3f} | {row.avg_prompt_tps:.1f} | "
            f"{row.avg_generation_tps:.1f} | {row.avg_peak_memory_gb:.2f} | "
            f"{row.error or '-'} |"
        )
    md_lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- **Threshold-cutoff recommended batch:** **`{recommended_batch}`**",
            f"- **Fastest observed batch:** `{fastest_batch}`",
            (
                f"- **Wall-clock chain (seconds):** {wall_clock_chain}"
                if wall_clock_chain
                else "- **Wall-clock chain (seconds):** n/a"
            ),
            (
                "- **Speed-up vs batch 1:** "
                + ", ".join(f"{batch}x={speedup:.2f}x" for batch, speedup in speedup_vs_batch1)
                if speedup_vs_batch1
                else "- **Speed-up vs batch 1:** n/a"
            ),
            "",
            "Rationale:",
            "",
            (
                f"- Batch `{recommended_batch}` is the first point where incremental gains fall below "
                f"`{args.improvement_threshold:.0%}` versus the previous batch."
                if recommended_batch is not None
                else "- No valid recommendation was produced (all sweeps failed)."
            ),
            (
                f"- Batch `{fastest_batch}` is fastest in this run, but use it only if memory headroom and stability remain acceptable."
                if fastest_batch is not None
                else "- No fastest batch available."
            ),
            "",
            "## Practical guidance",
            "",
            f"- Use batch `{recommended_batch}` as the default production setting on this machine.",
            (
                f"- If you want absolute throughput and can tolerate narrower headroom, batch `{fastest_batch}` is viable on this run."
                if fastest_batch is not None and recommended_batch != fastest_batch
                else "- Current fastest and recommended batch are aligned."
            ),
            "- Re-run this benchmark after major model/runtime changes (new MLX version, OS update, or model conversion changes).",
        ]
    )
    markdown_report = "\n".join(md_lines) + "\n"
    md_report_path.write_text(markdown_report, encoding="utf-8")
    md_report_latest_path.write_text(markdown_report, encoding="utf-8")

    print("\nBenchmark complete.")
    print(f"JSON report: {json_report_path}")
    print(f"Markdown report: {md_report_path}")
    if recommended_batch is not None:
        print(f"Recommended batch size: {recommended_batch}")


if __name__ == "__main__":
    main()
