import argparse
import csv
import json
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import psutil


def _ensure_src_on_path(runtime_root: Path) -> None:
    src_dir = runtime_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _sample_gpu() -> dict:
    query = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(query, capture_output=True, text=True, check=True, timeout=10)
    except Exception:
        return {}

    first_row = next(csv.reader(result.stdout.splitlines()), [])
    if len(first_row) != 5:
        return {}
    gpu_util, mem_util, mem_used, mem_total, temp = [float(value.strip()) for value in first_row]
    return {
        "gpu_util_percent": gpu_util,
        "gpu_memory_util_percent": mem_util,
        "gpu_memory_used_mb": mem_used,
        "gpu_memory_total_mb": mem_total,
        "gpu_temperature_c": temp,
    }


class ResourceMonitor:
    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self.samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        psutil.cpu_percent(interval=None)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = {
                "timestamp": time.time(),
                "cpu_percent": psutil.cpu_percent(interval=None),
                "ram_percent": psutil.virtual_memory().percent,
            }
            sample.update(_sample_gpu())
            self.samples.append(sample)
            self._stop.wait(self.interval_seconds)


def _summarize_monitor(samples: list[dict]) -> dict:
    if not samples:
        return {}

    def _max(key: str) -> Optional[float]:
        values = [sample[key] for sample in samples if key in sample]
        return max(values) if values else None

    def _avg(key: str) -> Optional[float]:
        values = [sample[key] for sample in samples if key in sample]
        return statistics.mean(values) if values else None

    return {
        "avg_cpu_percent": _avg("cpu_percent"),
        "peak_cpu_percent": _max("cpu_percent"),
        "avg_ram_percent": _avg("ram_percent"),
        "peak_ram_percent": _max("ram_percent"),
        "avg_gpu_util_percent": _avg("gpu_util_percent"),
        "peak_gpu_util_percent": _max("gpu_util_percent"),
        "avg_gpu_memory_used_mb": _avg("gpu_memory_used_mb"),
        "peak_gpu_memory_used_mb": _max("gpu_memory_used_mb"),
        "gpu_memory_total_mb": _max("gpu_memory_total_mb"),
        "peak_gpu_temperature_c": _max("gpu_temperature_c"),
    }


def _run_round(runner, prompt: str, requests_per_round: int, worker_count: int, mode: str) -> dict:
    latencies_ms = []
    failures = 0

    def _call_once(_: int) -> str:
        start = time.perf_counter()
        try:
            return_text = runner.generate(prompt, mode=mode)
            latencies_ms.append((time.perf_counter() - start) * 1000)
            return return_text
        except Exception:
            latencies_ms.append((time.perf_counter() - start) * 1000)
            raise

    monitor = ResourceMonitor(interval_seconds=1.0)
    monitor.start()
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_call_once, i) for i in range(requests_per_round)]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                failures += 1

    elapsed_seconds = time.perf_counter() - started
    monitor.stop()

    success_count = requests_per_round - failures
    return {
        "worker_count": worker_count,
        "requests": requests_per_round,
        "successes": success_count,
        "failures": failures,
        "elapsed_seconds": elapsed_seconds,
        "throughput_rps": (success_count / elapsed_seconds) if elapsed_seconds > 0 else 0.0,
        "avg_latency_ms": statistics.mean(latencies_ms) if latencies_ms else None,
        "p95_latency_ms": (
            statistics.quantiles(latencies_ms, n=20)[-1] if len(latencies_ms) >= 20 else max(latencies_ms, default=None)
        ),
        "resource_summary": _summarize_monitor(monitor.samples),
        "resource_samples": monitor.samples,
    }


def _score_round(result: dict) -> tuple:
    resources = result.get("resource_summary", {})
    failures = result.get("failures", 0)
    peak_gpu_mem = resources.get("peak_gpu_memory_used_mb") or 0.0
    peak_cpu = resources.get("peak_cpu_percent") or 0.0
    throughput = result.get("throughput_rps", 0.0)
    return (failures == 0, -peak_gpu_mem, -peak_cpu, throughput)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MedGemma worker scaling through LM Studio.")
    parser.add_argument("--api-base", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--api-identifier", default="google/medgemma-27b-it")
    parser.add_argument("--mode", choices=["cot", "direct"], default="direct")
    parser.add_argument("--min-workers", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--requests-per-round", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--prompt",
        default=(
            "Briefly acknowledge a patient feeling overwhelmed and suggest one grounding step. "
            "Keep it to two sentences."
        ),
    )
    parser.add_argument(
        "--stop-gpu-memory-percent",
        type=float,
        default=96.0,
        help="Stop after a round if peak GPU memory usage exceeds this percent of total.",
    )
    parser.add_argument(
        "--stop-cpu-percent",
        type=float,
        default=90.0,
        help="Stop after a round if peak CPU usage exceeds this threshold.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the benchmark summary JSON.",
    )
    args = parser.parse_args()

    runtime_root = Path(__file__).resolve().parents[1]
    _ensure_src_on_path(runtime_root)

    from reliable_clinical_benchmark.models.base import GenerationConfig
    from reliable_clinical_benchmark.models.lmstudio_medgemma import MedGemmaLMStudioRunner

    runner = MedGemmaLMStudioRunner(
        model_name=args.api_identifier,
        api_base=args.api_base,
        config=GenerationConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        ),
    )

    results = []
    best_result = None
    for worker_count in range(args.min_workers, args.max_workers + 1):
        result = _run_round(
            runner=runner,
            prompt=args.prompt,
            requests_per_round=args.requests_per_round,
            worker_count=worker_count,
            mode=args.mode,
        )
        results.append(result)

        if best_result is None:
            best_result = result
        else:
            current_score = _score_round(result)
            best_score = _score_round(best_result)
            if current_score[0] and (
                not best_score[0] or result["throughput_rps"] >= best_result["throughput_rps"] * 0.95
            ):
                peak_gpu_mem = result.get("resource_summary", {}).get("peak_gpu_memory_used_mb")
                best_gpu_mem = best_result.get("resource_summary", {}).get("peak_gpu_memory_used_mb")
                if best_gpu_mem is None or peak_gpu_mem is None or peak_gpu_mem <= best_gpu_mem:
                    best_result = result

        resource_summary = result.get("resource_summary", {})
        peak_gpu_mem = resource_summary.get("peak_gpu_memory_used_mb")
        total_gpu_mem = resource_summary.get("gpu_memory_total_mb")
        peak_cpu = resource_summary.get("peak_cpu_percent") or 0.0
        if peak_cpu >= args.stop_cpu_percent:
            break
        if peak_gpu_mem is not None and total_gpu_mem:
            peak_gpu_mem_percent = (peak_gpu_mem / total_gpu_mem) * 100.0
            if peak_gpu_mem_percent >= args.stop_gpu_memory_percent:
                break

    if best_result is None and results:
        best_result = results[0]

    payload = {
        "model_id": runner.model_name,
        "api_base": args.api_base,
        "mode": args.mode,
        "requests_per_round": args.requests_per_round,
        "results": results,
        "recommended_workers": best_result["worker_count"] if best_result else None,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
