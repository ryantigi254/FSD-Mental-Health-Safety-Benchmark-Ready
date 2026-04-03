"""Stress-test concurrent workers against LM Studio.

Ramps from 1 to MAX_WORKERS concurrent requests, measuring per-request
latency and system memory pressure at each concurrency level.  Designed
for mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2 on Apple
Silicon — will abort early if memory usage crosses 90 % to protect the
Mac from swapping/crashing.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import requests

# ── Config ───────────────────────────────────────────────────────────
MODEL_ID = "mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2"
BASE_URL = os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234/v1/chat/completions")
MAX_WORKERS = 8
MEMORY_CEILING_PCT = 90  # abort if used memory exceeds this %

PROMPT = (
    "A 34-year-old woman presents with persistent low mood, anhedonia, "
    "insomnia, and difficulty concentrating for the past three months. "
    "She reports no suicidal ideation. Provide a brief differential "
    "diagnosis and next steps."
)
SYSTEM = (
    "You are a clinical reasoning assistant. "
    "Respond with concise, evidence-based reasoning."
)


# ── Helpers ──────────────────────────────────────────────────────────
@dataclass
class WorkerResult:
    worker_id: int
    success: bool
    latency_s: float
    tokens: int = 0
    error: str = ""


@dataclass
class ConcurrencyResult:
    n_workers: int
    results: list[WorkerResult] = field(default_factory=list)
    mem_before_gb: float = 0.0
    mem_after_gb: float = 0.0
    mem_peak_gb: float = 0.0


def get_used_memory_gb() -> float:
    """Return used memory in GB via vm_stat (macOS)."""
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        page_size = 16384  # Apple Silicon default
        used_pages = 0
        for line in out.splitlines():
            for key in (
                "Pages active",
                "Pages wired down",
                "Pages speculative",
                "Pages occupied by compressor",
            ):
                if line.startswith(key):
                    used_pages += int(line.split(":")[1].strip().rstrip("."))
        return (used_pages * page_size) / (1024 ** 3)
    except Exception:
        return -1.0


def get_total_memory_gb() -> float:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return int(out.strip()) / (1024 ** 3)
    except Exception:
        return 48.0  # fallback


def send_request(worker_id: int) -> WorkerResult:
    """Fire a single chat completion request."""
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT},
        ],
        "temperature": 0.6,
        "stream": False,
    }
    start = time.perf_counter()
    try:
        resp = requests.post(
            BASE_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.perf_counter() - start
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return WorkerResult(worker_id=worker_id, success=True, latency_s=elapsed, tokens=tokens)
    except Exception as e:
        elapsed = time.perf_counter() - start
        return WorkerResult(worker_id=worker_id, success=False, latency_s=elapsed, error=str(e))


# ── Main ─────────────────────────────────────────────────────────────
def main() -> None:
    total_mem = get_total_memory_gb()
    print(f"System RAM: {total_mem:.0f} GB")
    print(f"Memory ceiling: {MEMORY_CEILING_PCT}% ({total_mem * MEMORY_CEILING_PCT / 100:.1f} GB)")
    print(f"Model: {MODEL_ID}")
    print(f"Max tokens: server default (no client-side limit)")
    print("=" * 72)

    all_results: list[ConcurrencyResult] = []

    for n in range(1, MAX_WORKERS + 1):
        mem_before = get_used_memory_gb()
        used_pct = (mem_before / total_mem) * 100

        print(f"\n── {n} worker(s) ──")
        print(f"  Memory before: {mem_before:.1f} GB ({used_pct:.0f}%)")

        if used_pct > MEMORY_CEILING_PCT:
            print(f"  ABORT: memory already at {used_pct:.0f}% — exceeds {MEMORY_CEILING_PCT}% ceiling")
            break

        cr = ConcurrencyResult(n_workers=n, mem_before_gb=mem_before)

        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {pool.submit(send_request, i): i for i in range(n)}
            peak_mem = mem_before
            for fut in as_completed(futures):
                result = fut.result()
                cr.results.append(result)
                current_mem = get_used_memory_gb()
                if current_mem > peak_mem:
                    peak_mem = current_mem

        mem_after = get_used_memory_gb()
        cr.mem_after_gb = mem_after
        cr.mem_peak_gb = peak_mem
        all_results.append(cr)

        successes = [r for r in cr.results if r.success]
        failures = [r for r in cr.results if not r.success]
        latencies = [r.latency_s for r in successes]
        tokens = [r.tokens for r in successes]

        print(f"  Memory after:  {mem_after:.1f} GB ({mem_after / total_mem * 100:.0f}%)")
        print(f"  Memory peak:   {peak_mem:.1f} GB ({peak_mem / total_mem * 100:.0f}%)")
        print(f"  Success: {len(successes)}/{n}  |  Failures: {len(failures)}/{n}")
        if latencies:
            avg_lat = sum(latencies) / len(latencies)
            max_lat = max(latencies)
            min_lat = min(latencies)
            avg_tok = sum(tokens) / len(tokens) if tokens else 0
            print(f"  Latency: min={min_lat:.1f}s  avg={avg_lat:.1f}s  max={max_lat:.1f}s")
            print(f"  Tokens:  avg={avg_tok:.0f}")
        for f in failures:
            print(f"  ERROR worker {f.worker_id}: {f.error[:120]}")

        # Safety check after run
        post_pct = (mem_after / total_mem) * 100
        if post_pct > MEMORY_CEILING_PCT:
            print(f"  STOP: post-run memory at {post_pct:.0f}% — halting before next level")
            break

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY")
    print(f"{'Workers':<10}{'OK':<6}{'Fail':<6}{'Avg lat(s)':<12}{'Max lat(s)':<12}{'Mem peak(GB)':<14}{'Mem %':<8}")
    print("-" * 72)
    for cr in all_results:
        ok = sum(1 for r in cr.results if r.success)
        fail = sum(1 for r in cr.results if not r.success)
        lats = [r.latency_s for r in cr.results if r.success]
        avg_l = sum(lats) / len(lats) if lats else 0
        max_l = max(lats) if lats else 0
        pct = cr.mem_peak_gb / total_mem * 100
        print(f"{cr.n_workers:<10}{ok:<6}{fail:<6}{avg_l:<12.1f}{max_l:<12.1f}{cr.mem_peak_gb:<14.1f}{pct:<8.0f}")


if __name__ == "__main__":
    main()
