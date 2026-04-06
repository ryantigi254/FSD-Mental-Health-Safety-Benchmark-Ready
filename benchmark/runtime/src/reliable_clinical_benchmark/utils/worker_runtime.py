"""Worker runtime helpers for generation pipelines."""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Sequence, Tuple, TypeVar

logger = logging.getLogger(__name__)

JobType = TypeVar("JobType")
ResultType = TypeVar("ResultType")


def is_lmstudio_runner(runner: object) -> bool:
    """Return True when a runner looks like an LM Studio-backed runner."""
    return hasattr(runner, "api_base")


def is_vllm_runner(runner: object) -> bool:
    """Return True when a runner is a vLLM-backed runner.

    vLLM handles batching internally via continuous batching, so
    client-side parallelism should be disabled to avoid contention.
    """
    try:
        from reliable_clinical_benchmark.models.vllm_runner import VLLMRunner
        return isinstance(runner, VLLMRunner)
    except ImportError:
        return False


def _gpt_oss_lmstudio_max_workers() -> int:
    """Upper bound for GPT-OSS via LM Studio (concurrent chat often crashes the engine)."""
    raw = os.getenv("LMSTUDIO_GPT_OSS_MAX_WORKERS", "1").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def supports_parallel_workers(runner: object) -> bool:
    """Return True when client-side thread parallelism is safe for this runner.

    LM Studio / Ollama API runners benefit from concurrent requests.
    vLLM runners do NOT — they batch internally and client-side threads
    add contention without throughput gain.
    """
    if is_vllm_runner(runner):
        return False
    return is_lmstudio_runner(runner)


def resolve_worker_count(
    requested_workers: Optional[int],
    runner: object,
    lmstudio_default: int = 4,
    non_lm_default: int = 1,
    log: Optional[logging.Logger] = None,
) -> int:
    """Resolve effective worker count with fail-closed gating."""
    target_log = log or logger
    if requested_workers is None:
        worker_count = lmstudio_default if supports_parallel_workers(runner) else non_lm_default
    else:
        worker_count = max(1, int(requested_workers))

    if worker_count > 1 and not supports_parallel_workers(runner):
        target_log.info(
            "Parallel workers >1 are only enabled for LM Studio/Ollama API runners "
            "(not vLLM). Falling back to 1 worker."
        )
        return 1

    if is_lmstudio_runner(runner):
        try:
            from reliable_clinical_benchmark.models.lmstudio_gpt_oss import GPTOSSLMStudioRunner

            if isinstance(runner, GPTOSSLMStudioRunner):
                gpt_oss_cap = _gpt_oss_lmstudio_max_workers()
                if worker_count > gpt_oss_cap:
                    target_log.info(
                        "Capping LM Studio workers from %d to %d for GPT-OSS "
                        "(concurrent requests often crash the backend; set "
                        "LMSTUDIO_GPT_OSS_MAX_WORKERS to raise this cap).",
                        worker_count,
                        gpt_oss_cap,
                    )
                    worker_count = gpt_oss_cap
        except ImportError:
            pass

        try:
            from reliable_clinical_benchmark.models.lmstudio_client import (
                get_loaded_model_runtime_limits,
            )

            runtime_limits = get_loaded_model_runtime_limits(
                getattr(runner, "api_base"),
                getattr(runner, "model_name"),
                timeout=5,
            )
            loaded_parallel = runtime_limits.get("parallel")
            if isinstance(loaded_parallel, int) and loaded_parallel > 0 and worker_count > loaded_parallel:
                target_log.info(
                    "Capping LM Studio workers from %d to loaded parallel=%d for %s.",
                    worker_count,
                    loaded_parallel,
                    getattr(runner, "model_name", "<unknown>"),
                )
                worker_count = loaded_parallel
        except Exception as error:
            target_log.debug("Could not inspect LM Studio loaded parallel setting: %s", error)

    return max(1, worker_count)


def append_jsonl_with_retry(
    cache_path: Path,
    entry: Dict,
    max_attempts: int = 3,
    base_sleep_seconds: float = 0.2,
    log: Optional[logging.Logger] = None,
) -> bool:
    """Append one JSONL entry with bounded retry for transient filesystem failures."""
    target_log = log or logger
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_attempts + 1):
        try:
            with cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            return True
        except OSError as error:
            target_log.warning(
                "JSONL write failed for %s (attempt %d/%d): %s",
                entry.get("id") or entry.get("case_id") or "<unknown>",
                attempt,
                max_attempts,
                error,
            )
            time.sleep(base_sleep_seconds * attempt)

    return False


def iter_threaded_results(
    jobs: Sequence[JobType],
    worker_count: int,
    worker_fn: Callable[[JobType], ResultType],
    progress_interval_seconds: int = 10,
    progress_label: str = "jobs",
    log: Optional[logging.Logger] = None,
) -> Iterator[Tuple[JobType, ResultType]]:
    """Yield completed job results using sequential or threaded execution."""
    target_log = log or logger
    total_jobs = len(jobs)
    if total_jobs == 0:
        return

    if worker_count <= 1:
        for job in jobs:
            yield job, worker_fn(job)
        return

    heartbeat_seconds = max(1, int(progress_interval_seconds))
    completed_count = 0
    pending_iter = iter(jobs)
    future_to_job: Dict = {}

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for _ in range(min(worker_count, total_jobs)):
            next_job = next(pending_iter, None)
            if next_job is None:
                break
            submitted = executor.submit(worker_fn, next_job)
            future_to_job[submitted] = next_job

        while future_to_job:
            finished_now, _ = wait(
                set(future_to_job.keys()),
                timeout=heartbeat_seconds,
                return_when=FIRST_COMPLETED,
            )
            if not finished_now:
                in_flight = len(future_to_job)
                queued = max(0, total_jobs - completed_count - in_flight)
                target_log.info(
                    "[progress:%s] completed=%d/%d in_flight=%d queued=%d",
                    progress_label,
                    completed_count,
                    total_jobs,
                    in_flight,
                    queued,
                )
                continue

            for finished in finished_now:
                job = future_to_job.pop(finished)
                result = finished.result()
                completed_count += 1
                yield job, result

                next_job = next(pending_iter, None)
                if next_job is not None:
                    submitted = executor.submit(worker_fn, next_job)
                    future_to_job[submitted] = next_job
