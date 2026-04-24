from __future__ import annotations

import math
import resource
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class MemoryMeasurement:
    backend: str
    peak_mb: float
    baseline_mb: float
    delta_mb: float

    def to_json(self) -> dict[str, float | str]:
        return asdict(self)


def bytes_to_mebibytes(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def linux_ru_maxrss_to_mebibytes(value: int | float) -> float:
    # Linux and WSL report ru_maxrss in KiB. This benchmark harness runs on Linux/WSL.
    return float(value) / 1024.0


def measure_cpu_peak_rss(
    fn: Callable[[], T],
    *,
    getrusage_fn: Callable[[int], object] = resource.getrusage,
) -> tuple[T, MemoryMeasurement]:
    before = getrusage_fn(resource.RUSAGE_SELF)
    baseline_mb = linux_ru_maxrss_to_mebibytes(getattr(before, "ru_maxrss"))
    result = fn()
    after = getrusage_fn(resource.RUSAGE_SELF)
    peak_mb = linux_ru_maxrss_to_mebibytes(getattr(after, "ru_maxrss"))
    return result, MemoryMeasurement(
        backend="cpu-rss",
        baseline_mb=baseline_mb,
        peak_mb=peak_mb,
        delta_mb=max(0.0, peak_mb - baseline_mb),
    )


def measure_torch_cuda_peak_allocated(
    fn: Callable[[], T],
    *,
    torch_module: object | None = None,
    device: object | None = None,
) -> tuple[T, MemoryMeasurement]:
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]

    cuda = getattr(torch_module, "cuda")
    synchronize = getattr(cuda, "synchronize")
    memory_allocated = getattr(cuda, "memory_allocated")
    reset_peak_memory_stats = getattr(cuda, "reset_peak_memory_stats")
    max_memory_allocated = getattr(cuda, "max_memory_allocated")

    synchronize(device)
    baseline_mb = bytes_to_mebibytes(memory_allocated(device))
    reset_peak_memory_stats(device)
    result = fn()
    synchronize(device)
    peak_mb = bytes_to_mebibytes(max_memory_allocated(device))
    return result, MemoryMeasurement(
        backend="cuda-max-memory-allocated",
        baseline_mb=baseline_mb,
        peak_mb=peak_mb,
        delta_mb=max(0.0, peak_mb - baseline_mb),
    )


def nearest_rank_percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot compute latency percentile for an empty latency fixture")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"Percentile fraction must be in [0, 1], got {fraction}")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(len(ordered) - 1, rank - 1)]


def summarize_latency(latencies_ms: Sequence[float]) -> dict[str, float]:
    return {
        "p50": nearest_rank_percentile(latencies_ms, 0.50),
        "p95": nearest_rank_percentile(latencies_ms, 0.95),
        "p99": nearest_rank_percentile(latencies_ms, 0.99),
    }


def summarize_performance(
    *,
    latencies_ms: Sequence[float],
    query_count: int,
    total_search_seconds: float,
    index_build_seconds: float,
    memory_mb: float | None = None,
    index_memory_mb: float | None = None,
    search_memory_mb: float | None = None,
) -> dict[str, float]:
    if not latencies_ms:
        raise ValueError("Cannot summarize performance without latency samples")
    if query_count < 0:
        raise ValueError("query_count must be non-negative")
    throughput_qps = 0.0 if total_search_seconds <= 0.0 else float(query_count) / float(total_search_seconds)
    summary = {
        **summarize_latency(latencies_ms),
        "throughput_qps": throughput_qps,
        "index_build_seconds": float(index_build_seconds),
        "total_search_seconds": float(total_search_seconds),
    }
    if memory_mb is not None:
        summary["memory_mb"] = float(memory_mb)
    if index_memory_mb is not None:
        summary["index_memory_mb"] = float(index_memory_mb)
    if search_memory_mb is not None:
        summary["search_memory_mb"] = float(search_memory_mb)
    return summary
