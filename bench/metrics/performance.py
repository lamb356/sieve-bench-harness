from __future__ import annotations

import math
import resource
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
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


def ru_maxrss_to_mebibytes(value: int | float, *, platform: str | None = None) -> float:
    platform_name = sys.platform if platform is None else platform
    if platform_name == "darwin":
        # macOS reports resource.getrusage(...).ru_maxrss in bytes.
        return bytes_to_mebibytes(value)
    # Linux and WSL report ru_maxrss in KiB. Treat unknown Unix-like platforms
    # like Linux so the benchmark does not under-report by 1024x on WSL.
    return float(value) / 1024.0


def linux_ru_maxrss_to_mebibytes(value: int | float) -> float:
    # Backward-compatible name for older tests/callers; Linux and WSL report KiB.
    return ru_maxrss_to_mebibytes(value, platform="linux")


def _read_linux_current_rss_mb(status_path: str | Path = "/proc/self/status") -> float | None:
    try:
        with Path(status_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("VmRSS:"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    return None
                # /proc status reports VmRSS in kB (KiB in kernel accounting).
                return float(parts[1]) / 1024.0
    except OSError:
        return None
    return None


def current_cpu_rss_mebibytes(
    *,
    status_path: str | Path = "/proc/self/status",
    getrusage_fn: Callable[[int], object] = resource.getrusage,
) -> float:
    current_rss = _read_linux_current_rss_mb(status_path)
    if current_rss is not None:
        return current_rss
    # Fallback for non-Linux platforms: this is a high-water mark, not a true
    # current RSS value, but it preserves old behavior where /proc is absent.
    return _current_cpu_rss_mb(getrusage_fn)


def _current_cpu_rss_mb(getrusage_fn: Callable[[int], object]) -> float:
    usage = getrusage_fn(resource.RUSAGE_SELF)
    return ru_maxrss_to_mebibytes(getattr(usage, "ru_maxrss"))


def measure_cpu_peak_rss(
    fn: Callable[[], T],
    *,
    getrusage_fn: Callable[[int], object] = resource.getrusage,
) -> tuple[T, MemoryMeasurement]:
    before = getrusage_fn(resource.RUSAGE_SELF)
    baseline_mb = ru_maxrss_to_mebibytes(getattr(before, "ru_maxrss"))
    result = fn()
    after = getrusage_fn(resource.RUSAGE_SELF)
    peak_mb = ru_maxrss_to_mebibytes(getattr(after, "ru_maxrss"))
    return result, MemoryMeasurement(
        backend="cpu-rss",
        baseline_mb=baseline_mb,
        peak_mb=peak_mb,
        delta_mb=max(0.0, peak_mb - baseline_mb),
    )


def measure_cpu_retriever_delta_rss(
    index_fn: Callable[[], object],
    search_fn: Callable[[], T],
    *,
    current_rss_fn: Callable[[], float] | None = None,
    getrusage_fn: Callable[[int], object] = resource.getrusage,
) -> tuple[T, dict[str, MemoryMeasurement]]:
    if current_rss_fn is None:
        current_rss_fn = lambda: current_cpu_rss_mebibytes(getrusage_fn=getrusage_fn)
    baseline_mb = current_rss_fn()
    index_fn()
    post_index_mb = current_rss_fn()
    result = search_fn()
    post_search_mb = current_rss_fn()

    index_measurement = MemoryMeasurement(
        backend="cpu-rss-delta",
        baseline_mb=baseline_mb,
        peak_mb=post_index_mb,
        delta_mb=max(0.0, post_index_mb - baseline_mb),
    )
    search_measurement = MemoryMeasurement(
        backend="cpu-rss-delta",
        baseline_mb=post_index_mb,
        peak_mb=post_search_mb,
        delta_mb=max(0.0, post_search_mb - post_index_mb),
    )
    total_measurement = MemoryMeasurement(
        backend="cpu-rss-delta",
        baseline_mb=baseline_mb,
        peak_mb=post_search_mb,
        delta_mb=max(0.0, post_search_mb - baseline_mb),
    )
    return result, {
        "index": index_measurement,
        "search": search_measurement,
        "total": total_measurement,
    }


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
