from __future__ import annotations

import math
from collections.abc import Sequence


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
) -> dict[str, float]:
    if not latencies_ms:
        raise ValueError("Cannot summarize performance without latency samples")
    if query_count < 0:
        raise ValueError("query_count must be non-negative")
    throughput_qps = 0.0 if total_search_seconds <= 0.0 else float(query_count) / float(total_search_seconds)
    return {
        **summarize_latency(latencies_ms),
        "throughput_qps": throughput_qps,
        "index_build_seconds": float(index_build_seconds),
        "total_search_seconds": float(total_search_seconds),
    }
