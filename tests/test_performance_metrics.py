from __future__ import annotations

from bench.metrics.performance import summarize_performance


def test_performance_metrics_compute_nearest_rank_percentiles_and_throughput() -> None:
    summary = summarize_performance(
        latencies_ms=[10.0, 50.0, 100.0],
        query_count=3,
        total_search_seconds=0.16,
        index_build_seconds=0.25,
    )

    assert summary["p50"] == 50.0
    assert summary["p95"] == 100.0
    assert summary["p99"] == 100.0
    assert round(summary["throughput_qps"], 6) == round(3 / 0.16, 6)
    assert summary["index_build_seconds"] == 0.25


def test_performance_metrics_reject_empty_latency_fixture() -> None:
    try:
        summarize_performance(latencies_ms=[], query_count=0, total_search_seconds=0.0, index_build_seconds=0.0)
    except ValueError as exc:
        assert "latency" in str(exc).lower()
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("empty latency fixture should fail")
