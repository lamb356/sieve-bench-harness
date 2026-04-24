from __future__ import annotations

from dataclasses import dataclass

from bench.metrics.performance import measure_cpu_peak_rss, measure_torch_cuda_peak_allocated, summarize_performance


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


def test_performance_summary_includes_memory_when_provided() -> None:
    summary = summarize_performance(
        latencies_ms=[1.0],
        query_count=1,
        total_search_seconds=0.5,
        index_build_seconds=0.25,
        memory_mb=12.5,
        index_memory_mb=10.0,
        search_memory_mb=12.5,
    )

    assert summary["memory_mb"] == 12.5
    assert summary["index_memory_mb"] == 10.0
    assert summary["search_memory_mb"] == 12.5


@dataclass(frozen=True)
class _FakeUsage:
    ru_maxrss: int


def test_cpu_memory_measurement_converts_known_linux_rss_fixture() -> None:
    usages = iter([_FakeUsage(ru_maxrss=1024), _FakeUsage(ru_maxrss=3072)])

    result, measurement = measure_cpu_peak_rss(lambda: "ok", getrusage_fn=lambda _who: next(usages))

    assert result == "ok"
    assert measurement.backend == "cpu-rss"
    assert measurement.baseline_mb == 1.0
    assert measurement.peak_mb == 3.0
    assert measurement.delta_mb == 2.0


class _FakeCuda:
    def __init__(self) -> None:
        self.reset_called = False
        self.synchronize_calls = 0

    def synchronize(self, device=None) -> None:  # noqa: ANN001
        self.synchronize_calls += 1

    def memory_allocated(self, device=None) -> int:  # noqa: ANN001
        return 1 * 1024 * 1024

    def reset_peak_memory_stats(self, device=None) -> None:  # noqa: ANN001
        self.reset_called = True

    def max_memory_allocated(self, device=None) -> int:  # noqa: ANN001
        return 5 * 1024 * 1024


class _FakeTorch:
    cuda = _FakeCuda()


def test_cuda_memory_measurement_converts_known_allocated_bytes() -> None:
    result, measurement = measure_torch_cuda_peak_allocated(lambda: 7, torch_module=_FakeTorch(), device="cuda:0")

    assert result == 7
    assert measurement.backend == "cuda-max-memory-allocated"
    assert measurement.baseline_mb == 1.0
    assert measurement.peak_mb == 5.0
    assert measurement.delta_mb == 4.0
    assert _FakeTorch.cuda.reset_called
    assert _FakeTorch.cuda.synchronize_calls == 2
