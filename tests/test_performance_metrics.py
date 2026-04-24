from __future__ import annotations

from dataclasses import dataclass

from bench.metrics.performance import (
    current_cpu_rss_mebibytes,
    measure_cpu_peak_rss,
    measure_cpu_retriever_delta_rss,
    measure_torch_cuda_peak_allocated,
    ru_maxrss_to_mebibytes,
    summarize_performance,
)


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


def test_platform_detection_correctly_converts_ru_maxrss_units() -> None:
    assert ru_maxrss_to_mebibytes(2048, platform="linux") == 2.0
    assert ru_maxrss_to_mebibytes(2048, platform="linux2") == 2.0
    assert ru_maxrss_to_mebibytes(3 * 1024 * 1024, platform="darwin") == 3.0


def test_current_cpu_rss_reads_linux_vmrss_instead_of_high_water(tmp_path) -> None:
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmPeak:\t999999 kB\nVmRSS:\t20480 kB\n", encoding="utf-8")

    rss = current_cpu_rss_mebibytes(
        status_path=status,
        getrusage_fn=lambda _who: _FakeUsage(ru_maxrss=999_999 * 1024),
    )

    assert rss == 20.0


def test_cpu_memory_measurement_uses_delta_not_absolute() -> None:
    rss_values = iter([1_000.0, 1_050.0, 1_075.0])
    calls: list[str] = []

    def index() -> None:
        calls.append("index")

    def search() -> str:
        calls.append("search")
        return "searched"

    result, measurements = measure_cpu_retriever_delta_rss(index, search, current_rss_fn=lambda: next(rss_values))

    assert result == "searched"
    assert calls == ["index", "search"]
    assert measurements["index"].baseline_mb == 1_000.0
    assert measurements["index"].peak_mb == 1_050.0
    assert measurements["index"].delta_mb == 50.0
    assert measurements["total"].baseline_mb == 1_000.0
    assert measurements["total"].peak_mb == 1_075.0
    assert measurements["total"].delta_mb == 75.0


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


def test_gpu_memory_measurement_unchanged_for_cuda_retrievers() -> None:
    result, measurement = measure_torch_cuda_peak_allocated(lambda: "cuda-ok", torch_module=_FakeTorch(), device="cuda:0")

    assert result == "cuda-ok"
    assert measurement.backend == "cuda-max-memory-allocated"
    assert measurement.peak_mb == 5.0
    assert measurement.delta_mb == 4.0
