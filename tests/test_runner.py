import os
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bench.constants import PHASE_B5_RESULTS_DIR
from bench.contamination.bloom import BloomFilter
from bench.loaders.base import CodeDocument, EvalExample
from bench.retrievers.base import SearchResult
from bench.runners import run_benchmark
from bench.runners.run_benchmark import (
    _multiprocessing_context,
    _phase_b_findings_and_gates,
    _phase_b_retriever_factories,
    _phase_b5_findings_and_gates,
    _phase_b5_retriever_factories,
    _run_cpu_retriever_in_subprocess,
    run_phase_a_quickcheck,
    run_phase_b_python_full,
)


def test_phase_a_quickcheck_requires_contamination_filter(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.bin"

    with pytest.raises(FileNotFoundError, match="Contamination checks are mandatory"):
        run_phase_a_quickcheck(bloom_path=missing_path, sample_size=1, top_k=5, output_dir=tmp_path / "out")


def test_phase_a_quickcheck_rejects_bloom_without_canary(tmp_path: Path) -> None:
    wrong_bloom = BloomFilter.create(expected_items=10, false_positive_rate=0.01)
    wrong_bloom.add("deadbeef" * 8)
    bloom_path = tmp_path / "wrong.bin"
    wrong_bloom.save(bloom_path)

    with pytest.raises(ValueError, match="canary"):
        run_phase_a_quickcheck(bloom_path=bloom_path, sample_size=1, top_k=5, output_dir=tmp_path / "out")


def test_phase_b_python_full_requires_contamination_filter_before_retrievers(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.bin"

    with pytest.raises(FileNotFoundError, match="Contamination checks are mandatory"):
        run_phase_b_python_full(bloom_path=missing_path, sample_size=1, top_k=10, output_dir=tmp_path / "out")


def test_phase_b_retriever_factories_are_lazy_and_keep_cpu_rows_before_neural_models() -> None:
    factories = _phase_b_retriever_factories()

    assert all(callable(factory) for factory in factories)
    assert [factory.retriever_name for factory in factories[:3]] == ["ripgrep", "bm25", "sieve"]
    assert [factory.retriever_name for factory in factories[3:]] == ["codebert", "unixcoder", "lateon-code-edge", "lateon-code"]
    assert all(factory.run_in_subprocess for factory in factories[:3])
    assert not any(factory.run_in_subprocess for factory in factories[3:])
    assert not any(hasattr(factory, "_documents") for factory in factories)


def test_phase_b5_uses_same_retriever_set_as_b3() -> None:
    assert [factory.retriever_name for factory in _phase_b5_retriever_factories()] == [
        factory.retriever_name for factory in _phase_b_retriever_factories()
    ]


def test_phase_b5_runner_outputs_to_correct_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    def fake_run_phase_b5_python_full(
        *, bloom_path: Path, sample_size: int | None, top_k: int, output_dir: Path, cpu_timeout_seconds: float
    ):
        del bloom_path, sample_size, top_k, cpu_timeout_seconds
        calls.append(output_dir)
        return {
            "retriever_summaries": [
                {"retriever": "ripgrep", "recall@5": 0.35},
                {"retriever": "bm25", "recall@5": 0.40},
                {"retriever": "unixcoder", "recall@5": 0.55},
                {"retriever": "lateon-code-edge", "recall@5": 0.65},
                {"retriever": "lateon-code", "recall@5": 0.75},
                {"retriever": "codebert", "recall@5": 0.01},
                {"retriever": "sieve", "recall@5": 0.00},
            ]
        }

    monkeypatch.setattr(run_benchmark, "run_phase_b5_python_full", fake_run_phase_b5_python_full)

    result = CliRunner().invoke(run_benchmark.app, ["phase-b5-python-full"])

    assert result.exit_code == 0, result.output
    assert calls == [PHASE_B5_RESULTS_DIR]


class _PidRecordingRetriever:
    name = "pid-recorder"
    display_name = "PID Recorder"

    def __init__(self) -> None:
        self._documents: tuple[CodeDocument, ...] = ()

    def index(self, corpus: tuple[CodeDocument, ...]) -> None:
        self._documents = tuple(corpus)

    def search(self, query: str, k: int) -> list[SearchResult]:
        del query, k
        document = self._documents[0]
        return [
            SearchResult(
                document_id=document.document_id,
                path=document.path,
                code=document.code,
                score=1.0,
                metadata={"pid": os.getpid()},
            )
        ]


def _pid_recording_retriever_factory() -> _PidRecordingRetriever:
    return _PidRecordingRetriever()


def test_cpu_memory_measurement_subprocess_isolation() -> None:
    document = CodeDocument(document_id="doc-1", path="doc_1.py", code="def answer(): return 1", language="python", index_text="answer return")
    example = EvalExample(
        query="answer",
        ground_truth_code=document.code,
        ground_truth_path=document.path,
        language="python",
        source="unit",
        corpus_id="unit-corpus",
        metadata={"query_id": "q1", "ground_truth_document_id": document.document_id},
    )

    summary, rows = _run_cpu_retriever_in_subprocess(
        retriever_factory=_pid_recording_retriever_factory,
        corpus=(document,),
        accepted_examples=[(example, "hash")],
        top_k=1,
    )

    assert summary["retriever"] == "pid-recorder"
    assert rows[0]["retriever"] == "pid-recorder"
    assert rows[0]["top_k_result_document_ids"] == ["doc-1"]
    assert summary["memory_mb"] >= 0.0
    assert summary["memory_measurement"]["process"]["mode"] == "subprocess"
    assert summary["memory_measurement"]["process"]["pid"] != os.getpid()


class _SlowIndexRetriever:
    name = "slow-index"
    display_name = "Slow Index"

    def index(self, corpus: tuple[CodeDocument, ...]) -> None:
        del corpus
        time.sleep(10.0)

    def search(self, query: str, k: int) -> list[SearchResult]:
        del query, k
        return []


def _slow_index_retriever_factory() -> _SlowIndexRetriever:
    return _SlowIndexRetriever()


def test_cpu_memory_measurement_subprocess_timeout_terminates_child() -> None:
    document = CodeDocument(document_id="doc-1", path="doc_1.py", code="def answer(): return 1", language="python", index_text="answer return")
    example = EvalExample(
        query="answer",
        ground_truth_code=document.code,
        ground_truth_path=document.path,
        language="python",
        source="unit",
        corpus_id="unit-corpus",
        metadata={"query_id": "q1", "ground_truth_document_id": document.document_id},
    )

    with pytest.raises(TimeoutError, match="CPU retriever subprocess exceeded"):
        _run_cpu_retriever_in_subprocess(
            retriever_factory=_slow_index_retriever_factory,
            corpus=(document,),
            accepted_examples=[(example, "hash")],
            top_k=1,
            timeout_seconds=0.05,
        )


def test_cpu_memory_measurement_subprocess_uses_spawn_for_clean_rss_baseline() -> None:
    assert _multiprocessing_context().get_start_method() == "spawn"


def _retriever_summary(name: str, recall5: float, recall10: float | None = None) -> dict[str, float | str]:
    return {
        "retriever": name,
        "display_name": name,
        "recall@5": recall5,
        "recall@10": recall5 if recall10 is None else recall10,
    }


def test_phase_b_v2_sanity_gates_require_retrieval_trained_ordering() -> None:
    summaries = [
        _retriever_summary("ripgrep", 0.20),
        _retriever_summary("bm25", 0.40),
        _retriever_summary("codebert", 0.01),
        _retriever_summary("unixcoder", 0.50),
        _retriever_summary("lateon-code-edge", 0.60),
        _retriever_summary("lateon-code", 0.70),
        _retriever_summary("sieve", 0.11, recall10=0.19),
    ]

    findings, gates = _phase_b_findings_and_gates(summaries, corpus_document_count=1000)

    assert findings == [
        "CodeBERT null baseline stayed near zero as expected (Recall@5=0.010).",
        "SIEVE real engine row included with Recall@5=0.110; quality is expected to move after Phase 1 weights replace random/local ONNX exports.",
    ]
    assert gates["unixcoder_beats_bm25_recall@5"]["passed"] is True
    assert gates["lateon_code_edge_beats_unixcoder_recall@5"]["passed"] is True
    assert gates["lateon_code_beats_lateon_code_edge_recall@5"]["passed"] is True
    assert gates["codebert_null_baseline_recall@5_lt_0.05"]["passed"] is True


def test_phase_b_v2_sanity_gates_fail_closed_for_bad_lateon_ordering() -> None:
    summaries = [
        _retriever_summary("ripgrep", 0.20),
        _retriever_summary("bm25", 0.40),
        _retriever_summary("codebert", 0.01),
        _retriever_summary("unixcoder", 0.50),
        _retriever_summary("lateon-code-edge", 0.45),
        _retriever_summary("lateon-code", 0.70),
        _retriever_summary("sieve", 0.11, recall10=0.19),
    ]

    with pytest.raises(RuntimeError, match="LateOn-Code-edge sanity gate failed"):
        _phase_b_findings_and_gates(summaries, corpus_document_count=1000)


def test_phase_b5_sanity_gates_are_observational_for_raw_surface_distribution() -> None:
    summaries = [
        _retriever_summary("ripgrep", 0.35),
        _retriever_summary("bm25", 0.60),
        _retriever_summary("codebert", 0.07),
        _retriever_summary("unixcoder", 0.55),
        _retriever_summary("lateon-code-edge", 0.52),
        _retriever_summary("lateon-code", 0.51),
        _retriever_summary("sieve", 0.00),
    ]

    findings, gates = _phase_b5_findings_and_gates(summaries)

    assert findings
    assert gates["unixcoder_beats_bm25_recall@5"]["passed"] is False
    assert gates["unixcoder_beats_bm25_recall@5"]["fatal"] is False
    assert gates["lateon_code_edge_beats_unixcoder_recall@5"]["passed"] is False
    assert gates["codebert_null_baseline_recall@5_lt_0.05"]["passed"] is False
    assert gates["codebert_null_baseline_recall@5_lt_0.05"]["fatal"] is False
