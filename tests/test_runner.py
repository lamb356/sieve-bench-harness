import os
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bench.constants import (
    PHASE_B5_GO_RESULTS_DIR,
    PHASE_B5_RESULTS_DIR,
    PHASE_B5_RUST_RESULTS_DIR,
    PHASE_B5_TYPESCRIPT_RESULTS_DIR,
    PHASE_B_GO_RESULTS_DIR,
    PHASE_B_RUST_RESULTS_DIR,
    PHASE_B_TYPESCRIPT_RESULTS_DIR,
)
from bench.contamination.bloom import BloomFilter
from bench.loaders.base import CodeDocument, EvalExample, LoadedBenchmark
from bench.retrievers.base import SearchResult
from bench.runners import run_benchmark
from bench.runners.run_benchmark import (
    _build_language_sieve,
    _build_typescript_sieve,
    _multiprocessing_context,
    _phase_b_findings_and_gates,
    _phase_b_go_retriever_factories,
    _phase_b_retriever_factories,
    _phase_b_rust_retriever_factories,
    _phase_b5_findings_and_gates,
    _phase_b5_go_retriever_factories,
    _phase_b5_retriever_factories,
    _phase_b5_rust_retriever_factories,
    _phase_b_typescript_retriever_factories,
    _phase_b5_typescript_retriever_factories,
    _run_cpu_retriever_in_subprocess,
    _language_findings_and_gates,
    _typescript_findings_and_gates,
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


def test_phase_b_python_full_payload_pins_raw_document_surface(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    document = CodeDocument(
        document_id="doc-1",
        path="python/raw.py",
        code="def rawonlysignal():\n    return 'target'\n",
        language="python",
        index_text="metadata decoy only",
    )
    example = EvalExample(
        query="rawonlysignal",
        ground_truth_code=document.code,
        ground_truth_path=document.path,
        language="python",
        source="unit",
        corpus_id="unit-corpus",
        metadata={"query_id": "q1", "ground_truth_document_id": document.document_id},
    )
    loaded = LoadedBenchmark(
        source="unit",
        language="python",
        revision="test-revision",
        corpus_id="unit-corpus",
        corpus=(document,),
        examples=(example,),
    )

    class FakeLoader:
        def load(self, sample_size: int | None = None) -> LoadedBenchmark:
            assert sample_size == 1
            return loaded

    class RawSurfaceRetriever:
        name = "raw-surface"
        display_name = "Raw Surface"

        def index(self, corpus: tuple[CodeDocument, ...]) -> None:
            self._corpus = corpus

        def search(self, query: str, k: int) -> list[SearchResult]:
            del query, k
            return [
                SearchResult(
                    document_id=document.document_id,
                    path=document.path,
                    score=1.0,
                    code=self._corpus[0].code,
                )
            ]

    monkeypatch.setattr(run_benchmark, "CoIRPythonLoader", FakeLoader)
    monkeypatch.setattr(run_benchmark, "_require_bloom_filter", lambda bloom_path: set())
    monkeypatch.setattr(
        run_benchmark,
        "_phase_b_retriever_factories",
        lambda: [run_benchmark.PhaseBRetrieverFactory("raw-surface", RawSurfaceRetriever)],
    )
    monkeypatch.setattr(run_benchmark, "_phase_b_findings_and_gates", lambda summaries, corpus_document_count: ([], {}))
    monkeypatch.setattr(run_benchmark, "write_phase_b_reports", lambda payload, output_dir: None)

    payload = run_phase_b_python_full(bloom_path=tmp_path / "bloom.bin", sample_size=1, top_k=1, output_dir=tmp_path / "out")

    assert payload["benchmark"]["document_surface"] == "document.code"
    assert "normalized_surface" not in payload["benchmark"]


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


def test_phase_b_typescript_runner_outputs_correct_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    def fake_run_phase_b_typescript_full(
        *, bloom_path: Path, sample_size: int, top_k: int, output_dir: Path, corpus_sample_size: int | None
    ):
        del bloom_path, sample_size, top_k, corpus_sample_size
        calls.append(output_dir)
        return {
            "retriever_summaries": [
                {"retriever": "ripgrep", "recall@5": 0.30},
                {"retriever": "bm25", "recall@5": 0.50},
                {"retriever": "unixcoder", "recall@5": 0.55},
                {"retriever": "lateon-code-edge", "recall@5": 0.57},
                {"retriever": "lateon-code", "recall@5": 0.60},
                {"retriever": "codebert", "recall@5": 0.01},
                {"retriever": "sieve", "recall@5": 0.00},
            ]
        }

    monkeypatch.setattr(run_benchmark, "run_phase_b_typescript_full", fake_run_phase_b_typescript_full)

    result = CliRunner().invoke(run_benchmark.app, ["phase-b-typescript-full"])

    assert result.exit_code == 0, result.output
    assert calls == [PHASE_B_TYPESCRIPT_RESULTS_DIR]


def test_phase_b_typescript_runner_forwards_nondefault_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_phase_b_typescript_full(
        *, bloom_path: Path, sample_size: int, top_k: int, output_dir: Path, corpus_sample_size: int | None
    ):
        calls.append({"sample_size": sample_size, "top_k": top_k, "output_dir": output_dir, "corpus_sample_size": corpus_sample_size})
        return {
            "retriever_summaries": [
                {"retriever": "ripgrep", "recall@5": 0.30},
                {"retriever": "bm25", "recall@5": 0.50},
                {"retriever": "unixcoder", "recall@5": 0.55},
                {"retriever": "lateon-code-edge", "recall@5": 0.57},
                {"retriever": "lateon-code", "recall@5": 0.60},
                {"retriever": "codebert", "recall@5": 0.01},
                {"retriever": "sieve", "recall@5": 0.00},
            ]
        }

    monkeypatch.setattr(run_benchmark, "run_phase_b_typescript_full", fake_run_phase_b_typescript_full)

    result = CliRunner().invoke(
        run_benchmark.app,
        [
            "phase-b-typescript-full",
            "--sample-size",
            "7",
            "--top-k",
            "3",
            "--corpus-sample-size",
            "11",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [{"sample_size": 7, "top_k": 3, "output_dir": tmp_path, "corpus_sample_size": 11}]


def test_phase_b5_typescript_runner_outputs_correct_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    def fake_run_phase_b5_typescript_full(
        *,
        bloom_path: Path,
        sample_size: int | None,
        top_k: int,
        output_dir: Path,
        cpu_timeout_seconds: float,
        corpus_sample_size: int | None,
    ):
        del bloom_path, sample_size, top_k, cpu_timeout_seconds, corpus_sample_size
        calls.append(output_dir)
        return {
            "retriever_summaries": [
                {"retriever": "ripgrep", "recall@5": 0.30},
                {"retriever": "bm25", "recall@5": 0.50},
                {"retriever": "unixcoder", "recall@5": 0.55},
                {"retriever": "lateon-code-edge", "recall@5": 0.57},
                {"retriever": "lateon-code", "recall@5": 0.60},
                {"retriever": "codebert", "recall@5": 0.01},
                {"retriever": "sieve", "recall@5": 0.00},
            ]
        }

    monkeypatch.setattr(run_benchmark, "run_phase_b5_typescript_full", fake_run_phase_b5_typescript_full)

    result = CliRunner().invoke(run_benchmark.app, ["phase-b5-typescript-full"])

    assert result.exit_code == 0, result.output
    assert calls == [PHASE_B5_TYPESCRIPT_RESULTS_DIR]


def test_phase_b5_typescript_runner_forwards_nondefault_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_phase_b5_typescript_full(
        *,
        bloom_path: Path,
        sample_size: int | None,
        top_k: int,
        output_dir: Path,
        cpu_timeout_seconds: float,
        corpus_sample_size: int | None,
    ):
        calls.append(
            {
                "sample_size": sample_size,
                "top_k": top_k,
                "output_dir": output_dir,
                "cpu_timeout_seconds": cpu_timeout_seconds,
                "corpus_sample_size": corpus_sample_size,
            }
        )
        return {
            "retriever_summaries": [
                {"retriever": "ripgrep", "recall@5": 0.30},
                {"retriever": "bm25", "recall@5": 0.50},
                {"retriever": "unixcoder", "recall@5": 0.55},
                {"retriever": "lateon-code-edge", "recall@5": 0.57},
                {"retriever": "lateon-code", "recall@5": 0.60},
                {"retriever": "codebert", "recall@5": 0.01},
                {"retriever": "sieve", "recall@5": 0.00},
            ]
        }

    monkeypatch.setattr(run_benchmark, "run_phase_b5_typescript_full", fake_run_phase_b5_typescript_full)

    result = CliRunner().invoke(
        run_benchmark.app,
        [
            "phase-b5-typescript-full",
            "--sample-size",
            "9",
            "--top-k",
            "4",
            "--cpu-timeout-seconds",
            "12.5",
            "--corpus-sample-size",
            "13",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {"sample_size": 9, "top_k": 4, "output_dir": tmp_path, "cpu_timeout_seconds": 12.5, "corpus_sample_size": 13}
    ]


def test_phase_b_go_runner_outputs_correct_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    def fake_run_phase_b_go_full(
        *, bloom_path: Path, sample_size: int, top_k: int, output_dir: Path, corpus_sample_size: int | None
    ):
        del bloom_path, sample_size, top_k, corpus_sample_size
        calls.append(output_dir)
        return {
            "retriever_summaries": [
                {"retriever": "ripgrep", "recall@5": 0.30},
                {"retriever": "bm25", "recall@5": 0.50},
                {"retriever": "unixcoder", "recall@5": 0.55},
                {"retriever": "lateon-code-edge", "recall@5": 0.57},
                {"retriever": "lateon-code", "recall@5": 0.60},
                {"retriever": "codebert", "recall@5": 0.01},
                {"retriever": "sieve", "recall@5": 0.00},
            ]
        }

    monkeypatch.setattr(run_benchmark, "run_phase_b_go_full", fake_run_phase_b_go_full)

    result = CliRunner().invoke(run_benchmark.app, ["phase-b-go-full"])

    assert result.exit_code == 0, result.output
    assert calls == [PHASE_B_GO_RESULTS_DIR]


def test_phase_b5_go_runner_forwards_nondefault_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_phase_b5_go_full(
        *,
        bloom_path: Path,
        sample_size: int | None,
        top_k: int,
        output_dir: Path,
        cpu_timeout_seconds: float,
        corpus_sample_size: int | None,
    ):
        calls.append(
            {
                "sample_size": sample_size,
                "top_k": top_k,
                "output_dir": output_dir,
                "cpu_timeout_seconds": cpu_timeout_seconds,
                "corpus_sample_size": corpus_sample_size,
            }
        )
        return {
            "retriever_summaries": [
                {"retriever": "ripgrep", "recall@5": 0.30},
                {"retriever": "bm25", "recall@5": 0.50},
                {"retriever": "unixcoder", "recall@5": 0.55},
                {"retriever": "lateon-code-edge", "recall@5": 0.57},
                {"retriever": "lateon-code", "recall@5": 0.60},
                {"retriever": "codebert", "recall@5": 0.01},
                {"retriever": "sieve", "recall@5": 0.00},
            ]
        }

    monkeypatch.setattr(run_benchmark, "run_phase_b5_go_full", fake_run_phase_b5_go_full)

    result = CliRunner().invoke(
        run_benchmark.app,
        [
            "phase-b5-go-full",
            "--sample-size",
            "9",
            "--top-k",
            "4",
            "--cpu-timeout-seconds",
            "12.5",
            "--corpus-sample-size",
            "13",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {"sample_size": 9, "top_k": 4, "output_dir": tmp_path, "cpu_timeout_seconds": 12.5, "corpus_sample_size": 13}
    ]


def test_phase_b_rust_runner_outputs_correct_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    def fake_run_phase_b_rust_full(
        *, bloom_path: Path, sample_size: int, top_k: int, output_dir: Path, corpus_sample_size: int | None
    ):
        del bloom_path, sample_size, top_k, corpus_sample_size
        calls.append(output_dir)
        return {
            "retriever_summaries": [
                {"retriever": "ripgrep", "recall@5": 0.30},
                {"retriever": "bm25", "recall@5": 0.50},
                {"retriever": "unixcoder", "recall@5": 0.55},
                {"retriever": "lateon-code-edge", "recall@5": 0.57},
                {"retriever": "lateon-code", "recall@5": 0.60},
                {"retriever": "codebert", "recall@5": 0.01},
                {"retriever": "sieve", "recall@5": 0.00},
            ]
        }

    monkeypatch.setattr(run_benchmark, "run_phase_b_rust_full", fake_run_phase_b_rust_full)

    result = CliRunner().invoke(run_benchmark.app, ["phase-b-rust-full"])

    assert result.exit_code == 0, result.output
    assert calls == [PHASE_B_RUST_RESULTS_DIR]


def test_phase_b5_rust_runner_forwards_nondefault_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_phase_b5_rust_full(
        *,
        bloom_path: Path,
        sample_size: int | None,
        top_k: int,
        output_dir: Path,
        cpu_timeout_seconds: float,
        corpus_sample_size: int | None,
    ):
        calls.append(
            {
                "sample_size": sample_size,
                "top_k": top_k,
                "output_dir": output_dir,
                "cpu_timeout_seconds": cpu_timeout_seconds,
                "corpus_sample_size": corpus_sample_size,
            }
        )
        return {
            "retriever_summaries": [
                {"retriever": "ripgrep", "recall@5": 0.30},
                {"retriever": "bm25", "recall@5": 0.50},
                {"retriever": "unixcoder", "recall@5": 0.55},
                {"retriever": "lateon-code-edge", "recall@5": 0.57},
                {"retriever": "lateon-code", "recall@5": 0.60},
                {"retriever": "codebert", "recall@5": 0.01},
                {"retriever": "sieve", "recall@5": 0.00},
            ]
        }

    monkeypatch.setattr(run_benchmark, "run_phase_b5_rust_full", fake_run_phase_b5_rust_full)

    result = CliRunner().invoke(
        run_benchmark.app,
        [
            "phase-b5-rust-full",
            "--sample-size",
            "9",
            "--top-k",
            "4",
            "--cpu-timeout-seconds",
            "12.5",
            "--corpus-sample-size",
            "13",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {"sample_size": 9, "top_k": 4, "output_dir": tmp_path, "cpu_timeout_seconds": 12.5, "corpus_sample_size": 13}
    ]


def test_multilanguage_retriever_sets_match_python_b3() -> None:
    python_b3 = [factory.retriever_name for factory in _phase_b_retriever_factories()]

    assert [factory.retriever_name for factory in _phase_b_typescript_retriever_factories()] == python_b3
    assert [factory.retriever_name for factory in _phase_b5_typescript_retriever_factories()] == python_b3
    assert [factory.retriever_name for factory in _phase_b_go_retriever_factories()] == python_b3
    assert [factory.retriever_name for factory in _phase_b5_go_retriever_factories()] == python_b3
    assert [factory.retriever_name for factory in _phase_b_rust_retriever_factories()] == python_b3
    assert [factory.retriever_name for factory in _phase_b5_rust_retriever_factories()] == python_b3


def test_typescript_retriever_set_matches_python_b3() -> None:
    python_b3 = [factory.retriever_name for factory in _phase_b_retriever_factories()]

    assert [factory.retriever_name for factory in _phase_b_typescript_retriever_factories()] == python_b3
    assert [factory.retriever_name for factory in _phase_b5_typescript_retriever_factories()] == python_b3


def test_language_findings_mark_sieve_pending_and_name_language() -> None:
    summaries = [
        {"retriever": "ripgrep", "recall@5": 0.29},
        {"retriever": "bm25", "recall@5": 0.38},
        {"retriever": "unixcoder", "recall@5": 0.57},
        {"retriever": "lateon-code-edge", "recall@5": 0.54},
        {"retriever": "lateon-code", "recall@5": 0.81},
        {"retriever": "codebert", "recall@5": 0.0},
        {"retriever": "sieve", "recall@5": 0.0},
    ]

    findings, gates = _language_findings_and_gates(
        summaries,
        phase_label="Phase B v3 Go full eval",
        language_title="Go",
        dataset_note="official CoIR/CodeSearchNet Go test qrels",
    )

    assert all("B.5 records" not in finding for finding in findings)
    assert any("Phase B v3 Go full eval records Go behavior" in finding for finding in findings)
    assert any("official CoIR/CodeSearchNet Go test qrels" in finding for finding in findings)
    assert any("SIEVE Go row is labeled Phase 1 weights pending" in finding for finding in findings)
    assert {gate.get("scope") for gate in gates.values()} == {"observational-go"}


def test_typescript_findings_are_phase_neutral() -> None:
    summaries = [
        {"retriever": "ripgrep", "recall@5": 0.29},
        {"retriever": "bm25", "recall@5": 0.38},
        {"retriever": "unixcoder", "recall@5": 0.57},
        {"retriever": "lateon-code-edge", "recall@5": 0.54},
        {"retriever": "lateon-code", "recall@5": 0.81},
        {"retriever": "codebert", "recall@5": 0.0},
        {"retriever": "sieve", "recall@5": 0.0},
    ]

    findings, gates = _typescript_findings_and_gates(summaries, phase_label="Phase B v3 TypeScript full eval")

    assert all("B.5 records" not in finding for finding in findings)
    assert any("Phase B v3 TypeScript full eval records canonical TypeScript behavior" in finding for finding in findings)
    assert any("Shuu12121/typescript-treesitter-dedupe-filtered-datasetsV2" in finding for finding in findings)
    assert all("ArkTS" not in finding for finding in findings)
    assert {gate.get("scope") for gate in gates.values()} == {"observational-typescript"}


def test_multilanguage_sieve_factory_degrades_to_labeled_pending_row_when_cli_route_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIEVE_BINARY", raising=False)
    monkeypatch.delenv("SIEVE_REPO", raising=False)
    monkeypatch.setenv("PATH", "")

    for language_title in ("TypeScript", "Go", "Rust"):
        retriever = _build_language_sieve(language_title=language_title)
        assert retriever.name == "sieve"
        assert retriever.display_name == "SIEVE (Phase 1 weights pending)"
        assert retriever.embedding_metadata()["route_status"] == "sieve-cli-unavailable"
        assert retriever.embedding_metadata()["language"] == language_title


def test_typescript_sieve_factory_degrades_to_labeled_pending_row_when_cli_route_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIEVE_BINARY", raising=False)
    monkeypatch.delenv("SIEVE_REPO", raising=False)
    monkeypatch.setenv("PATH", "")

    retriever = _build_typescript_sieve()

    assert retriever.name == "sieve"
    assert retriever.display_name == "SIEVE (Phase 1 weights pending)"
    assert retriever.embedding_metadata()["route_status"] == "sieve-cli-unavailable"


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
