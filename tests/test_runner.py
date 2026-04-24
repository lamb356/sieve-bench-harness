from pathlib import Path

import pytest

from bench.contamination.bloom import BloomFilter
from bench.runners.run_benchmark import _phase_b_findings_and_gates, _phase_b_retriever_factories, run_phase_a_quickcheck, run_phase_b_python_full


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
    assert [getattr(factory, "__name__", "") for factory in factories[1:4]] == ["BM25Retriever", "SieveStubRetriever", "CodeBERTRetriever"]
    assert getattr(factories[0], "__name__", "") == "<lambda>"
    assert not any(hasattr(factory, "_documents") for factory in factories)


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
        _retriever_summary("sieve-stub", 0.00, recall10=0.001),
    ]

    findings, gates = _phase_b_findings_and_gates(summaries, corpus_document_count=1000)

    assert findings == ["CodeBERT null baseline stayed near zero as expected (Recall@5=0.010)."]
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
        _retriever_summary("sieve-stub", 0.00, recall10=0.001),
    ]

    with pytest.raises(RuntimeError, match="LateOn-Code-edge sanity gate failed"):
        _phase_b_findings_and_gates(summaries, corpus_document_count=1000)
