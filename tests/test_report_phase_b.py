from __future__ import annotations

import csv
import json

from bench.report.generate_report import write_phase_b_reports


def _summary(retriever: str, recall5: float, *, display_name: str | None = None, table: str | None = None) -> dict[str, object]:
    row = {
        "retriever": retriever,
        "display_name": display_name or retriever,
        "recall@1": 0.1,
        "recall@5": recall5,
        "recall@10": min(1.0, recall5 + 0.05),
        "mrr@10": 0.2,
        "ndcg@10": 0.25,
        "p50": 3.0,
        "p95": 9.0,
        "throughput_qps": 20.0,
        "memory_mb": 64.0,
        "index_memory_mb": 32.0,
        "search_memory_mb": 64.0,
        "index_build_seconds": 0.1,
    }
    if table is not None:
        row["table"] = table
    return row


def test_phase_b_report_writes_hero_and_extended_tables_json_csv_and_findings(tmp_path) -> None:
    payload = {
        "summary": {
            "source": "coir",
            "language": "python",
            "query_count": 100,
            "corpus_document_count": 100,
            "contamination_rejected_count": 0,
            "findings": ["CodeBERT null baseline stayed near zero as expected."],
        },
        "retriever_summaries": [
            _summary("ripgrep", 0.33, display_name="ripgrep"),
            _summary("bm25", 0.50, display_name="BM25"),
            _summary("codebert", 0.01, display_name="CodeBERT"),
            _summary("unixcoder", 0.65, display_name="UniXcoder"),
            _summary("lateon-code-edge", 0.75, display_name="LateOn-Code-edge"),
            _summary("lateon-code", 0.85, display_name="LateOn-Code"),
            _summary("sieve", 0.02, display_name="SIEVE"),
            _summary("bge-small", 0.72, display_name="bge-small-en-v1.5"),
        ],
        "rows": [
            {"retriever": "ripgrep", "query_id": "q1", "recall@5": 1.0, "mrr@10": 1.0, "ndcg@10": 1.0},
            {"retriever": "bm25", "query_id": "q1", "recall@5": 1.0, "mrr@10": 1.0, "ndcg@10": 1.0},
        ],
    }

    write_phase_b_reports(payload, output_dir=tmp_path)

    table = (tmp_path / "benchmark-table.md").read_text(encoding="utf-8")
    assert "## Hero Table: Default Dense Backend and Deployment Comparators" in table
    assert "| Retriever | Role | Params | Recall@1 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | p50 latency | p95 latency | Throughput | Memory |" in table
    assert "| BM25 | Classical IR baseline | — | 0.100 | 0.500 | 0.550 | 0.200 | 0.250 | 3.00 ms | 9.00 ms | 20.00 q/s | 64.00 MB |" in table
    assert "| LateOn-Code-edge | Size-matched retrieval-trained (primary) | 17M |" in table
    assert "| bge-small-en-v1.5 | Default dense backend | 33M | 0.100 | 0.720 | 0.770 | 0.200 | 0.250 | 3.00 ms | 9.00 ms | 20.00 q/s | 64.00 MB |" in table
    assert "## Extended Table: Reference Baselines" in table
    assert "| Retriever | Role | Params | Recall@1 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | p50 latency | p95 latency | Throughput | Memory |" in table
    assert "| CodeBERT (pretrained features only) | NULL BASELINE: off-the-shelf encoder without retrieval fine-tuning | 125M | 0.100 | 0.010 | 0.060 | 0.200 | 0.250 | 3.00 ms | 9.00 ms | 20.00 q/s | 64.00 MB |" in table
    assert "| LateOn-Code | SOTA reference at 10x scale; not a fair comparison with SIEVE's 4.2M param class | 149M |" in table
    assert "LateOn's multi-vector MaxSim brute-force scoring is inherently slower than single-vector cosine" in table
    assert "## Findings" in table
    assert "CodeBERT null baseline" in table

    raw = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert raw["summary"]["language"] == "python"

    with (tmp_path / "benchmark-full.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["retriever"] == "ripgrep"
    assert rows[1]["retriever"] == "bm25"


def test_phase_b5_methodology_note_present_in_report(tmp_path) -> None:
    payload = {
        "summary": {
            "source": "coir",
            "language": "python",
            "query_count": 14702,
            "corpus_document_count": 14918,
            "contamination_rejected_count": 0,
            "findings": [],
        },
        "benchmark": {"phase": "B.5", "eval_split": "python-eval-full"},
        "retriever_summaries": [
            _summary("ripgrep", 0.45, display_name="ripgrep"),
            _summary("bm25", 0.60, display_name="BM25"),
            _summary("codebert", 0.01, display_name="CodeBERT"),
            _summary("unixcoder", 0.70, display_name="UniXcoder"),
            _summary("lateon-code-edge", 0.80, display_name="LateOn-Code-edge"),
            _summary("lateon-code", 0.88, display_name="LateOn-Code"),
            _summary("sieve", 0.02, display_name="SIEVE"),
        ],
        "rows": [],
    }

    write_phase_b_reports(payload, output_dir=tmp_path)

    table = (tmp_path / "benchmark-table.md").read_text(encoding="utf-8")
    assert "Phase B v3 is the semantic-hard subset" in table
    assert "Phase B.5 is the full CodeSearchNet Python eval distribution" in table
    assert "mixed semantic-hard + literal-query workload" in table
    assert "uses raw `document.code` for all retrievers" in table
    assert "Real-world agent retrieval performance lives between these two numbers" in table


def test_typescript_methodology_note_present_in_report(tmp_path) -> None:
    payload = {
        "summary": {
            "source": "typescript-treesitter-dedupe",
            "language": "typescript",
            "query_count": 11579,
            "corpus_document_count": 11579,
            "contamination_rejected_count": 0,
            "findings": [],
        },
        "benchmark": {
            "phase": "B.5",
            "eval_split": "typescript-treesitter-dedupe-test",
            "dataset_id": "Shuu12121/typescript-treesitter-dedupe-filtered-datasetsV2",
            "dataset_language": "typescript",
            "dataset_card_license": "apache-2.0",
            "row_license_set": ("Apache-2.0", "MIT"),
            "unique_repo_count": 74,
            "typescript_family": "TypeScript",
            "eval_source_splits": ("test",),
            "full_example_count": 11579,
            "corpus_sample_size": 1000,
            "corpus_sampling_note": "Corpus was deterministically reduced to sampled positives plus random negatives for a bounded validation run.",
        },
        "retriever_summaries": [
            _summary("ripgrep", 0.30, display_name="ripgrep"),
            _summary("bm25", 0.50, display_name="BM25"),
            _summary("codebert", 0.01, display_name="CodeBERT"),
            _summary("unixcoder", 0.55, display_name="UniXcoder"),
            _summary("lateon-code-edge", 0.57, display_name="LateOn-Code-edge"),
            _summary("lateon-code", 0.60, display_name="LateOn-Code"),
            {
                **_summary("sieve", 0.00, display_name="SIEVE"),
                "embedding": {"interface": "pending-sieve-placeholder", "route_status": "sieve-cli-unavailable"},
            },
        ],
        "rows": [],
    }

    write_phase_b_reports(payload, output_dir=tmp_path)

    table = (tmp_path / "benchmark-table.md").read_text(encoding="utf-8")
    assert "# Phase B.5 TypeScript benchmark — typescript-treesitter-dedupe / typescript" in table
    assert "Shuu12121/typescript-treesitter-dedupe-filtered-datasetsV2" in table
    assert "canonical TypeScript `.ts`" in table
    assert "official CodeXGLUE/CoIR/CodeSearchNet TypeScript split was not available" in table
    assert "Dataset card license is `apache-2.0`" in table
    assert "row license set is `Apache-2.0, MIT`" in table
    assert "Corpus was deterministically reduced" in table
    assert "Full eval examples: 11579" in table
    assert "ArkTS" not in table
    assert "Phase 1 weights pending" in table
    assert "zero-recall pending placeholder" in table
    assert "SIEVE row calls the real Rust engine" not in table


def test_go_methodology_note_present_in_report(tmp_path) -> None:
    payload = {
        "summary": {
            "source": "coir-go",
            "language": "go",
            "query_count": 100,
            "corpus_document_count": 1000,
            "contamination_rejected_count": 0,
            "findings": [],
        },
        "benchmark": {
            "phase": "B.5",
            "eval_split": "go-eval-full",
            "dataset_id": "CoIR-Retrieval/CodeSearchNet",
            "dataset_language": "go",
            "dataset_revision": "25e0292562b7bee26dd9b2d83a03981795862c77",
            "full_example_count": 8122,
            "corpus_sample_size": 1000,
            "corpus_sampling_note": "Corpus was deterministically reduced to sampled positives plus random negatives for a bounded validation run.",
            "methodology": "Official CoIR/CodeSearchNet Go test qrels.",
        },
        "retriever_summaries": [
            _summary("ripgrep", 0.30, display_name="ripgrep"),
            _summary("bm25", 0.50, display_name="BM25"),
            _summary("codebert", 0.01, display_name="CodeBERT"),
            _summary("unixcoder", 0.55, display_name="UniXcoder"),
            _summary("lateon-code-edge", 0.57, display_name="LateOn-Code-edge"),
            _summary("lateon-code", 0.60, display_name="LateOn-Code"),
            {
                **_summary("sieve", 0.00, display_name="SIEVE"),
                "embedding": {"interface": "pending-sieve-placeholder", "route_status": "sieve-cli-unavailable"},
            },
        ],
        "rows": [],
    }

    write_phase_b_reports(payload, output_dir=tmp_path)

    table = (tmp_path / "benchmark-table.md").read_text(encoding="utf-8")
    assert "# Phase B.5 Go benchmark — coir-go / go" in table
    assert "CoIR-Retrieval/CodeSearchNet" in table
    assert "official CoIR/CodeSearchNet Go test qrels" in table
    assert "Full eval examples: 8122" in table
    assert "Phase 1 weights pending" in table
    assert "zero-recall pending placeholder" in table


def test_rust_methodology_note_present_in_report(tmp_path) -> None:
    payload = {
        "summary": {
            "source": "rust-treesitter-dedupe",
            "language": "rust",
            "query_count": 100,
            "corpus_document_count": 1000,
            "contamination_rejected_count": 0,
            "findings": [],
        },
        "benchmark": {
            "phase": "B.5",
            "eval_split": "rust-treesitter-dedupe-test",
            "dataset_id": "Shuu12121/rust-treesitter-dedupe-filtered-datasetsV2",
            "dataset_language": "rust",
            "dataset_card_license": "apache-2.0",
            "row_license_set": ("Apache-2.0", "MIT"),
            "unique_repo_count": 42,
            "eval_source_splits": ("test",),
            "full_example_count": 8868,
            "corpus_sample_size": 1000,
            "corpus_sampling_note": "Corpus was deterministically reduced to sampled positives plus random negatives for a bounded validation run.",
            "methodology": "No official CodeSearchNet/CoIR/CornStack Rust retrieval qrels were identifiable; this route uses pinned Rust docstring/code pairs.",
        },
        "retriever_summaries": [
            _summary("ripgrep", 0.30, display_name="ripgrep"),
            _summary("bm25", 0.50, display_name="BM25"),
            _summary("codebert", 0.01, display_name="CodeBERT"),
            _summary("unixcoder", 0.55, display_name="UniXcoder"),
            _summary("lateon-code-edge", 0.57, display_name="LateOn-Code-edge"),
            _summary("lateon-code", 0.60, display_name="LateOn-Code"),
            {
                **_summary("sieve", 0.00, display_name="SIEVE"),
                "embedding": {"interface": "pending-sieve-placeholder", "route_status": "sieve-cli-unavailable"},
            },
        ],
        "rows": [],
    }

    write_phase_b_reports(payload, output_dir=tmp_path)

    table = (tmp_path / "benchmark-table.md").read_text(encoding="utf-8")
    assert "# Phase B.5 Rust benchmark — rust-treesitter-dedupe / rust" in table
    assert "Shuu12121/rust-treesitter-dedupe-filtered-datasetsV2" in table
    assert "No official CodeSearchNet/CoIR/CornStack Rust retrieval qrels" in table
    assert "pinned Rust `.rs` docstring/code pairs" in table
    assert "Dataset card license is `apache-2.0`" in table
    assert "Full eval examples: 8868" in table
    assert "Phase 1 weights pending" in table
    assert "zero-recall pending placeholder" in table


def test_diagnostic_warning_fires_on_near_identical_memory_values(tmp_path) -> None:
    payload = {
        "summary": {
            "source": "coir",
            "language": "python",
            "query_count": 2,
            "corpus_document_count": 2,
            "contamination_rejected_count": 0,
            "findings": [],
        },
        "retriever_summaries": [
            _summary("ripgrep", 0.50, display_name="ripgrep"),
            _summary("bm25", 0.75, display_name="BM25"),
        ],
        "rows": [],
    }

    write_phase_b_reports(payload, output_dir=tmp_path)

    table = (tmp_path / "benchmark-table.md").read_text(encoding="utf-8")
    assert "possible memory measurement bug: retriever ripgrep=64.00 MB and BM25=64.00 MB report nearly-identical memory" in table


def test_diagnostic_warning_fires_on_all_zero_cpu_subprocess_delta_memory(tmp_path) -> None:
    def zero_cpu_row(retriever: str, display_name: str) -> dict[str, object]:
        row = _summary(retriever, 0.50, display_name=display_name)
        row["memory_mb"] = 0.0
        row["memory_measurement"] = {
            "process": {"mode": "subprocess", "pid": 123},
            "total": {"backend": "cpu-rss-delta", "baseline_mb": 100.0, "peak_mb": 100.0, "delta_mb": 0.0},
        }
        return row

    payload = {
        "summary": {
            "source": "coir",
            "language": "python",
            "query_count": 2,
            "corpus_document_count": 2,
            "contamination_rejected_count": 0,
            "findings": [],
        },
        "retriever_summaries": [
            zero_cpu_row("ripgrep", "ripgrep"),
            zero_cpu_row("bm25", "BM25"),
        ],
        "rows": [],
    }

    write_phase_b_reports(payload, output_dir=tmp_path)

    table = (tmp_path / "benchmark-table.md").read_text(encoding="utf-8")
    assert "CPU subprocess retrievers ripgrep, BM25 report 0.00 MB delta RSS" in table
