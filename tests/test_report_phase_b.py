from __future__ import annotations

import csv
import json

from bench.report.generate_report import write_phase_b_reports


def test_phase_b_report_writes_hero_table_json_csv_and_findings(tmp_path) -> None:
    payload = {
        "summary": {
            "source": "coir",
            "language": "python",
            "query_count": 100,
            "corpus_document_count": 100,
            "contamination_rejected_count": 0,
            "findings": ["CodeBERT Recall@5 did not exceed BM25 on normalized code."],
        },
        "retriever_summaries": [
            {
                "retriever": "ripgrep",
                "display_name": "ripgrep",
                "recall@1": 0.1,
                "recall@5": 0.33,
                "recall@10": 0.4,
                "mrr@10": 0.2,
                "ndcg@10": 0.25,
                "p50": 3.0,
                "p95": 9.0,
                "throughput_qps": 20.0,
                "index_build_seconds": 0.1,
            },
            {
                "retriever": "bm25",
                "display_name": "BM25",
                "recall@1": 0.2,
                "recall@5": 0.5,
                "recall@10": 0.55,
                "mrr@10": 0.3,
                "ndcg@10": 0.35,
                "p50": 1.0,
                "p95": 2.0,
                "throughput_qps": 50.0,
                "index_build_seconds": 0.2,
            },
        ],
        "rows": [
            {"retriever": "ripgrep", "query_id": "q1", "recall@5": 1.0, "mrr@10": 1.0, "ndcg@10": 1.0},
            {"retriever": "bm25", "query_id": "q1", "recall@5": 1.0, "mrr@10": 1.0, "ndcg@10": 1.0},
        ],
    }

    write_phase_b_reports(payload, output_dir=tmp_path)

    table = (tmp_path / "benchmark-table.md").read_text(encoding="utf-8")
    assert "| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | p50 latency | p95 latency | Throughput |" in table
    assert "| BM25 | 0.200 | 0.500 | 0.550 | 0.300 | 0.350 | 1.00 ms | 2.00 ms | 50.00 q/s |" in table
    assert "## Findings" in table
    assert "CodeBERT Recall@5" in table

    raw = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert raw["summary"]["language"] == "python"

    with (tmp_path / "benchmark-full.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["retriever"] == "ripgrep"
    assert rows[1]["retriever"] == "bm25"
