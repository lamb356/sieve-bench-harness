from bench.metrics.retrieval import aggregate_retrieval_metrics, compute_query_metrics
from bench.retrievers.base import SearchResult


def test_compute_query_metrics_known_answer_fixture() -> None:
    results = [
        SearchResult(document_id="doc-9", path="doc-9.py", score=9.0, code="x = 9"),
        SearchResult(document_id="doc-2", path="doc-2.py", score=8.0, code="x = 2"),
        SearchResult(document_id="doc-1", path="doc-1.py", score=7.0, code="x = 1"),
    ]

    metrics = compute_query_metrics(ground_truth_document_id="doc-1", results=results, ks=(1, 5, 10))

    assert metrics["recall@1"] == 0.0
    assert metrics["recall@5"] == 1.0
    assert metrics["recall@10"] == 1.0
    assert round(metrics["mrr@10"], 6) == round(1.0 / 3.0, 6)
    assert round(metrics["ndcg@10"], 6) == round(1.0 / 2.0, 6)


def test_aggregate_retrieval_metrics_averages_rows() -> None:
    rows = [
        {"recall@1": 1.0, "recall@5": 1.0, "mrr@10": 1.0, "ndcg@10": 1.0},
        {"recall@1": 0.0, "recall@5": 1.0, "mrr@10": 0.5, "ndcg@10": 0.6309297535714575},
    ]

    summary = aggregate_retrieval_metrics(rows)

    assert summary["query_count"] == 2
    assert summary["recall@1"] == 0.5
    assert summary["recall@5"] == 1.0
    assert round(summary["mrr@10"], 6) == 0.75
