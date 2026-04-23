from __future__ import annotations

import math
from typing import Iterable, Mapping


def compute_query_metrics(
    *,
    ground_truth_document_id: str,
    results: list,
    ks: tuple[int, ...] = (1, 5, 10),
    mrr_k: int = 10,
    ndcg_k: int = 10,
) -> dict[str, float]:
    rank: int | None = None
    for index, result in enumerate(results, start=1):
        if result.document_id == ground_truth_document_id:
            rank = index
            break

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = 1.0 if rank is not None and rank <= k else 0.0

    metrics[f"mrr@{mrr_k}"] = 0.0 if rank is None or rank > mrr_k else 1.0 / float(rank)
    if rank is None or rank > ndcg_k:
        metrics[f"ndcg@{ndcg_k}"] = 0.0
    else:
        metrics[f"ndcg@{ndcg_k}"] = 1.0 / math.log2(rank + 1)
    return metrics


def aggregate_retrieval_metrics(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("Cannot aggregate zero metric rows")

    numeric_keys = sorted({key for row in materialized for key in row.keys()})
    summary: dict[str, float] = {"query_count": float(len(materialized))}
    for key in numeric_keys:
        summary[key] = sum(float(row[key]) for row in materialized) / float(len(materialized))
    summary["query_count"] = int(summary["query_count"])
    return summary
