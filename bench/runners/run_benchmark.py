from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import typer

from bench.constants import (
    BLOOM_CANARY_CODE,
    COIR_LANGUAGE,
    CORNSTACK_BLOOM_PATH,
    MRR_K,
    NDCG_K,
    PHASE_A_RESULTS_DIR,
    PHASE_B_RESULTS_DIR,
    PHASE_B_RIPGREP_INDEX_DIR,
    QUICKCHECK_OUTPUT_JSON,
    QUICKCHECK_OUTPUT_MD,
    QUICKCHECK_SAMPLE_SIZE,
    QUICKCHECK_TOP_K,
    RECALL_KS,
    RIPGREP_INDEX_DIR,
)
from bench.contamination.bloom import BloomFilter, assert_canary_membership, normalized_code_hash
from bench.loaders.base import CodeDocument, EvalExample, LoadedBenchmark
from bench.loaders.coir import CoIRPythonLoader
from bench.metrics.performance import summarize_performance
from bench.metrics.retrieval import aggregate_retrieval_metrics, compute_query_metrics
from bench.report.generate_report import write_hero_table, write_phase_b_reports, write_results_json
from bench.retrievers.bm25 import BM25Retriever
from bench.retrievers.codebert import CodeBERTRetriever
from bench.retrievers.ripgrep import RipgrepRetriever
from bench.retrievers.sieve import SieveStubRetriever
from bench.retrievers.unixcoder import UniXcoderRetriever

app = typer.Typer(help="Run the SIEVE public benchmark harness.")


def _require_bloom_filter(path: Path) -> BloomFilter:
    if not path.is_file():
        raise FileNotFoundError(
            f"Contamination checks are mandatory; missing bloom filter at {path}. Build or provide the filter before benchmarking."
        )
    bloom = BloomFilter.load(path)
    assert_canary_membership(bloom, language=COIR_LANGUAGE, canary_code=BLOOM_CANARY_CODE)
    return bloom


def _accepted_examples(
    *, loaded: LoadedBenchmark, bloom: BloomFilter
) -> tuple[list[tuple[EvalExample, str]], list[tuple[EvalExample, str]]]:
    accepted_examples = []
    rejected_examples = []
    for example in loaded.examples:
        ground_truth_hash = normalized_code_hash(example.ground_truth_code, language=example.language)
        if ground_truth_hash in bloom:
            rejected_examples.append((example, ground_truth_hash))
            continue
        accepted_examples.append((example, ground_truth_hash))
    if not accepted_examples:
        raise RuntimeError("All eval examples were rejected by the contamination filter; refusing to benchmark")
    return accepted_examples, rejected_examples


def run_phase_a_quickcheck(*, bloom_path: Path, sample_size: int, top_k: int, output_dir: Path) -> dict[str, Any]:
    bloom = _require_bloom_filter(bloom_path)
    loader = CoIRPythonLoader()
    loaded = loader.load(sample_size=sample_size)
    accepted_examples, rejected_examples = _accepted_examples(loaded=loaded, bloom=bloom)

    retriever = RipgrepRetriever(index_root=RIPGREP_INDEX_DIR)
    retriever.index(loaded.corpus)

    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    for example, ground_truth_hash in accepted_examples:
        results = retriever.search(example.query, k=max(top_k, max(RECALL_KS)))
        metrics = compute_query_metrics(
            ground_truth_document_id=str(example.metadata["ground_truth_document_id"]),
            results=results,
            ks=RECALL_KS,
        )
        metric_rows.append(metrics)
        rows.append(
            {
                "query_id": str(example.metadata["query_id"]),
                "source": example.source,
                "language": example.language,
                "retriever": retriever.name,
                "query": example.query,
                "ground_truth_path": example.ground_truth_path,
                "ground_truth_hash": ground_truth_hash,
                "ground_truth_document_id": str(example.metadata["ground_truth_document_id"]),
                "top_k_result_hashes": [normalized_code_hash(result.code, language=example.language) for result in results[:top_k]],
                "top_k_result_document_ids": [result.document_id for result in results[:top_k]],
                "top_k_result_paths": [result.path for result in results[:top_k]],
                "retrieved_code_samples": [result.code for result in results[:top_k]],
                **metrics,
            }
        )

    aggregate = aggregate_retrieval_metrics(metric_rows)
    contamination_rejection_rate = len(rejected_examples) / float(len(loaded.examples))
    summary = {
        "source": loaded.source,
        "language": loaded.language,
        "retriever": retriever.name,
        "query_count": len(rows),
        "corpus_document_count": len(loaded.corpus),
        "contamination_rejected_count": len(rejected_examples),
        "contamination_rejection_rate": contamination_rejection_rate,
        "contamination_flag": contamination_rejection_rate > 0.05,
        "contamination_recommendation": "exclude-source-for-language" if contamination_rejection_rate > 0.05 else "retain-source-for-language",
        **aggregate,
        **retriever.latency_ms(),
    }
    payload = {
        "summary": summary,
        "rows": rows,
        "rejected": [
            {
                "query_id": str(example.metadata["query_id"]),
                "ground_truth_document_id": str(example.metadata["ground_truth_document_id"]),
                "ground_truth_hash": ground_truth_hash,
            }
            for example, ground_truth_hash in rejected_examples
        ],
        "benchmark": {
            "phase": "A",
            "sample_size": sample_size,
            "top_k": top_k,
            "bloom_path": str(bloom_path),
            "dataset_revision": loaded.revision,
            "corpus_id": loaded.corpus_id,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_results_json(payload, output_path=output_dir / QUICKCHECK_OUTPUT_JSON.name)
    write_hero_table(payload, template_root=Path(__file__).resolve().parents[1] / "report" / "templates", output_path=output_dir / QUICKCHECK_OUTPUT_MD.name)
    return payload


def _phase_b_retrievers() -> list[Any]:
    return [
        RipgrepRetriever(index_root=PHASE_B_RIPGREP_INDEX_DIR),
        BM25Retriever(),
        CodeBERTRetriever(),
        UniXcoderRetriever(),
        SieveStubRetriever(),
    ]


def _retriever_display_name(retriever: Any) -> str:
    return str(getattr(retriever, "display_name", getattr(retriever, "name", retriever.__class__.__name__)))


def _run_retriever(
    *,
    retriever: Any,
    corpus: tuple[CodeDocument, ...],
    accepted_examples: list[tuple[EvalExample, str]],
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    index_started = time.perf_counter()
    retriever.index(corpus)
    index_build_seconds = time.perf_counter() - index_started

    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    search_latencies_ms: list[float] = []
    total_search_seconds = 0.0
    search_k = max(top_k, max(RECALL_KS), MRR_K, NDCG_K)
    for example, ground_truth_hash in accepted_examples:
        search_started = time.perf_counter()
        results = retriever.search(example.query, k=search_k)
        elapsed_seconds = time.perf_counter() - search_started
        total_search_seconds += elapsed_seconds
        search_latencies_ms.append(elapsed_seconds * 1000.0)
        metrics = compute_query_metrics(
            ground_truth_document_id=str(example.metadata["ground_truth_document_id"]),
            results=results,
            ks=RECALL_KS,
            mrr_k=MRR_K,
            ndcg_k=NDCG_K,
        )
        metric_rows.append(metrics)
        rows.append(
            {
                "query_id": str(example.metadata["query_id"]),
                "source": example.source,
                "language": example.language,
                "retriever": retriever.name,
                "display_name": _retriever_display_name(retriever),
                "query": example.query,
                "ground_truth_path": example.ground_truth_path,
                "ground_truth_hash": ground_truth_hash,
                "ground_truth_document_id": str(example.metadata["ground_truth_document_id"]),
                "top_k_result_hashes": [normalized_code_hash(result.code, language=example.language) for result in results[:top_k]],
                "top_k_result_document_ids": [result.document_id for result in results[:top_k]],
                "top_k_result_paths": [result.path for result in results[:top_k]],
                "top_k_result_scores": [result.score for result in results[:top_k]],
                "retrieved_code_samples": [result.code for result in results[:top_k]],
                **metrics,
            }
        )

    aggregate = aggregate_retrieval_metrics(metric_rows)
    performance = summarize_performance(
        latencies_ms=search_latencies_ms,
        query_count=len(accepted_examples),
        total_search_seconds=total_search_seconds,
        index_build_seconds=index_build_seconds,
    )
    summary = {
        "retriever": retriever.name,
        "display_name": _retriever_display_name(retriever),
        "query_count": len(accepted_examples),
        **aggregate,
        **performance,
    }
    if hasattr(retriever, "embedding_metadata"):
        summary["embedding"] = retriever.embedding_metadata()
    return summary, rows


def _phase_b_findings_and_gates(retriever_summaries: list[dict[str, Any]], *, corpus_document_count: int) -> tuple[list[str], dict[str, Any]]:
    by_name = {summary["retriever"]: summary for summary in retriever_summaries}
    findings: list[str] = []
    gates: dict[str, Any] = {}

    ripgrep_recall = float(by_name["ripgrep"]["recall@5"])
    bm25_recall = float(by_name["bm25"]["recall@5"])
    gates["bm25_beats_ripgrep_recall@5"] = {"passed": bm25_recall > ripgrep_recall, "ripgrep": ripgrep_recall, "bm25": bm25_recall}
    if bm25_recall <= ripgrep_recall:
        raise RuntimeError(
            f"BM25 sanity gate failed: BM25 Recall@5={bm25_recall:.3f} did not beat ripgrep Recall@5={ripgrep_recall:.3f}"
        )

    for retriever_name, display_name in [("codebert", "CodeBERT"), ("unixcoder", "UniXcoder")]:
        recall = float(by_name[retriever_name]["recall@5"])
        passed = recall > bm25_recall
        gates[f"{retriever_name}_beats_bm25_recall@5"] = {
            "passed": passed,
            "mode": "finding-not-blocking",
            retriever_name: recall,
            "bm25": bm25_recall,
        }
        if not passed:
            findings.append(
                f"{display_name} Recall@5 ({recall:.3f}) did not exceed BM25 Recall@5 ({bm25_recall:.3f}) on normalized code; recorded as an empirical finding per Phase B approval."
            )

    sieve = by_name["sieve-stub"]
    expected_recall10 = 10.0 / float(corpus_document_count)
    observed_recall10 = float(sieve["recall@10"])
    max_allowed = max(0.05, expected_recall10 * 50.0)
    gates["sieve_stub_random_baseline_recall@10"] = {
        "passed": observed_recall10 <= max_allowed,
        "observed": observed_recall10,
        "expected": expected_recall10,
        "max_allowed": max_allowed,
    }
    if observed_recall10 > max_allowed:
        raise RuntimeError(
            f"SIEVE stub sanity gate failed: Recall@10={observed_recall10:.3f} is too high for random baseline expected≈{expected_recall10:.6f}"
        )
    return findings, gates


def run_phase_b_python_full(*, bloom_path: Path, sample_size: int, top_k: int, output_dir: Path) -> dict[str, Any]:
    bloom = _require_bloom_filter(bloom_path)
    loader = CoIRPythonLoader()
    loaded = loader.load(sample_size=sample_size)
    accepted_examples, rejected_examples = _accepted_examples(loaded=loaded, bloom=bloom)

    retriever_summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for retriever in _phase_b_retrievers():
        summary, retriever_rows = _run_retriever(
            retriever=retriever,
            corpus=loaded.corpus,
            accepted_examples=accepted_examples,
            top_k=top_k,
        )
        retriever_summaries.append(summary)
        rows.extend(retriever_rows)

    contamination_rejection_rate = len(rejected_examples) / float(len(loaded.examples))
    findings, gates = _phase_b_findings_and_gates(retriever_summaries, corpus_document_count=len(loaded.corpus))
    payload = {
        "summary": {
            "source": loaded.source,
            "language": loaded.language,
            "query_count": len(accepted_examples),
            "corpus_document_count": len(loaded.corpus),
            "contamination_rejected_count": len(rejected_examples),
            "contamination_rejection_rate": contamination_rejection_rate,
            "contamination_flag": contamination_rejection_rate > 0.05,
            "contamination_recommendation": "exclude-source-for-language" if contamination_rejection_rate > 0.05 else "retain-source-for-language",
            "findings": findings,
            "sanity_gates": gates,
        },
        "retriever_summaries": retriever_summaries,
        "rows": rows,
        "rejected": [
            {
                "query_id": str(example.metadata["query_id"]),
                "ground_truth_document_id": str(example.metadata["ground_truth_document_id"]),
                "ground_truth_hash": ground_truth_hash,
            }
            for example, ground_truth_hash in rejected_examples
        ],
        "benchmark": {
            "phase": "B",
            "sample_size": sample_size,
            "top_k": top_k,
            "bloom_path": str(bloom_path),
            "dataset_revision": loaded.revision,
            "corpus_id": loaded.corpus_id,
            "normalized_surface": "document.index_text",
            "phase_b_5_scope_note": "After Phase B is verified and pushed, run the same 5-retriever benchmark on raw code into bench-results/phase-b.5-python-raw/.",
        },
    }
    write_phase_b_reports(payload, output_dir=output_dir)
    return payload


@app.command("phase-a-quickcheck")
def phase_a_quickcheck(
    bloom_path: Path = typer.Option(CORNSTACK_BLOOM_PATH, exists=False, dir_okay=False),
    sample_size: int = typer.Option(QUICKCHECK_SAMPLE_SIZE, min=1),
    top_k: int = typer.Option(QUICKCHECK_TOP_K, min=1),
    output_dir: Path = typer.Option(PHASE_A_RESULTS_DIR, file_okay=False),
) -> None:
    payload = run_phase_a_quickcheck(
        bloom_path=bloom_path,
        sample_size=sample_size,
        top_k=top_k,
        output_dir=output_dir,
    )
    typer.echo(
        f"Phase A quickcheck complete: source={payload['summary']['source']} language={payload['summary']['language']} recall@5={payload['summary']['recall@5']:.3f}"
    )


@app.command("phase-b-python-full")
def phase_b_python_full(
    bloom_path: Path = typer.Option(CORNSTACK_BLOOM_PATH, exists=False, dir_okay=False),
    sample_size: int = typer.Option(QUICKCHECK_SAMPLE_SIZE, min=1),
    top_k: int = typer.Option(max(RECALL_KS), min=1),
    output_dir: Path = typer.Option(PHASE_B_RESULTS_DIR, file_okay=False),
) -> None:
    payload = run_phase_b_python_full(
        bloom_path=bloom_path,
        sample_size=sample_size,
        top_k=top_k,
        output_dir=output_dir,
    )
    by_name = {summary["retriever"]: summary for summary in payload["retriever_summaries"]}
    typer.echo(
        "Phase B Python full benchmark complete: "
        f"BM25 recall@5={by_name['bm25']['recall@5']:.3f} "
        f"CodeBERT recall@5={by_name['codebert']['recall@5']:.3f} "
        f"UniXcoder recall@5={by_name['unixcoder']['recall@5']:.3f}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
