from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from bench.constants import (
    BLOOM_CANARY_CODE,
    COIR_LANGUAGE,
    CORNSTACK_BLOOM_PATH,
    PHASE_A_RESULTS_DIR,
    QUICKCHECK_OUTPUT_JSON,
    QUICKCHECK_OUTPUT_MD,
    QUICKCHECK_SAMPLE_SIZE,
    QUICKCHECK_TOP_K,
    RECALL_KS,
    RIPGREP_INDEX_DIR,
)
from bench.contamination.bloom import BloomFilter, assert_canary_membership, normalized_code_hash
from bench.loaders.coir import CoIRPythonLoader
from bench.metrics.retrieval import aggregate_retrieval_metrics, compute_query_metrics
from bench.report.generate_report import write_hero_table, write_results_json
from bench.retrievers.ripgrep import RipgrepRetriever

app = typer.Typer(help="Run the SIEVE public benchmark harness.")


def _require_bloom_filter(path: Path) -> BloomFilter:
    if not path.is_file():
        raise FileNotFoundError(
            f"Contamination checks are mandatory; missing bloom filter at {path}. Build or provide the filter before benchmarking."
        )
    bloom = BloomFilter.load(path)
    assert_canary_membership(bloom, language=COIR_LANGUAGE, canary_code=BLOOM_CANARY_CODE)
    return bloom


def run_phase_a_quickcheck(*, bloom_path: Path, sample_size: int, top_k: int, output_dir: Path) -> dict[str, Any]:
    bloom = _require_bloom_filter(bloom_path)
    loader = CoIRPythonLoader()
    loaded = loader.load(sample_size=sample_size)

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
