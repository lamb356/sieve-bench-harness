from __future__ import annotations

import gc
import multiprocessing as mp
import os
import queue
import shutil
import signal
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
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
    PHASE_B_V2_RESULTS_DIR,
    PHASE_B_V3_RESULTS_DIR,
    PHASE_B5_CPU_TIMEOUT_SECONDS,
    PHASE_B5_RESULTS_DIR,
    PHASE_B5_TYPESCRIPT_RESULTS_DIR,
    PHASE_B_RIPGREP_INDEX_DIR,
    PHASE_B_TYPESCRIPT_RESULTS_DIR,
    PHASE_B_TYPESCRIPT_RIPGREP_INDEX_DIR,
    PHASE_B5_RIPGREP_INDEX_DIR,
    PHASE_B5_TYPESCRIPT_RIPGREP_INDEX_DIR,
    PYTHON_EVAL_FULL,
    QUICKCHECK_OUTPUT_JSON,
    QUICKCHECK_OUTPUT_MD,
    QUICKCHECK_SAMPLE_SIZE,
    QUICKCHECK_TOP_K,
    RECALL_KS,
    RIPGREP_INDEX_DIR,
    TYPESCRIPT_DATASET_ID,
    TYPESCRIPT_EVAL_FULL,
)
from bench.contamination.bloom import BloomFilter, assert_canary_membership, normalized_code_hash
from bench.loaders.base import CodeDocument, EvalExample, LoadedBenchmark
from bench.loaders.coir import CoIRPythonLoader
from bench.loaders.typescript import TypeScriptEvalLoader
from bench.metrics.performance import measure_cpu_retriever_delta_rss, measure_torch_cuda_peak_allocated, summarize_performance
from bench.metrics.retrieval import aggregate_retrieval_metrics, compute_query_metrics
from bench.report.generate_report import write_hero_table, write_phase_b_reports, write_results_json
from bench.retrievers import RETRIEVER_REPORT_METADATA
from bench.retrievers.bm25 import BM25Retriever
from bench.retrievers.codebert import CodeBERTRetriever
from bench.retrievers.lateon_code import LateOnCodeRetriever
from bench.retrievers.lateon_code_edge import LateOnCodeEdgeRetriever
from bench.retrievers.ripgrep import RipgrepRetriever
from bench.retrievers.sieve import SieveRetriever
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


@dataclass(frozen=True)
class PhaseBRetrieverFactory:
    retriever_name: str
    factory: Callable[[], Any]
    run_in_subprocess: bool = False

    def __call__(self) -> Any:
        return self.factory()


def _build_phase_b_ripgrep() -> RipgrepRetriever:
    return RipgrepRetriever(index_root=PHASE_B_RIPGREP_INDEX_DIR)


def _build_phase_b5_ripgrep() -> RipgrepRetriever:
    return RipgrepRetriever(index_root=PHASE_B5_RIPGREP_INDEX_DIR)


def _build_phase_b_typescript_ripgrep() -> RipgrepRetriever:
    return RipgrepRetriever(index_root=PHASE_B_TYPESCRIPT_RIPGREP_INDEX_DIR)


def _build_phase_b5_typescript_ripgrep() -> RipgrepRetriever:
    return RipgrepRetriever(index_root=PHASE_B5_TYPESCRIPT_RIPGREP_INDEX_DIR)


class _PendingSieveRetriever:
    name = "sieve"
    display_name = "SIEVE (Phase 1 weights pending)"

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        self._query_count = 0

    def index(self, corpus: tuple[CodeDocument, ...]) -> None:
        if not corpus:
            raise ValueError("Pending SIEVE retriever requires a non-empty corpus")

    def search(self, query: str, k: int) -> list[SearchResult]:
        del query, k
        self._query_count += 1
        return []

    def latency_ms(self) -> dict[str, float]:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    def embedding_metadata(self) -> dict[str, Any]:
        return {
            "interface": "pending-sieve-placeholder",
            "weights_status": "Phase 1 weights pending",
            "route_status": "sieve-cli-unavailable",
            "reason": self.reason,
            "query_count": self._query_count,
        }


def _sieve_cli_route_available() -> tuple[bool, str]:
    env_binary = os.environ.get("SIEVE_BINARY")
    if env_binary:
        if Path(env_binary).is_file():
            return True, f"SIEVE_BINARY={env_binary}"
        return False, f"SIEVE_BINARY does not exist: {env_binary}"
    which_binary = shutil.which("sieve")
    if which_binary is not None:
        return True, f"PATH sieve={which_binary}"
    repo = os.environ.get("SIEVE_REPO")
    if repo:
        if Path(repo, "Cargo.toml").is_file():
            return True, f"SIEVE_REPO={repo}"
        return False, f"SIEVE_REPO lacks Cargo.toml: {repo}"
    return False, "could not locate sieve binary on PATH; set SIEVE_BINARY or SIEVE_REPO for real SIEVE scoring"


def _build_typescript_sieve() -> Any:
    available, reason = _sieve_cli_route_available()
    if available:
        return SieveRetriever()
    return _PendingSieveRetriever(reason=reason)


def _phase_b_retriever_factories() -> list[PhaseBRetrieverFactory]:
    return [
        PhaseBRetrieverFactory("ripgrep", _build_phase_b_ripgrep, run_in_subprocess=True),
        PhaseBRetrieverFactory("bm25", BM25Retriever, run_in_subprocess=True),
        PhaseBRetrieverFactory("sieve", SieveRetriever, run_in_subprocess=True),
        PhaseBRetrieverFactory("codebert", CodeBERTRetriever),
        PhaseBRetrieverFactory("unixcoder", UniXcoderRetriever),
        PhaseBRetrieverFactory("lateon-code-edge", LateOnCodeEdgeRetriever),
        PhaseBRetrieverFactory("lateon-code", LateOnCodeRetriever),
    ]


def _phase_b5_retriever_factories() -> list[PhaseBRetrieverFactory]:
    return [
        PhaseBRetrieverFactory("ripgrep", _build_phase_b5_ripgrep, run_in_subprocess=True),
        PhaseBRetrieverFactory("bm25", BM25Retriever, run_in_subprocess=True),
        PhaseBRetrieverFactory("sieve", SieveRetriever, run_in_subprocess=True),
        PhaseBRetrieverFactory("codebert", CodeBERTRetriever),
        PhaseBRetrieverFactory("unixcoder", UniXcoderRetriever),
        PhaseBRetrieverFactory("lateon-code-edge", LateOnCodeEdgeRetriever),
        PhaseBRetrieverFactory("lateon-code", LateOnCodeRetriever),
    ]


def _phase_b_typescript_retriever_factories() -> list[PhaseBRetrieverFactory]:
    # TypeScript-specific retrieval config:
    # - ripgrep is text-agnostic and indexes the normalized TypeScript `index_text` surface.
    # - BM25 uses the same punctuation-aware tokenization over normalized code; commas/braces are delimiters.
    # - SIEVE currently routes through the real CLI with random/local ONNX exports, so report rows are labeled pending.
    # - CodeBERT remains a null baseline; UniXcoder and LateOn-Code variants use their language-agnostic code encoders.
    return [
        PhaseBRetrieverFactory("ripgrep", _build_phase_b_typescript_ripgrep, run_in_subprocess=True),
        PhaseBRetrieverFactory("bm25", BM25Retriever, run_in_subprocess=True),
        PhaseBRetrieverFactory("sieve", _build_typescript_sieve, run_in_subprocess=True),
        PhaseBRetrieverFactory("codebert", CodeBERTRetriever),
        PhaseBRetrieverFactory("unixcoder", UniXcoderRetriever),
        PhaseBRetrieverFactory("lateon-code-edge", LateOnCodeEdgeRetriever),
        PhaseBRetrieverFactory("lateon-code", LateOnCodeRetriever),
    ]


def _phase_b5_typescript_retriever_factories() -> list[PhaseBRetrieverFactory]:
    return [
        PhaseBRetrieverFactory("ripgrep", _build_phase_b5_typescript_ripgrep, run_in_subprocess=True),
        PhaseBRetrieverFactory("bm25", BM25Retriever, run_in_subprocess=True),
        PhaseBRetrieverFactory("sieve", _build_typescript_sieve, run_in_subprocess=True),
        PhaseBRetrieverFactory("codebert", CodeBERTRetriever),
        PhaseBRetrieverFactory("unixcoder", UniXcoderRetriever),
        PhaseBRetrieverFactory("lateon-code-edge", LateOnCodeEdgeRetriever),
        PhaseBRetrieverFactory("lateon-code", LateOnCodeRetriever),
    ]


def _phase_b_retrievers() -> list[Any]:
    return [factory() for factory in _phase_b_retriever_factories()]


def _retriever_report_metadata(retriever_name: str) -> dict[str, Any]:
    metadata = RETRIEVER_REPORT_METADATA.get(retriever_name)
    if metadata is None:
        return {}
    return {
        "table": metadata.table,
        "role": metadata.role,
        "role_label": metadata.role_label,
        "params": metadata.params,
        "display_name": metadata.display_name,
        "order": metadata.order,
    }


def _retriever_display_name(retriever: Any) -> str:
    metadata = RETRIEVER_REPORT_METADATA.get(str(getattr(retriever, "name", "")))
    if metadata is not None:
        return metadata.display_name
    return str(getattr(retriever, "display_name", getattr(retriever, "name", retriever.__class__.__name__)))


def _retriever_uses_cuda(retriever: Any) -> tuple[bool, str | None]:
    if not hasattr(retriever, "embedding_metadata"):
        return False, None
    try:
        device = str(retriever.embedding_metadata().get("device", ""))
    except Exception:  # pragma: no cover - defensive metadata fallback
        return False, None
    return device.startswith("cuda"), device or None


def _run_retriever(
    *,
    retriever: Any,
    corpus: tuple[CodeDocument, ...],
    accepted_examples: list[tuple[EvalExample, str]],
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    search_latencies_ms: list[float] = []
    total_search_seconds = 0.0
    index_build_seconds = 0.0
    search_k = max(top_k, max(RECALL_KS), MRR_K, NDCG_K)

    def _index() -> None:
        nonlocal index_build_seconds
        index_started = time.perf_counter()
        retriever.index(corpus)
        index_build_seconds = time.perf_counter() - index_started

    def _search_all() -> None:
        nonlocal total_search_seconds
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

    uses_cuda, device = _retriever_uses_cuda(retriever)
    if uses_cuda:
        _, index_memory = measure_torch_cuda_peak_allocated(_index, device=device)
        _, search_memory = measure_torch_cuda_peak_allocated(_search_all, device=device)
        memory_mb = max(index_memory.peak_mb, search_memory.peak_mb)
        index_memory_mb = index_memory.peak_mb
        search_memory_mb = search_memory.peak_mb
        memory_measurement = {
            "index": index_memory.to_json(),
            "search": search_memory.to_json(),
        }
    else:
        _, memory_measurements = measure_cpu_retriever_delta_rss(_index, _search_all)
        index_memory = memory_measurements["index"]
        search_memory = memory_measurements["search"]
        total_memory = memory_measurements["total"]
        memory_mb = total_memory.delta_mb
        index_memory_mb = index_memory.delta_mb
        search_memory_mb = search_memory.delta_mb
        memory_measurement = {name: measurement.to_json() for name, measurement in memory_measurements.items()}

    aggregate = aggregate_retrieval_metrics(metric_rows)
    performance = summarize_performance(
        latencies_ms=search_latencies_ms,
        query_count=len(accepted_examples),
        total_search_seconds=total_search_seconds,
        index_build_seconds=index_build_seconds,
        memory_mb=memory_mb,
        index_memory_mb=index_memory_mb,
        search_memory_mb=search_memory_mb,
    )
    summary = {
        "retriever": retriever.name,
        "display_name": _retriever_display_name(retriever),
        "query_count": len(accepted_examples),
        **_retriever_report_metadata(retriever.name),
        **aggregate,
        **performance,
        "memory_measurement": memory_measurement,
    }
    if hasattr(retriever, "embedding_metadata"):
        summary["embedding"] = retriever.embedding_metadata()
    return summary, rows


def _run_cpu_retriever_child(
    result_queue: Any,
    retriever_factory: Callable[[], Any],
    corpus: tuple[CodeDocument, ...],
    accepted_examples: list[tuple[EvalExample, str]],
    top_k: int,
) -> None:
    try:
        if hasattr(os, "setsid"):
            try:
                os.setsid()
            except OSError:
                pass
        retriever = retriever_factory()
        summary, rows = _run_retriever(retriever=retriever, corpus=corpus, accepted_examples=accepted_examples, top_k=top_k)
        summary.setdefault("memory_measurement", {})["process"] = {"mode": "subprocess", "pid": os.getpid()}
        result_queue.put({"ok": True, "summary": summary, "rows": rows})
    except BaseException:  # pragma: no cover - parent re-raises the child traceback
        result_queue.put({"ok": False, "traceback": traceback.format_exc()})


def _multiprocessing_context() -> mp.context.BaseContext:
    if "spawn" in mp.get_all_start_methods():
        return mp.get_context("spawn")
    return mp.get_context()


def _terminate_cpu_retriever_process(process: mp.Process) -> None:
    if process.pid is not None and hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            process.terminate()
    else:
        process.terminate()
    process.join(timeout=5.0)
    if process.is_alive():
        if process.pid is not None and hasattr(os, "killpg"):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                process.kill()
        else:
            process.kill()
        process.join(timeout=5.0)


def _run_cpu_retriever_in_subprocess(
    *,
    retriever_factory: Callable[[], Any],
    corpus: tuple[CodeDocument, ...],
    accepted_examples: list[tuple[EvalExample, str]],
    top_k: int,
    timeout_seconds: float = 600.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    context = _multiprocessing_context()
    result_queue = context.Queue()
    process = context.Process(
        target=_run_cpu_retriever_child,
        args=(result_queue, retriever_factory, corpus, accepted_examples, top_k),
    )
    process.start()
    message: dict[str, Any] | None = None
    deadline = time.monotonic() + timeout_seconds
    try:
        while message is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                _terminate_cpu_retriever_process(process)
                raise TimeoutError(f"CPU retriever subprocess exceeded {timeout_seconds:.1f}s timeout")
            try:
                message = result_queue.get(timeout=min(1.0, remaining))
            except queue.Empty:
                if not process.is_alive():
                    process.join(timeout=0.0)
                    if process.exitcode not in (0, None):
                        raise RuntimeError(f"CPU retriever subprocess exited with code {process.exitcode} before returning a result")
                    raise RuntimeError("CPU retriever subprocess exited without returning a result")
    finally:
        result_queue.close()

    process.join(timeout=timeout_seconds)
    if process.is_alive():
        _terminate_cpu_retriever_process(process)
        raise TimeoutError(f"CPU retriever subprocess did not exit after returning a result within {timeout_seconds:.1f}s")
    if process.exitcode not in (0, None):
        raise RuntimeError(f"CPU retriever subprocess exited with code {process.exitcode}")
    if not message.get("ok"):
        raise RuntimeError("CPU retriever subprocess failed:\n" + str(message.get("traceback", "<missing traceback>")))
    return message["summary"], message["rows"]


def _phase_b_findings_and_gates(retriever_summaries: list[dict[str, Any]], *, corpus_document_count: int) -> tuple[list[str], dict[str, Any]]:
    del corpus_document_count  # Real SIEVE is no longer a random-baseline stub.
    by_name = {summary["retriever"]: summary for summary in retriever_summaries}
    findings: list[str] = []
    gates: dict[str, Any] = {}

    bm25_recall = float(by_name["bm25"]["recall@5"])
    unixcoder_recall = float(by_name["unixcoder"]["recall@5"])
    lateon_edge_recall = float(by_name["lateon-code-edge"]["recall@5"])
    lateon_recall = float(by_name["lateon-code"]["recall@5"])
    codebert_recall = float(by_name["codebert"]["recall@5"])

    gates["unixcoder_beats_bm25_recall@5"] = {"passed": unixcoder_recall > bm25_recall, "unixcoder": unixcoder_recall, "bm25": bm25_recall}
    if unixcoder_recall <= bm25_recall:
        raise RuntimeError(
            f"UniXcoder sanity gate failed: UniXcoder Recall@5={unixcoder_recall:.3f} did not beat BM25 Recall@5={bm25_recall:.3f}"
        )

    gates["lateon_code_edge_beats_unixcoder_recall@5"] = {
        "passed": lateon_edge_recall > unixcoder_recall,
        "lateon-code-edge": lateon_edge_recall,
        "unixcoder": unixcoder_recall,
    }
    if lateon_edge_recall <= unixcoder_recall:
        raise RuntimeError(
            f"LateOn-Code-edge sanity gate failed: LateOn-Code-edge Recall@5={lateon_edge_recall:.3f} did not beat UniXcoder Recall@5={unixcoder_recall:.3f}"
        )

    gates["lateon_code_beats_lateon_code_edge_recall@5"] = {
        "passed": lateon_recall > lateon_edge_recall,
        "lateon-code": lateon_recall,
        "lateon-code-edge": lateon_edge_recall,
    }
    if lateon_recall <= lateon_edge_recall:
        raise RuntimeError(
            f"LateOn-Code sanity gate failed: LateOn-Code Recall@5={lateon_recall:.3f} did not beat LateOn-Code-edge Recall@5={lateon_edge_recall:.3f}"
        )

    gates["codebert_null_baseline_recall@5_lt_0.05"] = {"passed": codebert_recall < 0.05, "codebert": codebert_recall, "max_allowed": 0.05}
    if codebert_recall >= 0.05:
        raise RuntimeError(
            f"CodeBERT null-baseline sanity gate failed: CodeBERT Recall@5={codebert_recall:.3f} is unexpectedly above the <0.05 null-baseline threshold"
        )
    findings.append(f"CodeBERT null baseline stayed near zero as expected (Recall@5={codebert_recall:.3f}).")

    findings.append(f"SIEVE real engine row included with Recall@5={float(by_name['sieve']['recall@5']):.3f}; quality is expected to move after Phase 1 weights replace random/local ONNX exports.")
    return findings, gates


def _phase_b5_findings_and_gates(retriever_summaries: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    by_name = {summary["retriever"]: summary for summary in retriever_summaries}
    findings: list[str] = []
    gates: dict[str, Any] = {}

    def _recall(name: str) -> float:
        return float(by_name[name]["recall@5"])

    def _record_comparison(key: str, left: str, right: str, *, relation: str) -> None:
        left_recall = _recall(left)
        right_recall = _recall(right)
        passed = left_recall > right_recall if relation == ">" else left_recall < right_recall
        gates[key] = {
            "passed": passed,
            left: left_recall,
            right: right_recall,
            "fatal": False,
            "scope": "observational-full-eval",
        }

    _record_comparison("unixcoder_beats_bm25_recall@5", "unixcoder", "bm25", relation=">")
    _record_comparison("lateon_code_edge_beats_unixcoder_recall@5", "lateon-code-edge", "unixcoder", relation=">")
    _record_comparison("lateon_code_beats_lateon_code_edge_recall@5", "lateon-code", "lateon-code-edge", relation=">")

    codebert_recall = _recall("codebert")
    gates["codebert_null_baseline_recall@5_lt_0.05"] = {
        "passed": codebert_recall < 0.05,
        "codebert": codebert_recall,
        "max_allowed": 0.05,
        "fatal": False,
        "scope": "observational-full-eval",
    }
    if codebert_recall < 0.05:
        findings.append(f"CodeBERT null baseline stayed near zero on full eval (Recall@5={codebert_recall:.3f}).")
    else:
        findings.append(f"CodeBERT null baseline exceeded the Phase B semantic-hard guardrail on full eval (Recall@5={codebert_recall:.3f}); recorded as observational, not fatal.")
    findings.append(
        f"SIEVE row included with Recall@5={_recall('sieve'):.3f}; B.5 records full-eval/raw-surface behavior without applying Phase B semantic-hard ordering gates."
    )
    return findings, gates


def _run_phase_b_factories(
    *,
    retriever_factories: list[PhaseBRetrieverFactory],
    loaded: LoadedBenchmark,
    accepted_examples: list[tuple[EvalExample, str]],
    top_k: int,
    cpu_timeout_seconds: float = 600.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retriever_summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for retriever_factory in retriever_factories:
        if retriever_factory.run_in_subprocess:
            summary, retriever_rows = _run_cpu_retriever_in_subprocess(
                retriever_factory=retriever_factory,
                corpus=loaded.corpus,
                accepted_examples=accepted_examples,
                top_k=top_k,
                timeout_seconds=cpu_timeout_seconds,
            )
        else:
            retriever = retriever_factory()
            summary, retriever_rows = _run_retriever(
                retriever=retriever,
                corpus=loaded.corpus,
                accepted_examples=accepted_examples,
                top_k=top_k,
            )
            del retriever
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:  # pragma: no cover - torch is a project dependency
                pass
        retriever_summaries.append(summary)
        rows.extend(retriever_rows)
    return retriever_summaries, rows


def _mark_typescript_sieve_pending(retriever_summaries: list[dict[str, Any]]) -> None:
    for summary in retriever_summaries:
        if summary.get("retriever") != "sieve":
            continue
        summary["display_name_override"] = "SIEVE (Phase 1 weights pending)"
        summary["role_label_override"] = "SIEVE Path 2 — Phase 1 weights pending"
        summary["weights_status"] = "Phase 1 weights pending"
        embedding = summary.get("embedding") if isinstance(summary.get("embedding"), dict) else {}
        if embedding.get("interface") == "pending-sieve-placeholder" or embedding.get("route_status") == "sieve-cli-unavailable":
            note = (
                "SIEVE CLI route was unavailable, so this row is a zero-recall pending placeholder; "
                "set SIEVE_BINARY or SIEVE_REPO and replace ONNX paths with Phase 1 weights before quality claims."
            )
        else:
            note = (
                "Current TypeScript run uses the existing SIEVE CLI path with random/local ONNX exports; "
                "replace ONNX paths with Phase 1 weights before quality claims."
            )
        summary.setdefault("notes", []).append(note)


def _typescript_findings_and_gates(retriever_summaries: list[dict[str, Any]], *, phase_label: str) -> tuple[list[str], dict[str, Any]]:
    findings, gates = _phase_b5_findings_and_gates(retriever_summaries)
    phase_scope = "observational-typescript"
    for gate in gates.values():
        if isinstance(gate, dict) and gate.get("scope") == "observational-full-eval":
            gate["scope"] = phase_scope
    findings = [
        finding.replace("on full eval", "on TypeScript eval")
        .replace(
            "B.5 records full-eval/raw-surface behavior without applying Phase B semantic-hard ordering gates",
            f"{phase_label} records canonical TypeScript behavior without applying Python semantic-hard ordering gates",
        )
        for finding in findings
    ]
    by_name = {summary["retriever"]: summary for summary in retriever_summaries}
    if "sieve" in by_name:
        findings.append(
            f"SIEVE TypeScript row is labeled Phase 1 weights pending (Recall@5={float(by_name['sieve']['recall@5']):.3f}); current numbers validate harness plumbing, not trained cross-language quality."
        )
    findings.append(
        f"{phase_label} uses canonical TypeScript `.ts` code/docstring pairs from `{TYPESCRIPT_DATASET_ID}` because official CodeXGLUE/CoIR/CodeSearchNet TypeScript retrieval qrels were not available."
    )
    return findings, gates


def _phase_b_payload(
    *,
    loaded: LoadedBenchmark,
    accepted_examples: list[tuple[EvalExample, str]],
    rejected_examples: list[tuple[EvalExample, str]],
    retriever_summaries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    findings: list[str],
    gates: dict[str, Any],
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    contamination_rejection_rate = len(rejected_examples) / float(len(loaded.examples))
    return {
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
        "benchmark": benchmark,
    }


def run_phase_b_python_full(*, bloom_path: Path, sample_size: int, top_k: int, output_dir: Path) -> dict[str, Any]:
    bloom = _require_bloom_filter(bloom_path)
    loader = CoIRPythonLoader()
    loaded = loader.load(sample_size=sample_size)
    accepted_examples, rejected_examples = _accepted_examples(loaded=loaded, bloom=bloom)

    retriever_summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for retriever_factory in _phase_b_retriever_factories():
        if retriever_factory.run_in_subprocess:
            summary, retriever_rows = _run_cpu_retriever_in_subprocess(
                retriever_factory=retriever_factory,
                corpus=loaded.corpus,
                accepted_examples=accepted_examples,
                top_k=top_k,
            )
        else:
            retriever = retriever_factory()
            summary, retriever_rows = _run_retriever(
                retriever=retriever,
                corpus=loaded.corpus,
                accepted_examples=accepted_examples,
                top_k=top_k,
            )
            del retriever
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:  # pragma: no cover - torch is a project dependency
                pass
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
            "phase": "B-v3",
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


def run_phase_b5_python_full(
    *,
    bloom_path: Path,
    sample_size: int | None,
    top_k: int,
    output_dir: Path,
    cpu_timeout_seconds: float = PHASE_B5_CPU_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    bloom = _require_bloom_filter(bloom_path)
    loader = CoIRPythonLoader()
    loaded = loader.load_full_eval(sample_size=sample_size)
    accepted_examples, rejected_examples = _accepted_examples(loaded=loaded, bloom=bloom)

    retriever_summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for retriever_factory in _phase_b5_retriever_factories():
        if retriever_factory.run_in_subprocess:
            summary, retriever_rows = _run_cpu_retriever_in_subprocess(
                retriever_factory=retriever_factory,
                corpus=loaded.corpus,
                accepted_examples=accepted_examples,
                top_k=top_k,
                timeout_seconds=cpu_timeout_seconds,
            )
        else:
            retriever = retriever_factory()
            summary, retriever_rows = _run_retriever(
                retriever=retriever,
                corpus=loaded.corpus,
                accepted_examples=accepted_examples,
                top_k=top_k,
            )
            del retriever
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:  # pragma: no cover - torch is a project dependency
                pass
        retriever_summaries.append(summary)
        rows.extend(retriever_rows)

    contamination_rejection_rate = len(rejected_examples) / float(len(loaded.examples))
    findings, gates = _phase_b5_findings_and_gates(retriever_summaries)
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
            "phase": "B.5",
            "sample_size": sample_size,
            "top_k": top_k,
            "bloom_path": str(bloom_path),
            "cpu_timeout_seconds": cpu_timeout_seconds,
            "dataset_revision": loaded.revision,
            "corpus_id": loaded.corpus_id,
            "eval_split": PYTHON_EVAL_FULL,
            "full_example_count": loaded.metadata.get("full_example_count"),
            "normalized_surface": "document.index_text",
        },
    }
    write_phase_b_reports(payload, output_dir=output_dir)
    return payload


def run_phase_b_typescript_full(
    *,
    bloom_path: Path,
    sample_size: int,
    top_k: int,
    output_dir: Path,
    corpus_sample_size: int | None = None,
) -> dict[str, Any]:
    bloom = _require_bloom_filter(bloom_path)
    loader = TypeScriptEvalLoader()
    loaded = loader.load_full_eval(sample_size=sample_size, corpus_sample_size=corpus_sample_size)
    accepted_examples, rejected_examples = _accepted_examples(loaded=loaded, bloom=bloom)

    retriever_summaries, rows = _run_phase_b_factories(
        retriever_factories=_phase_b_typescript_retriever_factories(),
        loaded=loaded,
        accepted_examples=accepted_examples,
        top_k=top_k,
    )
    _mark_typescript_sieve_pending(retriever_summaries)
    findings, gates = _typescript_findings_and_gates(retriever_summaries, phase_label="Phase B v3 TypeScript full eval")
    payload = _phase_b_payload(
        loaded=loaded,
        accepted_examples=accepted_examples,
        rejected_examples=rejected_examples,
        retriever_summaries=retriever_summaries,
        rows=rows,
        findings=findings,
        gates=gates,
        benchmark={
            "phase": "B-v3",
            "sample_size": sample_size,
            "top_k": top_k,
            "bloom_path": str(bloom_path),
            "dataset_id": loaded.metadata.get("dataset_id"),
            "dataset_revision": loaded.revision,
            "dataset_language": loaded.metadata.get("dataset_language"),
            "dataset_card_license": loaded.metadata.get("dataset_card_license"),
            "row_license_set": loaded.metadata.get("row_license_set"),
            "unique_repo_count": loaded.metadata.get("unique_repo_count"),
            "typescript_family": loaded.metadata.get("typescript_family"),
            "corpus_id": loaded.corpus_id,
            "eval_split": TYPESCRIPT_EVAL_FULL,
            "eval_source_splits": loaded.metadata.get("eval_source_splits"),
            "official_split_counts": loaded.metadata.get("official_split_counts"),
            "full_example_count": loaded.metadata.get("full_example_count"),
            "corpus_sample_size": loaded.metadata.get("corpus_sample_size"),
            "corpus_sampling_note": loaded.metadata.get("corpus_sampling_note"),
            "split_counts": loaded.metadata.get("split_counts"),
            "normalized_surface": "document.index_text",
            "phase_b_typescript_scope_note": "No semantic-hard TypeScript split was identifiable; this route uses the canonical TypeScript test split while preserving Phase B v3 retriever methodology.",
            "methodology": loaded.metadata.get("methodology"),
        },
    )
    write_phase_b_reports(payload, output_dir=output_dir)
    return payload


def run_phase_b5_typescript_full(
    *,
    bloom_path: Path,
    sample_size: int | None,
    top_k: int,
    output_dir: Path,
    cpu_timeout_seconds: float = PHASE_B5_CPU_TIMEOUT_SECONDS,
    corpus_sample_size: int | None = None,
) -> dict[str, Any]:
    bloom = _require_bloom_filter(bloom_path)
    loader = TypeScriptEvalLoader()
    loaded = loader.load_full_eval(sample_size=sample_size, corpus_sample_size=corpus_sample_size)
    accepted_examples, rejected_examples = _accepted_examples(loaded=loaded, bloom=bloom)

    retriever_summaries, rows = _run_phase_b_factories(
        retriever_factories=_phase_b5_typescript_retriever_factories(),
        loaded=loaded,
        accepted_examples=accepted_examples,
        top_k=top_k,
        cpu_timeout_seconds=cpu_timeout_seconds,
    )
    _mark_typescript_sieve_pending(retriever_summaries)
    findings, gates = _typescript_findings_and_gates(retriever_summaries, phase_label="Phase B.5 TypeScript full eval")
    payload = _phase_b_payload(
        loaded=loaded,
        accepted_examples=accepted_examples,
        rejected_examples=rejected_examples,
        retriever_summaries=retriever_summaries,
        rows=rows,
        findings=findings,
        gates=gates,
        benchmark={
            "phase": "B.5",
            "sample_size": sample_size,
            "top_k": top_k,
            "bloom_path": str(bloom_path),
            "cpu_timeout_seconds": cpu_timeout_seconds,
            "dataset_id": loaded.metadata.get("dataset_id"),
            "dataset_revision": loaded.revision,
            "dataset_language": loaded.metadata.get("dataset_language"),
            "dataset_card_license": loaded.metadata.get("dataset_card_license"),
            "row_license_set": loaded.metadata.get("row_license_set"),
            "unique_repo_count": loaded.metadata.get("unique_repo_count"),
            "typescript_family": loaded.metadata.get("typescript_family"),
            "corpus_id": loaded.corpus_id,
            "eval_split": TYPESCRIPT_EVAL_FULL,
            "eval_source_splits": loaded.metadata.get("eval_source_splits"),
            "official_split_counts": loaded.metadata.get("official_split_counts"),
            "full_example_count": loaded.metadata.get("full_example_count"),
            "corpus_sample_size": loaded.metadata.get("corpus_sample_size"),
            "corpus_sampling_note": loaded.metadata.get("corpus_sampling_note"),
            "split_counts": loaded.metadata.get("split_counts"),
            "normalized_surface": "document.index_text",
            "methodology": loaded.metadata.get("methodology"),
        },
    )
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
    output_dir: Path = typer.Option(PHASE_B_V3_RESULTS_DIR, file_okay=False),
) -> None:
    payload = run_phase_b_python_full(
        bloom_path=bloom_path,
        sample_size=sample_size,
        top_k=top_k,
        output_dir=output_dir,
    )
    by_name = {summary["retriever"]: summary for summary in payload["retriever_summaries"]}
    typer.echo(
        "Phase B v3 Python full benchmark complete: "
        f"BM25 recall@5={by_name['bm25']['recall@5']:.3f} "
        f"UniXcoder recall@5={by_name['unixcoder']['recall@5']:.3f} "
        f"LateOn-Code-edge recall@5={by_name['lateon-code-edge']['recall@5']:.3f} "
        f"LateOn-Code recall@5={by_name['lateon-code']['recall@5']:.3f} "
        f"CodeBERT-null recall@5={by_name['codebert']['recall@5']:.3f}"
    )


@app.command("phase-b-typescript-full")
def phase_b_typescript_full(
    bloom_path: Path = typer.Option(CORNSTACK_BLOOM_PATH, exists=False, dir_okay=False),
    sample_size: int = typer.Option(QUICKCHECK_SAMPLE_SIZE, min=1),
    top_k: int = typer.Option(max(RECALL_KS), min=1),
    output_dir: Path = typer.Option(PHASE_B_TYPESCRIPT_RESULTS_DIR, file_okay=False),
    corpus_sample_size: int | None = typer.Option(None, min=1),
) -> None:
    payload = run_phase_b_typescript_full(
        bloom_path=bloom_path,
        sample_size=sample_size,
        top_k=top_k,
        output_dir=output_dir,
        corpus_sample_size=corpus_sample_size,
    )
    by_name = {summary["retriever"]: summary for summary in payload["retriever_summaries"]}
    typer.echo(
        "Phase B v3 TypeScript full benchmark complete: "
        f"ripgrep recall@5={by_name['ripgrep']['recall@5']:.3f} "
        f"BM25 recall@5={by_name['bm25']['recall@5']:.3f} "
        f"UniXcoder recall@5={by_name['unixcoder']['recall@5']:.3f} "
        f"LateOn-Code-edge recall@5={by_name['lateon-code-edge']['recall@5']:.3f} "
        f"SIEVE recall@5={by_name['sieve']['recall@5']:.3f} (Phase 1 weights pending) "
        f"LateOn-Code recall@5={by_name['lateon-code']['recall@5']:.3f} "
        f"CodeBERT-null recall@5={by_name['codebert']['recall@5']:.3f}"
    )


@app.command("phase-b5-python-full")
def phase_b5_python_full(
    bloom_path: Path = typer.Option(CORNSTACK_BLOOM_PATH, exists=False, dir_okay=False),
    sample_size: int | None = typer.Option(None, min=1),
    top_k: int = typer.Option(max(RECALL_KS), min=1),
    output_dir: Path = typer.Option(PHASE_B5_RESULTS_DIR, file_okay=False),
    cpu_timeout_seconds: float = typer.Option(PHASE_B5_CPU_TIMEOUT_SECONDS, min=1.0),
) -> None:
    payload = run_phase_b5_python_full(
        bloom_path=bloom_path,
        sample_size=sample_size,
        top_k=top_k,
        output_dir=output_dir,
        cpu_timeout_seconds=cpu_timeout_seconds,
    )
    by_name = {summary["retriever"]: summary for summary in payload["retriever_summaries"]}
    typer.echo(
        "Phase B.5 Python full-eval benchmark complete: "
        f"ripgrep recall@5={by_name['ripgrep']['recall@5']:.3f} "
        f"BM25 recall@5={by_name['bm25']['recall@5']:.3f} "
        f"UniXcoder recall@5={by_name['unixcoder']['recall@5']:.3f} "
        f"LateOn-Code-edge recall@5={by_name['lateon-code-edge']['recall@5']:.3f} "
        f"SIEVE recall@5={by_name['sieve']['recall@5']:.3f} "
        f"LateOn-Code recall@5={by_name['lateon-code']['recall@5']:.3f} "
        f"CodeBERT-null recall@5={by_name['codebert']['recall@5']:.3f}"
    )


@app.command("phase-b5-typescript-full")
def phase_b5_typescript_full(
    bloom_path: Path = typer.Option(CORNSTACK_BLOOM_PATH, exists=False, dir_okay=False),
    sample_size: int | None = typer.Option(None, min=1),
    top_k: int = typer.Option(max(RECALL_KS), min=1),
    output_dir: Path = typer.Option(PHASE_B5_TYPESCRIPT_RESULTS_DIR, file_okay=False),
    cpu_timeout_seconds: float = typer.Option(PHASE_B5_CPU_TIMEOUT_SECONDS, min=1.0),
    corpus_sample_size: int | None = typer.Option(None, min=1),
) -> None:
    payload = run_phase_b5_typescript_full(
        bloom_path=bloom_path,
        sample_size=sample_size,
        top_k=top_k,
        output_dir=output_dir,
        cpu_timeout_seconds=cpu_timeout_seconds,
        corpus_sample_size=corpus_sample_size,
    )
    by_name = {summary["retriever"]: summary for summary in payload["retriever_summaries"]}
    typer.echo(
        "Phase B.5 TypeScript full-eval benchmark complete: "
        f"ripgrep recall@5={by_name['ripgrep']['recall@5']:.3f} "
        f"BM25 recall@5={by_name['bm25']['recall@5']:.3f} "
        f"UniXcoder recall@5={by_name['unixcoder']['recall@5']:.3f} "
        f"LateOn-Code-edge recall@5={by_name['lateon-code-edge']['recall@5']:.3f} "
        f"SIEVE recall@5={by_name['sieve']['recall@5']:.3f} (Phase 1 weights pending) "
        f"LateOn-Code recall@5={by_name['lateon-code']['recall@5']:.3f} "
        f"CodeBERT-null recall@5={by_name['codebert']['recall@5']:.3f}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
