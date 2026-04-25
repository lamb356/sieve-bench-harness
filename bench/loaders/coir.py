from __future__ import annotations

"""CoIR Python loader for Phase A.

Upstream provenance:
- Public Hugging Face dataset: CoIR-Retrieval/CodeSearchNet
- Pinned revision: 25e0292562b7bee26dd9b2d83a03981795862c77
- Benchmark family: CoIR / MTEB public code-retrieval evaluation assets

Audit note:
- The Hugging Face card does not currently expose strong license metadata in card_data.
- For Phase A we treat this as a public benchmark provenance issue to document, not a blocking route failure.
- This loader only consumes the official public CoIR test qrels and public associated parquet files.
"""

from typing import Any, Iterable
import random

from datasets import load_dataset

from bench.contamination.normalize import normalize_for_search
from bench.constants import (
    COIR_CORPUS_ID,
    COIR_CORPUS_PATH,
    COIR_DATASET_ID,
    COIR_DATASET_REVISION,
    COIR_LANGUAGE,
    COIR_QRELS_TEST_PATH,
    COIR_QUERIES_PATH,
    COIR_SOURCE_NAME,
    GLOBAL_RANDOM_SEED,
    PYTHON_EVAL_FULL,
    PYTHON_EVAL_FULL_QUERY_COUNT,
)
from bench.loaders.base import CodeDocument, EvalExample, LoadedBenchmark


def _dataset_url(relative_path: str) -> str:
    return f"https://huggingface.co/datasets/{COIR_DATASET_ID}/resolve/{COIR_DATASET_REVISION}/{relative_path}"


def _resource_path(row: dict[str, Any]) -> str:
    meta = row.get("meta_information") or {}
    resource = str(meta.get("resource") or "").strip()
    if resource:
        return resource
    title = str(row.get("title") or row.get("_id") or "document").strip()
    safe_title = title.replace("::", "/").replace(".", "/").replace(" ", "_")
    suffix = ".py" if not safe_title.endswith(".py") else ""
    return f"python/{safe_title}{suffix}"


class CoIRPythonLoader:
    name = "coir-python"

    def __init__(self, *, revision: str = COIR_DATASET_REVISION) -> None:
        self.revision = revision

    @staticmethod
    def _build_loaded_benchmark(
        *,
        query_rows: Iterable[dict[str, Any]],
        corpus_rows: Iterable[dict[str, Any]],
        qrel_rows: Iterable[dict[str, Any]],
        sample_size: int | None,
        eval_split: str = PYTHON_EVAL_FULL,
        expected_example_count: int | None = None,
    ) -> LoadedBenchmark:
        query_rows = list(query_rows)
        corpus_rows = list(corpus_rows)
        qrel_rows = list(qrel_rows)
        query_rows_by_id = {
            str(row["_id"]): row
            for row in query_rows
            if str(row.get("partition", "")).lower() == "test"
        }
        corpus_rows_by_id = {
            str(row["_id"]): row
            for row in corpus_rows
            if str(row.get("partition", "")).lower() == "test"
        }

        corpus_documents = tuple(
            CodeDocument(
                document_id=query_id,
                path=_resource_path(row),
                code=str(row.get("text") or ""),
                language=COIR_LANGUAGE,
                index_text=normalize_for_search(str(row.get("text") or ""), language=COIR_LANGUAGE),
                metadata={
                    "title": str(row.get("title") or ""),
                    "split": str(row.get("partition") or ""),
                    "resource": _resource_path(row),
                },
            )
            for query_id, row in sorted(query_rows_by_id.items())
        )

        examples: list[EvalExample] = []
        for qrel in qrel_rows:
            code_row_id = str(qrel["query-id"])
            nl_row_id = str(qrel["corpus-id"])
            if code_row_id not in query_rows_by_id or nl_row_id not in corpus_rows_by_id:
                continue
            code_row = query_rows_by_id[code_row_id]
            nl_row = corpus_rows_by_id[nl_row_id]
            examples.append(
                EvalExample(
                    query=str(nl_row.get("text") or ""),
                    ground_truth_code=str(code_row.get("text") or ""),
                    ground_truth_path=_resource_path(code_row),
                    language=COIR_LANGUAGE,
                    source=COIR_SOURCE_NAME,
                    corpus_id=COIR_CORPUS_ID,
                    metadata={
                        "query_id": nl_row_id,
                        "ground_truth_document_id": code_row_id,
                        "coir_query_id": code_row_id,
                        "coir_corpus_id": nl_row_id,
                        "ground_truth_title": str(code_row.get("title") or ""),
                        "split": "test",
                    },
                )
            )

        examples = sorted(examples, key=lambda item: str(item.metadata["query_id"]))
        full_example_count = len(examples)
        if sample_size is not None and sample_size < len(examples):
            rng = random.Random(GLOBAL_RANDOM_SEED)
            examples = sorted(rng.sample(examples, k=sample_size), key=lambda item: str(item.metadata["query_id"]))
        if sample_size is None and expected_example_count is not None and full_example_count != expected_example_count:
            raise ValueError(
                f"Expected {expected_example_count} examples for {eval_split}, got {full_example_count}. "
                "Pinned CoIR data may have changed or the wrong eval split was loaded."
            )

        return LoadedBenchmark(
            source=COIR_SOURCE_NAME,
            language=COIR_LANGUAGE,
            revision=COIR_DATASET_REVISION,
            corpus_id=COIR_CORPUS_ID,
            corpus=corpus_documents,
            examples=tuple(examples),
            metadata={
                "dataset_id": COIR_DATASET_ID,
                "dataset_revision": COIR_DATASET_REVISION,
                "eval_split": eval_split,
                "expected_example_count": expected_example_count,
                "full_example_count": full_example_count,
                "query_row_count": len(query_rows_by_id),
                "corpus_row_count": len(corpus_rows_by_id),
                "qrel_row_count": len(qrel_rows),
            },
        )

    def load(self, sample_size: int | None = None) -> LoadedBenchmark:
        query_rows = list(load_dataset("parquet", data_files=_dataset_url(COIR_QUERIES_PATH), split="train"))
        corpus_rows = list(load_dataset("parquet", data_files=_dataset_url(COIR_CORPUS_PATH), split="train"))
        qrel_rows = list(load_dataset("parquet", data_files=_dataset_url(COIR_QRELS_TEST_PATH), split="train"))
        return self._build_loaded_benchmark(
            query_rows=query_rows,
            corpus_rows=corpus_rows,
            qrel_rows=qrel_rows,
            sample_size=sample_size,
            eval_split=PYTHON_EVAL_FULL,
            expected_example_count=None,
        )

    def load_full_eval(self, sample_size: int | None = None) -> LoadedBenchmark:
        query_rows = list(load_dataset("parquet", data_files=_dataset_url(COIR_QUERIES_PATH), split="train"))
        corpus_rows = list(load_dataset("parquet", data_files=_dataset_url(COIR_CORPUS_PATH), split="train"))
        qrel_rows = list(load_dataset("parquet", data_files=_dataset_url(COIR_QRELS_TEST_PATH), split="train"))
        return self._build_loaded_benchmark(
            query_rows=query_rows,
            corpus_rows=corpus_rows,
            qrel_rows=qrel_rows,
            sample_size=sample_size,
            eval_split=PYTHON_EVAL_FULL,
            expected_example_count=PYTHON_EVAL_FULL_QUERY_COUNT if sample_size is None else None,
        )


def load_python_eval_full(*, sample_size: int | None = None) -> LoadedBenchmark:
    """Load the full CoIR/CodeSearchNet Python eval distribution for Phase B.5."""
    return CoIRPythonLoader().load_full_eval(sample_size=sample_size)
