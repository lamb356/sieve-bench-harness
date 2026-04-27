from __future__ import annotations

"""CoIR/CodeSearchNet Go loader for Phase B Go routes.

Dataset choice:
- Public Hugging Face dataset: CoIR-Retrieval/CodeSearchNet
- Pinned revision: 25e0292562b7bee26dd9b2d83a03981795862c77
- Eval surface: official Go test qrels, 8,122 qrel rows over CodeSearchNet Go.

Why this route exists:
- Go is one of the official CodeSearchNet languages and is also present in the
  CoIR public retrieval assets with language-specific queries/corpus/qrels.
- This gives the Go track higher-quality provenance than a constructed
  docstring-pair corpus: public benchmark qrels define the query/code pairing.
"""

import random
from collections.abc import Iterable
from typing import Any

from datasets import load_dataset

from bench.constants import (
    GLOBAL_RANDOM_SEED,
    GO_CORPUS_ID,
    GO_CORPUS_PATH,
    GO_DATASET_ID,
    GO_DATASET_REVISION,
    GO_EVAL_FULL,
    GO_EVAL_FULL_QUERY_COUNT,
    GO_LANGUAGE,
    GO_QRELS_TEST_PATH,
    GO_QUERIES_PATH,
    GO_SOURCE_NAME,
)
from bench.contamination.normalize import normalize_for_search
from bench.loaders.base import CodeDocument, EvalExample, LoadedBenchmark


def _dataset_url(relative_path: str, *, revision: str) -> str:
    return f"https://huggingface.co/datasets/{GO_DATASET_ID}/resolve/{revision}/{relative_path}"


def _resource_path(row: dict[str, Any], *, document_id: str | None = None) -> str:
    meta = row.get("meta_information") or {}
    resource = str(meta.get("resource") or "").strip()
    if resource:
        base = resource if resource.endswith(".go") else f"{resource}.go"
    else:
        title = str(row.get("title") or row.get("_id") or "document").strip()
        safe_title = title.replace("::", "/").replace(".", "/").replace(" ", "_")
        suffix = ".go" if not safe_title.endswith(".go") else ""
        base = f"{safe_title}{suffix}"
    base = base.lstrip("/")
    if document_id:
        return f"go/{document_id}/{base}"
    return base if base.startswith("go/") else f"go/{base}"


class CoIRGoLoader:
    name = GO_SOURCE_NAME

    def __init__(self, *, revision: str = GO_DATASET_REVISION) -> None:
        self.revision = revision

    @staticmethod
    def _build_loaded_benchmark(
        *,
        query_rows: Iterable[dict[str, Any]],
        corpus_rows: Iterable[dict[str, Any]],
        qrel_rows: Iterable[dict[str, Any]],
        sample_size: int | None,
        corpus_sample_size: int | None = None,
        eval_split: str = GO_EVAL_FULL,
        expected_example_count: int | None = None,
        revision: str = GO_DATASET_REVISION,
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
        }

        corpus_documents = [
            CodeDocument(
                document_id=query_id,
                path=_resource_path(row, document_id=query_id),
                code=str(row.get("text") or ""),
                language=GO_LANGUAGE,
                index_text=normalize_for_search(str(row.get("text") or ""), language=GO_LANGUAGE),
                metadata={
                    "dataset_id": GO_DATASET_ID,
                    "dataset_revision": revision,
                    "dataset_language": GO_LANGUAGE,
                    "title": str(row.get("title") or ""),
                    "split": str(row.get("partition") or ""),
                    "resource": _resource_path(row),
                    "unique_resource": _resource_path(row, document_id=query_id),
                },
            )
            for query_id, row in sorted(query_rows_by_id.items())
        ]

        examples: list[EvalExample] = []
        for qrel in qrel_rows:
            code_row_id = str(qrel["query-id"])
            nl_row_id = str(qrel["corpus-id"])
            if code_row_id not in query_rows_by_id or nl_row_id not in corpus_rows_by_id:
                continue
            code_row = query_rows_by_id[code_row_id]
            nl_row = corpus_rows_by_id[nl_row_id]
            query_id = f"go-query-{nl_row_id}-{code_row_id}"
            examples.append(
                EvalExample(
                    query=str(nl_row.get("text") or ""),
                    ground_truth_code=str(code_row.get("text") or ""),
                    ground_truth_path=_resource_path(code_row, document_id=code_row_id),
                    language=GO_LANGUAGE,
                    source=GO_SOURCE_NAME,
                    corpus_id=GO_CORPUS_ID,
                    metadata={
                        "query_id": query_id,
                        "ground_truth_document_id": code_row_id,
                        "coir_query_id": code_row_id,
                        "coir_corpus_id": nl_row_id,
                        "ground_truth_title": str(code_row.get("title") or ""),
                        "split": "test",
                        "dataset_id": GO_DATASET_ID,
                        "dataset_revision": revision,
                        "dataset_language": GO_LANGUAGE,
                    },
                )
            )

        examples = sorted(examples, key=lambda item: str(item.metadata["query_id"]))
        full_example_count = len(examples)
        if expected_example_count is not None and full_example_count != expected_example_count:
            raise ValueError(
                f"Expected {expected_example_count} examples for {eval_split}, got {full_example_count}. "
                "Pinned CoIR Go data may have changed or the wrong eval split was loaded."
            )

        if sample_size is not None and sample_size < len(examples):
            rng = random.Random(GLOBAL_RANDOM_SEED)
            examples = sorted(rng.sample(examples, k=sample_size), key=lambda item: str(item.metadata["query_id"]))

        if corpus_sample_size is not None and corpus_sample_size < len(corpus_documents):
            required_document_ids = {
                str(example.metadata["ground_truth_document_id"])
                for example in examples
            }
            if len(required_document_ids) > corpus_sample_size:
                raise ValueError(
                    f"corpus_sample_size={corpus_sample_size} is smaller than the {len(required_document_ids)} required positive documents"
                )
            required = [document for document in corpus_documents if document.document_id in required_document_ids]
            negatives = [document for document in corpus_documents if document.document_id not in required_document_ids]
            rng = random.Random(GLOBAL_RANDOM_SEED + 1)
            sampled_negative_ids = {
                document.document_id
                for document in rng.sample(negatives, k=min(len(negatives), corpus_sample_size - len(required)))
            }
            selected_ids = required_document_ids | sampled_negative_ids
            corpus_documents = [document for document in corpus_documents if document.document_id in selected_ids]

        qrels = {
            str(example.metadata["query_id"]): {str(example.metadata["ground_truth_document_id"]): 1}
            for example in examples
        }

        return LoadedBenchmark(
            source=GO_SOURCE_NAME,
            language=GO_LANGUAGE,
            revision=revision,
            corpus_id=GO_CORPUS_ID,
            corpus=tuple(corpus_documents),
            examples=tuple(examples),
            metadata={
                "dataset_id": GO_DATASET_ID,
                "dataset_revision": revision,
                "dataset_language": GO_LANGUAGE,
                "dataset_card_license": None,
                "eval_split": eval_split,
                "expected_example_count": expected_example_count,
                "full_example_count": full_example_count,
                "query_row_count": len(query_rows_by_id),
                "corpus_row_count": len(corpus_rows_by_id),
                "qrel_row_count": len(qrel_rows),
                "document_count": len(corpus_documents),
                "corpus_sample_size": corpus_sample_size,
                "corpus_sampling_note": (
                    "Corpus was deterministically reduced to sampled positives plus random negatives for a bounded validation run."
                    if corpus_sample_size is not None and corpus_sample_size < len(query_rows_by_id)
                    else None
                ),
                "qrels": qrels,
                "methodology": (
                    "Official CoIR/CodeSearchNet Go test qrels: index Go code rows from the test partition, "
                    "use paired natural-language rows as queries, and preserve the public qrel mapping."
                ),
            },
        )

    def load(self, sample_size: int | None = None, *, corpus_sample_size: int | None = None) -> LoadedBenchmark:
        return self.load_full_eval(sample_size=sample_size, corpus_sample_size=corpus_sample_size)

    def load_full_eval(self, sample_size: int | None = None, *, corpus_sample_size: int | None = None) -> LoadedBenchmark:
        query_rows = list(load_dataset("parquet", data_files=_dataset_url(GO_QUERIES_PATH, revision=self.revision), split="train"))
        corpus_rows = list(load_dataset("parquet", data_files=_dataset_url(GO_CORPUS_PATH, revision=self.revision), split="train"))
        qrel_rows = list(load_dataset("parquet", data_files=_dataset_url(GO_QRELS_TEST_PATH, revision=self.revision), split="train"))
        return self._build_loaded_benchmark(
            query_rows=query_rows,
            corpus_rows=corpus_rows,
            qrel_rows=qrel_rows,
            sample_size=sample_size,
            corpus_sample_size=corpus_sample_size,
            eval_split=GO_EVAL_FULL,
            expected_example_count=GO_EVAL_FULL_QUERY_COUNT,
            revision=self.revision,
        )


def load_go_eval_full(*, sample_size: int | None = None, corpus_sample_size: int | None = None) -> LoadedBenchmark:
    """Load the official CoIR/CodeSearchNet Go test-qrels distribution."""
    return CoIRGoLoader().load_full_eval(sample_size=sample_size, corpus_sample_size=corpus_sample_size)
