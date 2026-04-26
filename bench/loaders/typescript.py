from __future__ import annotations

"""TypeScript-family code-search loader for Phase B TypeScript routes.

Dataset choice:
- Public Hugging Face dataset: hreyulog/arkts-code-docstring
- Pinned revision: b10cf6c85767455aef80fc02557614a408c183c1
- Rows: 24,452 non-empty docstring/function pairs across train/validation/test.
- Surface: ArkTS `.ets`, a TypeScript-family language used by OpenHarmony.

Why this route exists:
- CodeSearchNet and CoIR do not expose a clean official TypeScript split with
  CodeSearchNet-style natural-language/code qrels.
- CrossCodeEval TypeScript is a cross-file code-completion benchmark, not an
  NL-to-code retrieval benchmark comparable to the current Python Phase B
  harness.
- ArkTS-CodeSearch is the highest-quality public TypeScript-family source found
  with deterministic code/docstring pairs and enough examples for a full
  distribution route. The report labels this caveat explicitly.
"""

import hashlib
import random
from collections.abc import Iterable, Mapping
from typing import Any

from datasets import DatasetDict, load_dataset

from bench.constants import (
    GLOBAL_RANDOM_SEED,
    TYPESCRIPT_CORPUS_ID,
    TYPESCRIPT_DATASET_ID,
    TYPESCRIPT_DATASET_LANGUAGE,
    TYPESCRIPT_DATASET_REVISION,
    TYPESCRIPT_EVAL_FULL,
    TYPESCRIPT_EVAL_FULL_QUERY_COUNT,
    TYPESCRIPT_FAMILY_NAME,
    TYPESCRIPT_LANGUAGE,
    TYPESCRIPT_SOURCE_NAME,
    TYPESCRIPT_SPLIT_COUNTS,
)
from bench.contamination.normalize import normalize_for_search
from bench.loaders.base import CodeDocument, EvalExample, LoadedBenchmark

_SPLIT_ORDER = ("train", "validation", "test")


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _document_id(split: str, ordinal: int, row: Mapping[str, Any]) -> str:
    function_sha = _clean_text(row.get("function_sha"))
    if function_sha:
        suffix = function_sha[:16]
    else:
        digest_source = "\0".join((_clean_text(row.get("sha")), _clean_text(row.get("path")), str(ordinal)))
        suffix = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:16]
    return f"arkts-{split}-{ordinal:06d}-{suffix}"


def _query_id(split: str, ordinal: int) -> str:
    return f"arkts-query-{split}-{ordinal:06d}"


def _unique_document_path(split: str, ordinal: int, row: Mapping[str, Any]) -> str:
    # Multiple function rows can come from the same source file. Prefix with the
    # split/ordinal so file-backed retrievers such as ripgrep never overwrite
    # documents while still preserving the original `.ets` TypeScript-family suffix.
    original_path = _clean_text(row.get("path")) or "unknown.ets"
    return f"arkts/{split}/{ordinal:06d}/{original_path}"


class TypeScriptEvalLoader:
    name = "arkts-codesearch-typescript"

    def __init__(self, *, revision: str = TYPESCRIPT_DATASET_REVISION) -> None:
        self.revision = revision

    @staticmethod
    def _build_loaded_benchmark(
        *,
        rows_by_split: Mapping[str, Iterable[Mapping[str, Any]]],
        sample_size: int | None,
        corpus_sample_size: int | None = None,
        eval_split: str = TYPESCRIPT_EVAL_FULL,
        expected_example_count: int | None = None,
        revision: str = TYPESCRIPT_DATASET_REVISION,
    ) -> LoadedBenchmark:
        ordered_rows: list[tuple[str, int, Mapping[str, Any]]] = []
        split_counts: dict[str, int] = {}
        skipped_empty = 0
        for split in _SPLIT_ORDER:
            rows = list(rows_by_split.get(split, ()))
            split_counts[split] = len(rows)
            for ordinal, row in enumerate(rows):
                docstring = _clean_text(row.get("docstring"))
                function = _clean_text(row.get("function"))
                if not docstring or not function:
                    skipped_empty += 1
                    continue
                ordered_rows.append((split, ordinal, row))

        corpus: list[CodeDocument] = []
        examples: list[EvalExample] = []
        qrels: dict[str, dict[str, int]] = {}
        for split, ordinal, row in ordered_rows:
            code = _clean_text(row.get("function"))
            query = _clean_text(row.get("docstring"))
            doc_id = _document_id(split, ordinal, row)
            query_id = _query_id(split, ordinal)
            path = _unique_document_path(split, ordinal, row)
            original_path = _clean_text(row.get("path"))
            source_language = _clean_text(row.get("language")) or TYPESCRIPT_DATASET_LANGUAGE
            document_metadata = {
                "dataset_id": TYPESCRIPT_DATASET_ID,
                "dataset_revision": revision,
                "dataset_language": TYPESCRIPT_DATASET_LANGUAGE,
                "typescript_family": TYPESCRIPT_FAMILY_NAME,
                "source_language": source_language,
                "split": split,
                "nwo": _clean_text(row.get("nwo")),
                "sha": _clean_text(row.get("sha")),
                "original_path": original_path,
                "identifier": _clean_text(row.get("identifier")),
                "url": _clean_text(row.get("url")),
                "function_sha": _clean_text(row.get("function_sha")),
                "source": _clean_text(row.get("source")),
            }
            corpus.append(
                CodeDocument(
                    document_id=doc_id,
                    path=path,
                    code=code,
                    language=TYPESCRIPT_LANGUAGE,
                    index_text=normalize_for_search(code, language=TYPESCRIPT_LANGUAGE),
                    metadata=document_metadata,
                )
            )
            examples.append(
                EvalExample(
                    query=query,
                    ground_truth_code=code,
                    ground_truth_path=path,
                    language=TYPESCRIPT_LANGUAGE,
                    source=TYPESCRIPT_SOURCE_NAME,
                    corpus_id=TYPESCRIPT_CORPUS_ID,
                    metadata={
                        "query_id": query_id,
                        "ground_truth_document_id": doc_id,
                        "dataset_id": TYPESCRIPT_DATASET_ID,
                        "dataset_revision": revision,
                        "dataset_language": TYPESCRIPT_DATASET_LANGUAGE,
                        "typescript_family": TYPESCRIPT_FAMILY_NAME,
                        "source_language": source_language,
                        "split": split,
                        "nwo": document_metadata["nwo"],
                        "sha": document_metadata["sha"],
                        "original_path": original_path,
                        "identifier": document_metadata["identifier"],
                        "url": document_metadata["url"],
                        "function_sha": document_metadata["function_sha"],
                    },
                )
            )
            qrels[query_id] = {doc_id: 1}

        full_example_count = len(examples)
        if sample_size is not None and sample_size < len(examples):
            rng = random.Random(GLOBAL_RANDOM_SEED)
            sampled = sorted(rng.sample(range(len(examples)), k=sample_size))
            examples = [examples[index] for index in sampled]

        if corpus_sample_size is not None and corpus_sample_size < len(corpus):
            required_document_ids = {
                str(example.metadata["ground_truth_document_id"])
                for example in examples
            }
            if len(required_document_ids) > corpus_sample_size:
                raise ValueError(
                    f"corpus_sample_size={corpus_sample_size} is smaller than the {len(required_document_ids)} required positive documents"
                )
            required = [document for document in corpus if document.document_id in required_document_ids]
            negatives = [document for document in corpus if document.document_id not in required_document_ids]
            rng = random.Random(GLOBAL_RANDOM_SEED + 1)
            sampled_negative_ids = {
                document.document_id
                for document in rng.sample(negatives, k=min(len(negatives), corpus_sample_size - len(required)))
            }
            selected_ids = required_document_ids | sampled_negative_ids
            corpus = [document for document in corpus if document.document_id in selected_ids]

        qrels = {
            str(example.metadata["query_id"]): {str(example.metadata["ground_truth_document_id"]): 1}
            for example in examples
        }

        if sample_size is None and expected_example_count is not None and full_example_count != expected_example_count:
            raise ValueError(
                f"Expected {expected_example_count} examples for {eval_split}, got {full_example_count}. "
                "Pinned ArkTS-CodeSearch data may have changed or the wrong eval split was loaded."
            )

        return LoadedBenchmark(
            source=TYPESCRIPT_SOURCE_NAME,
            language=TYPESCRIPT_LANGUAGE,
            revision=revision,
            corpus_id=TYPESCRIPT_CORPUS_ID,
            corpus=tuple(corpus),
            examples=tuple(examples),
            metadata={
                "dataset_id": TYPESCRIPT_DATASET_ID,
                "dataset_revision": revision,
                "dataset_language": TYPESCRIPT_DATASET_LANGUAGE,
                "typescript_family": TYPESCRIPT_FAMILY_NAME,
                "eval_split": eval_split,
                "expected_example_count": expected_example_count,
                "full_example_count": full_example_count,
                "split_counts": split_counts,
                "document_count": len(corpus),
                "corpus_sample_size": corpus_sample_size,
                "corpus_sampling_note": (
                    "Corpus was deterministically reduced to sampled positives plus random negatives for a bounded validation run."
                    if corpus_sample_size is not None and corpus_sample_size < full_example_count
                    else None
                ),
                "qrel_count": len(qrels),
                "skipped_empty_count": skipped_empty,
                "official_split_counts": dict(TYPESCRIPT_SPLIT_COUNTS),
                "qrels": qrels,
                "methodology": (
                    "One natural-language docstring query per ArkTS function; the paired function is the only relevant document. "
                    "ArkTS is treated as a TypeScript-family eval route because no clean CodeSearchNet/CoIR TypeScript split was found."
                ),
            },
        )

    def _load_rows_by_split(self) -> dict[str, list[Mapping[str, Any]]]:
        dataset = load_dataset(TYPESCRIPT_DATASET_ID, revision=self.revision)
        if not isinstance(dataset, DatasetDict):
            raise TypeError(f"Expected DatasetDict for {TYPESCRIPT_DATASET_ID}, got {type(dataset).__name__}")
        return {split: list(dataset[split]) for split in _SPLIT_ORDER}

    def load(self, sample_size: int | None = None, *, corpus_sample_size: int | None = None) -> LoadedBenchmark:
        return self.load_full_eval(sample_size=sample_size, corpus_sample_size=corpus_sample_size)

    def load_full_eval(self, sample_size: int | None = None, *, corpus_sample_size: int | None = None) -> LoadedBenchmark:
        return self._build_loaded_benchmark(
            rows_by_split=self._load_rows_by_split(),
            sample_size=sample_size,
            corpus_sample_size=corpus_sample_size,
            eval_split=TYPESCRIPT_EVAL_FULL,
            expected_example_count=TYPESCRIPT_EVAL_FULL_QUERY_COUNT if sample_size is None else None,
            revision=self.revision,
        )


def load_typescript_eval_full(*, sample_size: int | None = None, corpus_sample_size: int | None = None) -> LoadedBenchmark:
    """Load the full public ArkTS-CodeSearch TypeScript-family eval distribution."""
    return TypeScriptEvalLoader().load_full_eval(sample_size=sample_size, corpus_sample_size=corpus_sample_size)
