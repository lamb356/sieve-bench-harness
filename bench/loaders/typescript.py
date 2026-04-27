from __future__ import annotations

"""Canonical TypeScript code-search loader for Phase B TypeScript routes.

Dataset choice:
- Public Hugging Face dataset: Shuu12121/typescript-treesitter-dedupe-filtered-datasetsV2
- Pinned revision: 1e2fcd3764fb9126a33eaea58961925e667769f0
- Eval surface: the `test` split, 11,579 non-empty docstring/code pairs.
- Surface: real TypeScript `.ts` files, not ArkTS/HarmonyOS `.ets` files.

Why this route exists:
- The official CoIR/CodeSearchNet mirror used by the Python harness does not
  expose a TypeScript qrels split, and CodeXGLUE's code-to-text mirrors do not
  provide a clean maintained TypeScript retrieval split with comparable public
  qrels.
- This dataset provides permissively licensed, deduplicated, tree-sitter parsed
  TypeScript functions with natural-language docstrings, repository/path/url,
  and row-level license metadata. The harness keeps the same one-query/one-code
  pairing protocol used by the previous TypeScript route while replacing the
  underlying ArkTS data with canonical TypeScript.
"""

import hashlib
import random
from collections.abc import Iterable, Mapping
from typing import Any

from datasets import load_dataset

from bench.constants import (
    GLOBAL_RANDOM_SEED,
    TYPESCRIPT_CORPUS_ID,
    TYPESCRIPT_DATASET_ID,
    TYPESCRIPT_DATASET_LANGUAGE,
    TYPESCRIPT_DATASET_REVISION,
    TYPESCRIPT_EVAL_FULL,
    TYPESCRIPT_EVAL_FULL_QUERY_COUNT,
    TYPESCRIPT_EVAL_SOURCE_SPLITS,
    TYPESCRIPT_FAMILY_NAME,
    TYPESCRIPT_LANGUAGE,
    TYPESCRIPT_SOURCE_NAME,
    TYPESCRIPT_SPLIT_COUNTS,
)
from bench.contamination.normalize import normalize_for_search
from bench.loaders.base import CodeDocument, EvalExample, LoadedBenchmark

_SPLIT_ORDER = TYPESCRIPT_EVAL_SOURCE_SPLITS


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _row_digest(row: Mapping[str, Any], ordinal: int) -> str:
    digest_source = "\0".join(
        (
            _clean_text(row.get("repo")),
            _clean_text(row.get("path")),
            _clean_text(row.get("url")),
            _clean_text(row.get("func_name")),
            _clean_text(row.get("code")),
            str(ordinal),
        )
    )
    return hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:16]


def _document_id(split: str, ordinal: int, row: Mapping[str, Any]) -> str:
    return f"typescript-{split}-{ordinal:06d}-{_row_digest(row, ordinal)}"


def _query_id(split: str, ordinal: int) -> str:
    return f"typescript-query-{split}-{ordinal:06d}"


def _unique_document_path(split: str, ordinal: int, row: Mapping[str, Any]) -> str:
    # Multiple function rows can come from the same source file. Prefix with the
    # split/ordinal so file-backed retrievers such as ripgrep never overwrite
    # documents while still preserving the original `.ts` suffix.
    original_path = _clean_text(row.get("path")) or "unknown.ts"
    return f"typescript/{split}/{ordinal:06d}/{original_path}"


class TypeScriptEvalLoader:
    name = "typescript-treesitter-dedupe"

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
                code = _clean_text(row.get("code"))
                if not docstring or not code:
                    skipped_empty += 1
                    continue
                ordered_rows.append((split, ordinal, row))

        corpus: list[CodeDocument] = []
        examples: list[EvalExample] = []
        qrels: dict[str, dict[str, int]] = {}
        for split, ordinal, row in ordered_rows:
            code = _clean_text(row.get("code"))
            query = _clean_text(row.get("docstring"))
            doc_id = _document_id(split, ordinal, row)
            query_id = _query_id(split, ordinal)
            path = _unique_document_path(split, ordinal, row)
            original_path = _clean_text(row.get("path"))
            source_language = _clean_text(row.get("language")) or TYPESCRIPT_DATASET_LANGUAGE
            repo = _clean_text(row.get("repo"))
            func_name = _clean_text(row.get("func_name"))
            url = _clean_text(row.get("url"))
            license_name = _clean_text(row.get("license"))
            document_metadata = {
                "dataset_id": TYPESCRIPT_DATASET_ID,
                "dataset_revision": revision,
                "dataset_language": TYPESCRIPT_DATASET_LANGUAGE,
                "typescript_family": TYPESCRIPT_FAMILY_NAME,
                "source_language": source_language,
                "split": split,
                "repo": repo,
                "original_path": original_path,
                "func_name": func_name,
                "url": url,
                "license": license_name,
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
                        "repo": repo,
                        "original_path": original_path,
                        "func_name": func_name,
                        "url": url,
                        "license": license_name,
                    },
                )
            )
            qrels[query_id] = {doc_id: 1}

        full_example_count = len(examples)
        full_license_set = tuple(sorted({str(document.metadata.get("license")) for document in corpus if document.metadata.get("license")}))
        full_repo_count = len({str(document.metadata.get("repo")) for document in corpus if document.metadata.get("repo")})
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

        if expected_example_count is not None and full_example_count != expected_example_count:
            raise ValueError(
                f"Expected {expected_example_count} examples for {eval_split}, got {full_example_count}. "
                "Pinned TypeScript dataset may have changed or the wrong eval split was loaded."
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
                "dataset_card_license": "apache-2.0",
                "row_license_set": full_license_set,
                "unique_repo_count": full_repo_count,
                "typescript_family": TYPESCRIPT_FAMILY_NAME,
                "eval_split": eval_split,
                "eval_source_splits": tuple(TYPESCRIPT_EVAL_SOURCE_SPLITS),
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
                    "One natural-language docstring query per canonical TypeScript function; "
                    "the paired TypeScript function is the only relevant document. "
                    "Rows come from the pinned test split of a permissively licensed tree-sitter/deduplicated TypeScript dataset."
                ),
            },
        )

    def _load_rows_by_split(self) -> dict[str, list[Mapping[str, Any]]]:
        rows_by_split: dict[str, list[Mapping[str, Any]]] = {}
        for split in TYPESCRIPT_EVAL_SOURCE_SPLITS:
            dataset = load_dataset(TYPESCRIPT_DATASET_ID, revision=self.revision, split=split)
            rows_by_split[split] = [dict(row) for row in dataset]
        return rows_by_split

    def load(self, sample_size: int | None = None, *, corpus_sample_size: int | None = None) -> LoadedBenchmark:
        return self.load_full_eval(sample_size=sample_size, corpus_sample_size=corpus_sample_size)

    def load_full_eval(self, sample_size: int | None = None, *, corpus_sample_size: int | None = None) -> LoadedBenchmark:
        return self._build_loaded_benchmark(
            rows_by_split=self._load_rows_by_split(),
            sample_size=sample_size,
            corpus_sample_size=corpus_sample_size,
            eval_split=TYPESCRIPT_EVAL_FULL,
            expected_example_count=TYPESCRIPT_EVAL_FULL_QUERY_COUNT,
            revision=self.revision,
        )


def load_typescript_eval_full(*, sample_size: int | None = None, corpus_sample_size: int | None = None) -> LoadedBenchmark:
    """Load the canonical public TypeScript eval distribution."""
    return TypeScriptEvalLoader().load_full_eval(sample_size=sample_size, corpus_sample_size=corpus_sample_size)
