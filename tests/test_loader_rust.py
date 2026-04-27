from __future__ import annotations

import pytest

from bench.constants import (
    RUST_DATASET_ID,
    RUST_DATASET_REVISION,
    RUST_EVAL_FULL,
    RUST_EVAL_FULL_QUERY_COUNT,
)
from bench.loaders.rust import RustEvalLoader


def _rust_row(index: int, *, split: str = "test") -> dict[str, str]:
    return {
        "repo": "example/rust-repo",
        "path": f"src/helpers/example_{index}.rs",
        "language": "rust",
        "func_name": f"format_example_{index}",
        "docstring": f"Format example value {index} for display",
        "code": f"pub fn format_example_{index}(value: i32) -> String {{\n    value.to_string()\n}}",
        "url": f"https://example.invalid/rust-repo/blob/main/src/helpers/example_{index}.rs#L1-L3",
        "license": "MIT",
        "split": split,
    }


def test_loader_rust_returns_pinned_treesitter_test_split_metadata() -> None:
    rows_by_split = {
        "train": [],
        "validation": [],
        "test": [_rust_row(i, split="test") for i in range(RUST_EVAL_FULL_QUERY_COUNT)],
    }

    loaded = RustEvalLoader._build_loaded_benchmark(
        rows_by_split=rows_by_split,
        sample_size=None,
        eval_split=RUST_EVAL_FULL,
        expected_example_count=RUST_EVAL_FULL_QUERY_COUNT,
    )

    assert loaded.source == "rust-treesitter-dedupe"
    assert loaded.language == "rust"
    assert loaded.metadata["dataset_id"] == RUST_DATASET_ID
    assert loaded.metadata["dataset_revision"] == RUST_DATASET_REVISION
    assert loaded.metadata["dataset_language"] == "rust"
    assert loaded.metadata["dataset_card_license"] == "apache-2.0"
    assert loaded.metadata["row_license_set"] == ("MIT",)
    assert loaded.metadata["unique_repo_count"] == 1
    assert loaded.metadata["eval_split"] == RUST_EVAL_FULL
    assert loaded.metadata["eval_source_splits"] == ("test",)
    assert loaded.metadata["expected_example_count"] == RUST_EVAL_FULL_QUERY_COUNT
    assert len(loaded.corpus) == RUST_EVAL_FULL_QUERY_COUNT
    assert len(loaded.examples) == RUST_EVAL_FULL_QUERY_COUNT
    assert loaded.examples[0].query.startswith("Format example value")
    assert loaded.examples[0].ground_truth_code.startswith("pub fn format_example")
    assert loaded.corpus[0].path.endswith(".rs")
    assert loaded.corpus[0].metadata["source_language"] == "rust"
    assert loaded.corpus[0].document_id.startswith("rust-test-")
    assert loaded.examples[0].metadata["query_id"].startswith("rust-query-test-")
    assert "No official CodeSearchNet/CoIR/CornStack Rust retrieval qrels" in loaded.metadata["methodology"]


def test_loader_rust_can_bound_corpus_for_practical_validation_runs() -> None:
    rows_by_split = {
        "train": [],
        "validation": [],
        "test": [_rust_row(i, split="test") for i in range(20)],
    }

    loaded = RustEvalLoader._build_loaded_benchmark(
        rows_by_split=rows_by_split,
        sample_size=5,
        corpus_sample_size=10,
        eval_split=RUST_EVAL_FULL,
    )

    positive_ids = {str(example.metadata["ground_truth_document_id"]) for example in loaded.examples}
    corpus_ids = {document.document_id for document in loaded.corpus}
    assert len(loaded.examples) == 5
    assert len(loaded.corpus) == 10
    assert positive_ids <= corpus_ids
    assert loaded.metadata["corpus_sampling_note"]


def test_loader_rust_validates_expected_count_before_sampling() -> None:
    with pytest.raises(ValueError, match="Expected 2 examples"):
        RustEvalLoader._build_loaded_benchmark(
            rows_by_split={"train": [], "validation": [], "test": [_rust_row(1, split="test")]},
            sample_size=1,
            eval_split=RUST_EVAL_FULL,
            expected_example_count=2,
        )


def test_loader_rust_preserves_custom_revision_metadata() -> None:
    loaded = RustEvalLoader._build_loaded_benchmark(
        rows_by_split={"train": [], "validation": [], "test": [_rust_row(1, split="test")]},
        sample_size=None,
        eval_split=RUST_EVAL_FULL,
        revision="custom-rust-revision",
    )

    assert loaded.revision == "custom-rust-revision"
    assert loaded.metadata["dataset_revision"] == "custom-rust-revision"
    assert loaded.corpus[0].metadata["dataset_revision"] == "custom-rust-revision"
    assert loaded.examples[0].metadata["dataset_revision"] == "custom-rust-revision"
