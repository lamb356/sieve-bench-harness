import pytest

from bench.constants import (
    TYPESCRIPT_DATASET_ID,
    TYPESCRIPT_DATASET_REVISION,
    TYPESCRIPT_EVAL_FULL,
    TYPESCRIPT_EVAL_FULL_QUERY_COUNT,
)
from bench.loaders.typescript import TypeScriptEvalLoader


def _typescript_row(index: int, *, split: str = "test") -> dict[str, str]:
    return {
        "repo": "example/repo",
        "path": f"src/helpers/example_{index}.ts",
        "language": "typescript",
        "func_name": f"formatExample{index}",
        "docstring": f"Format example value {index} for display",
        "code": f"export function formatExample{index}(value: number): string {{\n  return value.toString();\n}}",
        "url": f"https://example.invalid/repo/blob/main/src/helpers/example_{index}.ts#L1-L3",
        "license": "MIT",
        "split": split,
    }


def test_loader_typescript_returns_canonical_typescript_test_split_size() -> None:
    rows_by_split = {
        "train": [],
        "validation": [],
        "test": [_typescript_row(i, split="test") for i in range(TYPESCRIPT_EVAL_FULL_QUERY_COUNT)],
    }

    loaded = TypeScriptEvalLoader._build_loaded_benchmark(
        rows_by_split=rows_by_split,
        sample_size=None,
        eval_split=TYPESCRIPT_EVAL_FULL,
        expected_example_count=TYPESCRIPT_EVAL_FULL_QUERY_COUNT,
    )

    assert loaded.source == "typescript-treesitter-dedupe"
    assert loaded.language == "typescript"
    assert loaded.metadata["dataset_id"] == "Shuu12121/typescript-treesitter-dedupe-filtered-datasetsV2"
    assert loaded.metadata["dataset_revision"] == TYPESCRIPT_DATASET_REVISION
    assert loaded.metadata["dataset_language"] == "typescript"
    assert loaded.metadata["dataset_card_license"] == "apache-2.0"
    assert loaded.metadata["row_license_set"] == ("MIT",)
    assert loaded.metadata["unique_repo_count"] == 1
    assert loaded.metadata["typescript_family"] == "TypeScript"
    assert loaded.metadata["eval_split"] == TYPESCRIPT_EVAL_FULL
    assert loaded.metadata["eval_source_splits"] == ("test",)
    assert loaded.metadata["expected_example_count"] == TYPESCRIPT_EVAL_FULL_QUERY_COUNT
    assert len(loaded.corpus) == TYPESCRIPT_EVAL_FULL_QUERY_COUNT
    assert len(loaded.examples) == TYPESCRIPT_EVAL_FULL_QUERY_COUNT
    assert loaded.examples[0].query.startswith("Format example value")
    assert loaded.examples[0].ground_truth_code.startswith("export function formatExample")
    assert loaded.corpus[0].path.endswith(".ts")
    assert loaded.corpus[0].metadata["source_language"] == "typescript"
    assert loaded.corpus[0].metadata["license"] == "MIT"
    assert loaded.corpus[0].metadata["repo"] == "example/repo"
    assert loaded.corpus[0].document_id.startswith("typescript-test-")
    assert loaded.examples[0].metadata["query_id"].startswith("typescript-query-test-")


def test_loader_typescript_can_bound_corpus_for_practical_validation_runs() -> None:
    rows_by_split = {
        "train": [],
        "validation": [],
        "test": [_typescript_row(i, split="test") for i in range(20)],
    }

    loaded = TypeScriptEvalLoader._build_loaded_benchmark(
        rows_by_split=rows_by_split,
        sample_size=5,
        corpus_sample_size=10,
        eval_split=TYPESCRIPT_EVAL_FULL,
    )

    positive_ids = {str(example.metadata["ground_truth_document_id"]) for example in loaded.examples}
    corpus_ids = {document.document_id for document in loaded.corpus}
    assert len(loaded.examples) == 5
    assert len(loaded.corpus) == 10
    assert positive_ids <= corpus_ids
    assert loaded.metadata["corpus_sampling_note"]


def test_loader_typescript_validates_expected_count_before_sampling() -> None:
    with pytest.raises(ValueError, match="Expected 2 examples"):
        TypeScriptEvalLoader._build_loaded_benchmark(
            rows_by_split={"train": [], "validation": [], "test": [_typescript_row(1, split="test")]},
            sample_size=1,
            eval_split=TYPESCRIPT_EVAL_FULL,
            expected_example_count=2,
        )


def test_loader_typescript_preserves_custom_revision_metadata() -> None:
    loaded = TypeScriptEvalLoader._build_loaded_benchmark(
        rows_by_split={"train": [], "validation": [], "test": [_typescript_row(1, split="test")]},
        sample_size=None,
        eval_split=TYPESCRIPT_EVAL_FULL,
        revision="custom-test-revision",
    )

    assert TYPESCRIPT_DATASET_ID == "Shuu12121/typescript-treesitter-dedupe-filtered-datasetsV2"
    assert loaded.revision == "custom-test-revision"
    assert loaded.metadata["dataset_revision"] == "custom-test-revision"
    assert loaded.corpus[0].metadata["dataset_revision"] == "custom-test-revision"
    assert loaded.examples[0].metadata["dataset_revision"] == "custom-test-revision"
