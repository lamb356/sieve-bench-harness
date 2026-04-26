from bench.constants import TYPESCRIPT_EVAL_FULL, TYPESCRIPT_EVAL_FULL_QUERY_COUNT
from bench.loaders.typescript import TypeScriptEvalLoader


def _arkts_row(index: int, *, split: str = "test") -> dict[str, str]:
    return {
        "nwo": "example/repo",
        "sha": "a" * 40,
        "path": f"src/helpers/example_{index}.ets",
        "language": "arkts",
        "identifier": f"formatExample{index}",
        "docstring": f"Format example value {index} for display",
        "function": f"export function formatExample{index}(value: number): string {{\n  return value.toString();\n}}",
        "ast_function": "",
        "obf_function": "",
        "url": f"https://example.invalid/repo/src/helpers/example_{index}.ets#L1-L3",
        "function_sha": f"{index:040x}"[-40:],
        "source": "github",
        "split": split,
    }


def test_loader_typescript_returns_correct_split_size() -> None:
    rows_by_split = {
        "train": [_arkts_row(i, split="train") for i in range(2)],
        "validation": [_arkts_row(i + 2, split="validation") for i in range(1)],
        "test": [_arkts_row(i + 3, split="test") for i in range(TYPESCRIPT_EVAL_FULL_QUERY_COUNT - 3)],
    }

    loaded = TypeScriptEvalLoader._build_loaded_benchmark(
        rows_by_split=rows_by_split,
        sample_size=None,
        eval_split=TYPESCRIPT_EVAL_FULL,
        expected_example_count=TYPESCRIPT_EVAL_FULL_QUERY_COUNT,
    )

    assert loaded.source == "arkts-codesearch"
    assert loaded.language == "typescript"
    assert loaded.metadata["dataset_language"] == "arkts"
    assert loaded.metadata["typescript_family"] == "ArkTS"
    assert loaded.metadata["eval_split"] == TYPESCRIPT_EVAL_FULL
    assert loaded.metadata["expected_example_count"] == TYPESCRIPT_EVAL_FULL_QUERY_COUNT
    assert len(loaded.corpus) == TYPESCRIPT_EVAL_FULL_QUERY_COUNT
    assert len(loaded.examples) == TYPESCRIPT_EVAL_FULL_QUERY_COUNT
    assert loaded.examples[0].query.startswith("Format example value")
    assert loaded.examples[0].ground_truth_code.startswith("export function formatExample")
    assert loaded.corpus[0].path.endswith(".ets")
    assert loaded.corpus[0].metadata["source_language"] == "arkts"


def test_loader_typescript_can_bound_corpus_for_practical_validation_runs() -> None:
    rows_by_split = {
        "train": [_arkts_row(i, split="train") for i in range(20)],
        "validation": [],
        "test": [],
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


def test_loader_typescript_preserves_custom_revision_metadata() -> None:
    loaded = TypeScriptEvalLoader._build_loaded_benchmark(
        rows_by_split={"train": [_arkts_row(1, split="train")], "validation": [], "test": []},
        sample_size=None,
        eval_split=TYPESCRIPT_EVAL_FULL,
        revision="custom-test-revision",
    )

    assert loaded.revision == "custom-test-revision"
    assert loaded.metadata["dataset_revision"] == "custom-test-revision"
    assert loaded.corpus[0].metadata["dataset_revision"] == "custom-test-revision"
    assert loaded.examples[0].metadata["dataset_revision"] == "custom-test-revision"
