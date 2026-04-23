from bench.loaders.coir import CoIRPythonLoader


def test_coir_python_loader_uses_official_test_pairs_only() -> None:
    query_rows = [
        {"_id": "q-train", "title": "train_fn", "partition": "train", "text": "def train_fn():\n    return 'train'", "language": "python", "meta_information": {"resource": "train.py"}},
        {"_id": "q-test-1", "title": "test_one", "partition": "test", "text": "def test_one():\n    return 1", "language": "python", "meta_information": {"resource": "pkg/test_one.py"}},
        {"_id": "q-test-2", "title": "test_two", "partition": "test", "text": "def test_two():\n    return 2", "language": "python", "meta_information": {"resource": "pkg/test_two.py"}},
    ]
    corpus_rows = [
        {"_id": "c-train", "title": "", "partition": "train", "text": "training helper text", "language": "python", "meta_information": {"resource": "train.txt"}},
        {"_id": "c-test-1", "title": "", "partition": "test", "text": "find the function that returns one", "language": "python", "meta_information": {"resource": "docs/one.txt"}},
        {"_id": "c-test-2", "title": "", "partition": "test", "text": "find the function that returns two", "language": "python", "meta_information": {"resource": "docs/two.txt"}},
    ]
    qrel_rows = [
        {"query-id": "q-train", "corpus-id": "c-train", "score": 1},
        {"query-id": "q-test-1", "corpus-id": "c-test-1", "score": 1},
        {"query-id": "q-test-2", "corpus-id": "c-test-2", "score": 1},
    ]

    loaded = CoIRPythonLoader._build_loaded_benchmark(
        query_rows=query_rows,
        corpus_rows=corpus_rows,
        qrel_rows=qrel_rows,
        sample_size=10,
    )

    assert loaded.source == "coir"
    assert loaded.language == "python"
    assert len(loaded.corpus) == 2
    assert len(loaded.examples) == 2
    assert loaded.examples[0].query in {"find the function that returns one", "find the function that returns two"}
    assert loaded.examples[0].ground_truth_code.startswith("def test_")
    assert all(example.metadata["split"] == "test" for example in loaded.examples)
