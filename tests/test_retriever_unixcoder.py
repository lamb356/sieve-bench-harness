from __future__ import annotations

from types import SimpleNamespace

import torch

from bench.constants import UNIXCODER_MODEL_ID, UNIXCODER_MODEL_REVISION
from bench.retrievers.unixcoder import UNIXCODER_ENCODER_ONLY_TOKEN_ID, UniXcoderEmbeddingBackend, UniXcoderRetriever


class FakeUniXcoderTokenizer:
    cls_token_id = 0
    sep_token_id = 2
    pad_token_id = 1
    unk_token_id = 3
    model_max_length = 512

    def __init__(self) -> None:
        self.prepare_for_model_calls = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        base = sum(ord(ch) for ch in text) % 50
        return [10 + base, 11 + base, 12 + base]

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<encoder-only>":
            return UNIXCODER_ENCODER_ONLY_TOKEN_ID
        return self.unk_token_id

    def prepare_for_model(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
        self.prepare_for_model_calls += 1
        raise AssertionError("plain RoBERTa prepare_for_model path must not be used for UniXcoder")

    def pad(self, encoded_items, *, padding, max_length, return_tensors):  # noqa: ANN001
        del padding, max_length
        assert return_tensors == "pt"
        width = max(len(item["input_ids"]) for item in encoded_items)
        rows = []
        masks = []
        for item in encoded_items:
            pad_width = width - len(item["input_ids"])
            rows.append(item["input_ids"] + [self.pad_token_id] * pad_width)
            masks.append(item["attention_mask"] + [0] * pad_width)
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class FakeUniXcoderModel:
    def __init__(self) -> None:
        self.input_id_batches: list[list[list[int]]] = []

    def to(self, _device):  # noqa: ANN001
        return self

    def eval(self) -> None:
        return None

    def __call__(self, **encoded):  # noqa: ANN003
        input_ids = encoded["input_ids"]
        self.input_id_batches.append(input_ids.detach().cpu().tolist())
        hidden = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 4)
        return SimpleNamespace(last_hidden_state=hidden)


def test_unixcoder_backend_injects_encoder_only_token_for_queries_and_documents(tmp_path) -> None:
    tokenizer = FakeUniXcoderTokenizer()
    model = FakeUniXcoderModel()
    backend = UniXcoderEmbeddingBackend(
        model_id=UNIXCODER_MODEL_ID,
        model_revision=UNIXCODER_MODEL_REVISION,
        cache_dir=tmp_path / "models",
        tokenizer=tokenizer,
        model=model,
        device="cpu",
    )

    backend.encode_queries(["natural language query"])
    backend.encode_documents(["def add(a, b): return a + b"])

    query_ids = model.input_id_batches[0][0]
    document_ids = model.input_id_batches[1][0]
    assert query_ids[:3] == [tokenizer.cls_token_id, UNIXCODER_ENCODER_ONLY_TOKEN_ID, tokenizer.sep_token_id]
    assert document_ids[:3] == [tokenizer.cls_token_id, UNIXCODER_ENCODER_ONLY_TOKEN_ID, tokenizer.sep_token_id]
    assert query_ids[-1] == tokenizer.sep_token_id
    assert document_ids[-1] == tokenizer.sep_token_id
    assert tokenizer.prepare_for_model_calls == 0


def test_unixcoder_retriever_uses_encoder_only_backend_by_default(tmp_path) -> None:
    retriever = UniXcoderRetriever(
        model_cache_dir=tmp_path / "models",
        embedding_backend=UniXcoderEmbeddingBackend(
            model_id=UNIXCODER_MODEL_ID,
            model_revision=UNIXCODER_MODEL_REVISION,
            cache_dir=tmp_path / "models",
            tokenizer=FakeUniXcoderTokenizer(),
            model=FakeUniXcoderModel(),
            device="cpu",
        ),
    )

    metadata = retriever.embedding_metadata()

    assert metadata["model_id"] == UNIXCODER_MODEL_ID
    assert metadata["encoder_only_token_id"] == UNIXCODER_ENCODER_ONLY_TOKEN_ID
    assert "unixcoder-encoder-only" in metadata["truncation_strategy"]
    assert metadata["max_tokens"] == {"query": 512, "document": 512}
