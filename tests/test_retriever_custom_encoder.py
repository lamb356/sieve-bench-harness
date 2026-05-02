from __future__ import annotations

import torch

from bench.loaders.base import CodeDocument
from bench.retrievers import RETRIEVER_REPORT_METADATA
from bench.retrievers.custom_encoder import CustomEncoderRetriever, _score_query_against_documents


class MockPatchMaxSimBackend:
    model_id = "test/mock-custom-byte-encoder"
    checkpoint_path = "/tmp/mock-custom-byte-encoder.pt"
    checkpoint_sha256 = "test-sha256"
    param_count = 1234
    query_seq = 256
    doc_seq = 2048
    event_threshold = 0.5
    device = "cpu"

    def __init__(self) -> None:
        self.encoded_document_batches: list[list[str]] = []
        self.encoded_query_batches: list[list[str]] = []

    def encode_documents(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        self.encoded_document_batches.append(list(texts))
        vectors = []
        masks = []
        for text in texts:
            lowered = text.lower()
            if "rawonlysignal" in lowered:
                patch_vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
            elif "decoy" in lowered:
                patch_vectors = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
            else:
                patch_vectors = torch.tensor([[0.2, 0.8], [0.1, 0.9]], dtype=torch.float32)
            vectors.append(patch_vectors)
            masks.append(torch.tensor([True, True], dtype=torch.bool))
        return torch.stack(vectors, dim=0), torch.stack(masks, dim=0)

    def encode_queries(self, texts: list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        self.encoded_query_batches.append(list(texts))
        query_vectors = []
        query_weights = []
        for text in texts:
            if "rawonlysignal" in text.lower():
                query_vectors.append(torch.tensor([[1.0, 0.0]], dtype=torch.float32))
            else:
                query_vectors.append(torch.tensor([[0.0, 1.0]], dtype=torch.float32))
            query_weights.append(torch.tensor([1.0], dtype=torch.float32))
        return query_vectors, query_weights

    def metadata(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "param_count": self.param_count,
            "query_seq": self.query_seq,
            "doc_seq": self.doc_seq,
            "event_threshold": self.event_threshold,
            "device": self.device,
        }


def test_custom_encoder_retriever_encodes_raw_code_not_index_text() -> None:
    backend = MockPatchMaxSimBackend()
    retriever = CustomEncoderRetriever(encoder_backend=backend)
    corpus = [
        CodeDocument(
            document_id="raw-doc",
            path="python/raw.py",
            code="# keep this comment/whitespace\ndef rawonlysignal():\n    return 'target'\n",
            language="python",
            index_text="metadata decoy only",
        ),
        CodeDocument(
            document_id="index-text-decoy",
            path="python/decoy.py",
            code="def unrelated_decoy():\n    return 'noise'\n",
            language="python",
            index_text="rawonlysignal rawonlysignal rawonlysignal",
        ),
    ]

    retriever.index(corpus)
    results = retriever.search("find rawonlysignal implementation", k=2)

    assert backend.encoded_document_batches == [[document.code for document in corpus]]
    assert "# keep this comment/whitespace" in backend.encoded_document_batches[0][0]
    assert all("rawonlysignal rawonlysignal" not in text for text in backend.encoded_document_batches[0])
    assert results[0].document_id == "raw-doc"
    assert results[0].score > results[1].score
    assert results[0].metadata["ranker"] == "custom-encoder"
    assert results[0].metadata["score_function"] == "alpha_weighted_patch_maxsim"


def test_fully_masked_documents_score_below_real_negative_matches() -> None:
    query_vectors = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    query_weights = torch.tensor([1.0], dtype=torch.float32)
    doc_vectors = torch.tensor(
        [
            [[0.0, 0.0]],
            [[-1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    doc_mask = torch.tensor([[False], [True]], dtype=torch.bool)

    scores = _score_query_against_documents(query_vectors, query_weights, doc_vectors, doc_mask)

    assert torch.isfinite(scores).all()
    assert scores[0] < scores[1]
    assert scores[1].item() == -1.0


def test_custom_encoder_is_registered_as_our_model_hero_row() -> None:
    metadata = RETRIEVER_REPORT_METADATA["custom-encoder"]

    assert metadata.table == "hero"
    assert metadata.role == "our_model"
    assert metadata.params == "4.2M"
    assert metadata.display_name == "Custom encoder"
