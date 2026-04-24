from __future__ import annotations

from pathlib import Path
from typing import Any

from bench.constants import BENCH_MODEL_CACHE_DIR, UNIXCODER_MODEL_ID, UNIXCODER_MODEL_REVISION
from bench.retrievers.codebert import TransformerEmbeddingBackend, TransformerEmbeddingRetriever

UNIXCODER_ENCODER_ONLY_TOKEN = "<encoder-only>"
UNIXCODER_ENCODER_ONLY_TOKEN_ID = 6
UNIXCODER_TRUNCATION_STRATEGY = (
    "unixcoder-encoder-only;documents=head+tail-balanced-to-512;"
    "queries=head-only-to-512;mean-pool-attention-mask"
)


class UniXcoderEmbeddingBackend(TransformerEmbeddingBackend):
    def __init__(
        self,
        *,
        model_id: str = UNIXCODER_MODEL_ID,
        model_revision: str = UNIXCODER_MODEL_REVISION,
        cache_dir: Path = BENCH_MODEL_CACHE_DIR,
        max_length: int = 512,
        batch_size: int = 16,
        device: str | None = None,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        if tokenizer is None or model is None:
            super().__init__(
                model_id=model_id,
                model_revision=model_revision,
                cache_dir=cache_dir,
                max_length=max_length,
                batch_size=batch_size,
                device=device,
            )
        else:
            import torch

            self.model_id = model_id
            self.model_revision = model_revision
            self.cache_dir = Path(cache_dir)
            self.max_length = int(max_length)
            self.batch_size = int(batch_size)
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._torch = torch
            self.tokenizer = tokenizer
            self.model = model.to(self.device)
            self.model.eval()
        self.truncation_strategy = UNIXCODER_TRUNCATION_STRATEGY
        self.encoder_only_token_id = self._encoder_only_token_id()

    def _encoder_only_token_id(self) -> int:
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            token_id = convert(UNIXCODER_ENCODER_ONLY_TOKEN)
            unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
            if token_id is not None and token_id != unk_token_id:
                return int(token_id)
        return UNIXCODER_ENCODER_ONLY_TOKEN_ID

    def _token_ids(self, text: str, *, for_document: bool) -> list[int]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        token_budget = max(1, self.max_length - 4)
        if len(token_ids) > token_budget:
            if not for_document:
                token_ids = token_ids[:token_budget]
            else:
                head = token_budget // 2
                tail = token_budget - head
                token_ids = token_ids[:head] + token_ids[-tail:]
        cls_token_id = int(getattr(self.tokenizer, "cls_token_id"))
        sep_token_id = int(getattr(self.tokenizer, "sep_token_id"))
        return [cls_token_id, self.encoder_only_token_id, sep_token_id, *token_ids, sep_token_id]

    def _manual_encoder_only_batch(self, texts: list[str], *, for_document: bool) -> dict[str, Any]:
        encoded_items = []
        for text in texts:
            input_ids = self._token_ids(text, for_document=for_document)
            encoded_items.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
        return self.tokenizer.pad(
            encoded_items,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def _official_encoder_only_batch(self, texts: list[str]) -> dict[str, Any] | None:
        tokenize = getattr(self.model, "tokenize", None)
        if not callable(tokenize):
            return None
        attempts = [
            {"mode": UNIXCODER_ENCODER_ONLY_TOKEN},
            {"mode": UNIXCODER_ENCODER_ONLY_TOKEN, "max_length": self.max_length, "padding": True},
        ]
        for kwargs in attempts:
            try:
                encoded = tokenize(texts, **kwargs)
            except TypeError:
                continue
            if isinstance(encoded, dict) and "input_ids" in encoded and "attention_mask" in encoded:
                torch = self._torch
                return {key: value if hasattr(value, "to") else torch.as_tensor(value) for key, value in encoded.items()}
        return None

    def _encode(self, texts, *, for_document: bool):  # noqa: ANN001
        vectors: list[Any] = []
        torch = self._torch
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = list(texts[start : start + self.batch_size])
                encoded = self._official_encoder_only_batch(batch_texts)
                if encoded is None:
                    encoded = self._manual_encoder_only_batch(batch_texts, for_document=for_document)
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                hidden = outputs.last_hidden_state
                attention_mask = encoded["attention_mask"]
                if attention_mask.ndim == 3:
                    attention_mask = torch.diagonal(attention_mask, dim1=-2, dim2=-1)
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.detach().cpu().numpy().astype("float32"))
        if not vectors:
            import numpy as np

            return np.empty((0, 0), dtype="float32")
        import numpy as np

        return np.concatenate(vectors, axis=0)


class UniXcoderRetriever(TransformerEmbeddingRetriever):
    name = "unixcoder"
    display_name = "UniXcoder"

    def __init__(
        self,
        *,
        model_cache_dir: Path = BENCH_MODEL_CACHE_DIR,
        embedding_backend: Any | None = None,
        batch_size: int = 16,
    ) -> None:
        unixcoder_backend = embedding_backend or UniXcoderEmbeddingBackend(
            model_id=UNIXCODER_MODEL_ID,
            model_revision=UNIXCODER_MODEL_REVISION,
            cache_dir=model_cache_dir,
            batch_size=batch_size,
        )
        super().__init__(
            model_id=UNIXCODER_MODEL_ID,
            model_revision=UNIXCODER_MODEL_REVISION,
            model_cache_dir=model_cache_dir,
            embedding_backend=unixcoder_backend,
            batch_size=batch_size,
        )

    def embedding_metadata(self) -> dict[str, Any]:
        metadata = super().embedding_metadata()
        metadata["truncation_strategy"] = str(getattr(self.embedding_backend, "truncation_strategy", UNIXCODER_TRUNCATION_STRATEGY))
        metadata["encoder_only_token_id"] = int(getattr(self.embedding_backend, "encoder_only_token_id", UNIXCODER_ENCODER_ONLY_TOKEN_ID))
        metadata["max_tokens"] = {"query": int(metadata.get("max_length", 512)), "document": int(metadata.get("max_length", 512))}
        return metadata
