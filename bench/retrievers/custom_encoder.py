from __future__ import annotations

import hashlib
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from bench.loaders.base import CodeDocument
from bench.metrics.performance import summarize_latency
from bench.retrievers.base import SearchResult

CUSTOM_ENCODER_CHECKPOINT_ENV = "CUSTOM_ENCODER_CHECKPOINT_PATH"
CUSTOM_ENCODER_MODEL_ID = "sieve/custom-byte-encoder-phase1"
CUSTOM_ENCODER_QUERY_SEQ = 256
CUSTOM_ENCODER_DOC_SEQ = 2048
CUSTOM_ENCODER_EVENT_THRESHOLD = 0.05
CUSTOM_ENCODER_TRAIN_LOOPS = 6
CUSTOM_ENCODER_SCORE_FUNCTION = "alpha_weighted_patch_maxsim"
CUSTOM_ENCODER_TRUNCATION_STRATEGY = (
    "raw-utf8-bytes;queries=head-only-to-256;documents=head-only-to-2048;"
    "patches=window24-stride8;retrieval-mask=patch_mask+event_scores>=0.05;"
    "query-alpha-weighted-maxsim;eval_numeric_harden=True"
)
EVAL_NUMERIC_EPS = 1e-8
EVAL_MASKED_SCORE_FLOOR = -1.0e4
NGRAM_RANGE = (3, 4, 5, 6)
MAX_NGRAM = max(NGRAM_RANGE)


def _checkpoint_path_from_env() -> Path:
    raw_path = os.environ.get(CUSTOM_ENCODER_CHECKPOINT_ENV)
    if raw_path is None or not raw_path.strip():
        raise RuntimeError(
            f"Custom encoder retriever is experimental and requires {CUSTOM_ENCODER_CHECKPOINT_ENV}; "
            "set it to a checkpoint path to enable this retriever."
        )
    return Path(raw_path).expanduser()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _encode_text_batch(texts: Sequence[str], max_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    batch = torch.zeros((len(texts), max_length), dtype=torch.long, device=device)
    lengths = torch.zeros((len(texts),), dtype=torch.long, device=device)
    for row, text in enumerate(texts):
        payload = text.encode("utf-8", errors="replace")[:max_length]
        if payload:
            values = torch.tensor(list(payload), dtype=torch.long, device=device)
            batch[row, : values.numel()] = values
            lengths[row] = values.numel()
    return batch, lengths


def _finite_eval_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_eval_vectors(tensor: torch.Tensor) -> torch.Tensor:
    tensor = _finite_eval_tensor(tensor)
    denom = tensor.norm(dim=-1, keepdim=True).clamp_min(EVAL_NUMERIC_EPS)
    return tensor / denom


def _sanitize_query_weights(weights: torch.Tensor, *, patch_count: int) -> torch.Tensor:
    weights = _finite_eval_tensor(weights).clamp_min(0.0)
    if patch_count <= 0:
        return weights.new_zeros((0,), dtype=torch.float32)
    if weights.numel() != patch_count:
        weights = weights.reshape(-1)[:patch_count]
        if weights.numel() < patch_count:
            weights = F.pad(weights, (0, patch_count - weights.numel()))
    denom = weights.sum().clamp_min(EVAL_NUMERIC_EPS)
    if not bool(torch.isfinite(denom).item()) or float(denom.item()) <= EVAL_NUMERIC_EPS:
        return torch.full((patch_count,), 1.0 / float(patch_count), dtype=torch.float32, device=weights.device)
    return weights / denom


def build_retrieval_mask(patch_mask: torch.Tensor, event_scores: torch.Tensor, threshold: float) -> torch.Tensor:
    return patch_mask.to(dtype=torch.bool) & (_finite_eval_tensor(event_scores).to(device=patch_mask.device) >= float(threshold))


class ByteEmbedding(nn.Module):
    def __init__(self, dim: int = 16, ngram_vocab: int = 65_536, ngram_range: tuple[int, ...] = NGRAM_RANGE) -> None:
        super().__init__()
        self.dim = dim
        self.ngram_vocab = ngram_vocab
        self.ngram_range = ngram_range
        self.byte_table = nn.Embedding(256, dim)
        self.ngram_table = nn.Embedding(ngram_vocab * len(ngram_range), dim)
        base_powers = [257**power for power in range(MAX_NGRAM)]
        hash_rows = []
        for ngram in ngram_range:
            hash_rows.append([0] * (MAX_NGRAM - ngram) + base_powers[:ngram])
        self.register_buffer(
            "hash_weights",
            torch.tensor(hash_rows, dtype=torch.long).view(1, 1, len(ngram_range), MAX_NGRAM),
            persistent=False,
        )
        self.register_buffer(
            "table_offsets",
            torch.arange(len(ngram_range), dtype=torch.long).view(1, 1, len(ngram_range)) * ngram_vocab,
            persistent=False,
        )
        self.register_buffer(
            "ngram_sizes",
            torch.tensor(ngram_range, dtype=torch.long).view(1, 1, len(ngram_range)),
            persistent=False,
        )

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        byte_ids = byte_ids.to(torch.long)
        seq_len = byte_ids.size(1)
        base = self.byte_table(byte_ids)
        if seq_len == 0:
            return base
        padded = F.pad(byte_ids, (MAX_NGRAM - 1, 0))
        windows = padded.unfold(dimension=1, size=MAX_NGRAM, step=1)
        hashes = (windows.unsqueeze(-2) * self.hash_weights).sum(dim=-1).remainder(self.ngram_vocab)
        indices = hashes + self.table_offsets
        embeds = self.ngram_table(indices)
        valid_mask = (torch.arange(seq_len, device=byte_ids.device).view(1, seq_len, 1) + 1) >= self.ngram_sizes.to(
            byte_ids.device
        )
        augment = (embeds * valid_mask.unsqueeze(-1)).sum(dim=-2) / len(self.ngram_range)
        return base + augment


class PrePatchSmoother(nn.Module):
    def __init__(self, dim: int = 16, kernel_size: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2), bias=False).to(
            memory_format=torch.channels_last
        )
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, byte_mask: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 0:
            return x
        y = x.transpose(1, 2).unsqueeze(2).contiguous()
        y = y.to(memory_format=torch.channels_last)
        y = self.conv(y)
        y = y.squeeze(2).transpose(1, 2).contiguous()
        y = self.norm(y)
        y = self.act(y)
        return y * byte_mask.unsqueeze(-1).to(dtype=y.dtype)


class PatchExtractor(nn.Module):
    def __init__(self, window: int = 24, stride: int = 8) -> None:
        super().__init__()
        self.window = window
        self.stride = stride

    def _padded_length(self, seq_len: int) -> int:
        if seq_len <= 0:
            return 0
        patch_count = math.ceil(seq_len / self.stride)
        return max(self.window, (patch_count - 1) * self.stride + self.window)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, seq_len, dim = x.shape
        lengths = lengths.to(device=x.device, dtype=torch.long).clamp(min=0, max=seq_len)
        if seq_len == 0:
            empty_bool = torch.zeros((batch_size, 0), device=x.device, dtype=torch.bool)
            empty_long = torch.zeros((batch_size, 0), device=x.device, dtype=torch.long)
            return {
                "patch_inputs": x.new_zeros(batch_size, 0, dim),
                "patch_lengths": torch.zeros((batch_size,), device=x.device, dtype=torch.long),
                "patch_mask": empty_bool,
                "patch_positions": empty_long,
                "patch_sizes": empty_long,
                "group_counts": empty_long,
            }
        padded_len = self._padded_length(seq_len)
        pad_right = padded_len - seq_len
        byte_mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        if pad_right > 0:
            x = F.pad(x, (0, 0, 0, pad_right))
            byte_mask = F.pad(byte_mask, (0, pad_right), value=False)
        patch_windows = x.unfold(dimension=1, size=self.window, step=self.stride).permute(0, 1, 3, 2).contiguous()
        mask_windows = byte_mask.unfold(dimension=1, size=self.window, step=self.stride)
        window_counts = mask_windows.sum(dim=-1, dtype=torch.long)
        pooled = (patch_windows * mask_windows.unsqueeze(-1).to(dtype=x.dtype)).sum(dim=2)
        pooled = pooled / window_counts.clamp_min(1).unsqueeze(-1).to(dtype=x.dtype)
        patch_count = pooled.size(1)
        patch_positions = (torch.arange(patch_count, device=x.device, dtype=torch.long) * self.stride).unsqueeze(0).expand(
            batch_size, -1
        )
        patch_sizes = (lengths.unsqueeze(1) - patch_positions).clamp(min=0, max=self.stride)
        patch_mask = patch_sizes > 0
        patch_lengths = patch_mask.sum(dim=-1, dtype=torch.long)
        pooled = pooled * patch_mask.unsqueeze(-1).to(dtype=pooled.dtype)
        return {
            "patch_inputs": pooled,
            "patch_lengths": patch_lengths,
            "patch_mask": patch_mask,
            "patch_positions": patch_positions,
            "patch_sizes": patch_sizes,
            "group_counts": patch_sizes,
        }


def _apply_patch_mask(x: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
    return x * patch_mask.unsqueeze(-1).to(dtype=x.dtype)


class LoopedMixerBlock(nn.Module):
    def __init__(self, dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 0:
            return x
        masked_x = _apply_patch_mask(x, patch_mask)
        y = self.fc2(self.act(self.fc1(self.norm(masked_x))))
        return _apply_patch_mask(masked_x + y, patch_mask)


class LoopedMixer(nn.Module):
    def __init__(self, dim: int = 64, hidden_dim: int = 128, num_loops: int = CUSTOM_ENCODER_TRAIN_LOOPS) -> None:
        super().__init__()
        if num_loops < 1:
            raise ValueError(f"num_loops must be >= 1, got {num_loops}")
        self.num_loops = int(num_loops)
        self.block = LoopedMixerBlock(dim=dim, hidden_dim=hidden_dim)

    def _resolve_loops(self, loops: int | None) -> int:
        if loops is None:
            return self.num_loops
        if loops < 1:
            raise ValueError(f"loops must be >= 1, got {loops}")
        return int(loops)

    def forward(self, x: torch.Tensor, patch_mask: torch.Tensor, *, loops: int | None = None) -> torch.Tensor:
        state = x
        for _ in range(self._resolve_loops(loops)):
            state = self.block(state, patch_mask)
        return state


class PatchGate(nn.Module):
    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, patch_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.size(1) == 0:
            empty = x.new_zeros(x.size(0), 0)
            return {"patch_states": x, "gate_probs": empty, "gate_values": empty, "keep_mask": patch_mask}
        gate_probs = torch.sigmoid(self.proj(x)).squeeze(-1)
        gate_probs = gate_probs * patch_mask.to(dtype=gate_probs.dtype)
        patch_states = x * gate_probs.unsqueeze(-1)
        patch_states = patch_states * patch_mask.unsqueeze(-1).to(dtype=patch_states.dtype)
        return {"patch_states": patch_states, "gate_probs": gate_probs, "gate_values": gate_probs, "keep_mask": patch_mask}


@dataclass
class NeuralEventBatch:
    scores: torch.Tensor
    mask: torch.Tensor
    counts: torch.Tensor


class ProbeScorer(nn.Module):
    def __init__(self, state_dim: int = 64, num_probes: int = 12) -> None:
        super().__init__()
        self.query_probes = nn.Parameter(torch.randn(num_probes, state_dim) * 0.02)
        self.query_weight_proj = nn.Linear(state_dim, 1)
        nn.init.normal_(self.query_weight_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.query_weight_proj.bias)

    def compute_query_patch_weights(self, query_patch_states: torch.Tensor, force_uniform: bool = False) -> torch.Tensor:
        if query_patch_states.size(1) == 0:
            return query_patch_states.new_zeros(query_patch_states.size(0), 0)
        if force_uniform:
            return torch.full(
                (query_patch_states.size(0), query_patch_states.size(1)),
                1.0 / float(query_patch_states.size(1)),
                device=query_patch_states.device,
                dtype=torch.float32,
            )
        logits = self.query_weight_proj(query_patch_states.float()).squeeze(-1)
        return torch.softmax(logits, dim=-1)

    def forward(self, patch_states: torch.Tensor, patch_mask: torch.Tensor, event_threshold: float) -> NeuralEventBatch:
        if patch_states.size(1) == 0:
            zero = patch_states.new_zeros(patch_states.size(0), 0)
            counts = patch_states.new_zeros(patch_states.size(0), dtype=torch.long)
            return NeuralEventBatch(scores=zero, mask=zero.bool(), counts=counts)
        probes = F.normalize(self.query_probes.float(), dim=-1)
        states = F.normalize(patch_states.float(), dim=-1)
        scores = torch.einsum("bpd,kd->bpk", states, probes).max(dim=-1).values
        event_mask = (scores >= event_threshold) & patch_mask
        event_counts = event_mask.sum(dim=-1, dtype=torch.long)
        return NeuralEventBatch(scores=scores, mask=event_mask, counts=event_counts)


class CustomByteEncoder(nn.Module):
    def __init__(self, *, event_threshold: float = CUSTOM_ENCODER_EVENT_THRESHOLD, mixer_num_loops: int = CUSTOM_ENCODER_TRAIN_LOOPS) -> None:
        super().__init__()
        self.event_threshold = float(event_threshold)
        self.mixer_num_loops = int(mixer_num_loops)
        self.byte_embedding = ByteEmbedding(dim=16, ngram_vocab=65_536, ngram_range=NGRAM_RANGE)
        self.pre_patch = PrePatchSmoother(dim=16, kernel_size=5)
        self.patch_extractor = PatchExtractor(window=24, stride=8)
        self.patch_proj = nn.Linear(16, 64)
        self.patch_mixer = LoopedMixer(dim=64, hidden_dim=128, num_loops=mixer_num_loops)
        self.patch_gate = PatchGate(dim=64)
        self.probe_scorer = ProbeScorer(state_dim=64, num_probes=12)

    def forward(
        self,
        byte_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        event_threshold: float | None = None,
        mixer_loops: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if byte_ids.dtype != torch.long:
            byte_ids = byte_ids.to(torch.long)
        if lengths is None:
            lengths = torch.full((byte_ids.size(0),), byte_ids.size(1), device=byte_ids.device, dtype=torch.long)
        else:
            lengths = lengths.to(device=byte_ids.device, dtype=torch.long)
        lengths = lengths.clamp(min=0, max=byte_ids.size(1))
        byte_mask = torch.arange(byte_ids.size(1), device=byte_ids.device).unsqueeze(0) < lengths.unsqueeze(1)
        embeddings = self.byte_embedding(byte_ids)
        embeddings = embeddings * byte_mask.unsqueeze(-1).to(dtype=embeddings.dtype)
        smoothed = self.pre_patch(embeddings, byte_mask=byte_mask)
        patch_outputs = self.patch_extractor(smoothed, lengths=lengths)
        patch_projected = self.patch_proj(patch_outputs["patch_inputs"])
        patch_projected = patch_projected * patch_outputs["patch_mask"].unsqueeze(-1).to(dtype=patch_projected.dtype)
        patch_mixed = self.patch_mixer(patch_projected, patch_outputs["patch_mask"], loops=mixer_loops)
        gated = self.patch_gate(patch_mixed, patch_outputs["patch_mask"])
        events = self.probe_scorer(
            gated["patch_states"],
            patch_outputs["patch_mask"],
            self.event_threshold if event_threshold is None else event_threshold,
        )
        return {
            "patch_states": gated["patch_states"],
            "patch_mask": patch_outputs["patch_mask"],
            "event_scores": events.scores,
            "event_mask": events.mask,
            "event_counts": events.counts,
        }


def load_custom_encoder_model(checkpoint_path: Path, *, device: torch.device) -> tuple[CustomByteEncoder, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"custom encoder checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"custom encoder checkpoint payload must be a dict, got {type(payload).__name__}")
    state_dict = payload.get("state_dict") or payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("custom encoder checkpoint missing state_dict/model_state_dict")
    train_loops = int(payload.get("train_loops", CUSTOM_ENCODER_TRAIN_LOOPS))
    model = CustomByteEncoder(event_threshold=CUSTOM_ENCODER_EVENT_THRESHOLD, mixer_num_loops=train_loops).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    metadata = {
        "train_loops": train_loops,
        "train_step": payload.get("train_step"),
        "param_count": _count_parameters(model),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "event_threshold": CUSTOM_ENCODER_EVENT_THRESHOLD,
    }
    return model, metadata


class CustomEncoderBackend:
    model_id = CUSTOM_ENCODER_MODEL_ID
    model_revision = "checkpoint"
    query_seq = CUSTOM_ENCODER_QUERY_SEQ
    doc_seq = CUSTOM_ENCODER_DOC_SEQ
    event_threshold = CUSTOM_ENCODER_EVENT_THRESHOLD
    truncation_strategy = CUSTOM_ENCODER_TRUNCATION_STRATEGY
    score_function = CUSTOM_ENCODER_SCORE_FUNCTION

    def __init__(
        self,
        *,
        checkpoint_path: Path | None = None,
        device: str | torch.device | None = None,
        batch_size: int = 8,
        doc_batch_size: int | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else _checkpoint_path_from_env()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = int(batch_size)
        self.doc_batch_size = int(doc_batch_size or batch_size)
        self.model, self._checkpoint_metadata = load_custom_encoder_model(self.checkpoint_path, device=self.device)
        self.param_count = int(self._checkpoint_metadata["param_count"])
        self.checkpoint_sha256 = str(self._checkpoint_metadata["checkpoint_sha256"])

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "param_count": self.param_count,
            "query_seq": self.query_seq,
            "doc_seq": self.doc_seq,
            "event_threshold": self.event_threshold,
            "truncation_strategy": self.truncation_strategy,
            "score_function": self.score_function,
            "device": str(self.device),
            "train_loops": self._checkpoint_metadata.get("train_loops"),
            "train_step": self._checkpoint_metadata.get("train_step"),
        }

    def _forward_batches(self, texts: Sequence[str], *, max_length: int, batch_size: int) -> list[dict[str, torch.Tensor]]:
        outputs: list[dict[str, torch.Tensor]] = []
        with torch.no_grad():
            for start in range(0, len(texts), max(1, batch_size)):
                batch_texts = texts[start : start + max(1, batch_size)]
                byte_ids, lengths = _encode_text_batch(batch_texts, max_length, self.device)
                batch_outputs = self.model(byte_ids, lengths=lengths, event_threshold=self.event_threshold)
                outputs.append({key: value.detach().cpu() for key, value in batch_outputs.items()})
        return outputs

    def encode_documents(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        batches = self._forward_batches(texts, max_length=self.doc_seq, batch_size=self.doc_batch_size)
        if not batches:
            return torch.empty((0, 0, 64), dtype=torch.float32), torch.empty((0, 0), dtype=torch.bool)
        vectors = []
        masks = []
        for batch in batches:
            vectors.append(_normalize_eval_vectors(batch["patch_states"]))
            masks.append(build_retrieval_mask(batch["patch_mask"], batch["event_scores"], self.event_threshold).to(dtype=torch.bool))
        return torch.cat(vectors, dim=0), torch.cat(masks, dim=0)

    def encode_queries(self, texts: Sequence[str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        query_vectors: list[torch.Tensor] = []
        query_weights: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(texts), max(1, self.batch_size)):
                batch_texts = texts[start : start + max(1, self.batch_size)]
                byte_ids, lengths = _encode_text_batch(batch_texts, self.query_seq, self.device)
                outputs = self.model(byte_ids, lengths=lengths, event_threshold=self.event_threshold)
                retrieval_mask = build_retrieval_mask(outputs["patch_mask"], outputs["event_scores"], self.event_threshold)
                for batch_index in range(outputs["patch_states"].size(0)):
                    selected = outputs["patch_states"][batch_index, retrieval_mask[batch_index]]
                    if selected.numel() == 0:
                        selected = outputs["patch_states"][batch_index, outputs["patch_mask"][batch_index]]
                    if selected.numel() == 0:
                        selected = outputs["patch_states"].new_zeros((1, outputs["patch_states"].size(-1)))
                        weights = outputs["patch_states"].new_ones((1,), dtype=torch.float32)
                    else:
                        selected = _finite_eval_tensor(selected)
                        weights = self.model.probe_scorer.compute_query_patch_weights(selected.unsqueeze(0), force_uniform=False)[0]
                    normalized = _normalize_eval_vectors(selected).detach().cpu()
                    sanitized = _sanitize_query_weights(weights.detach().cpu(), patch_count=int(normalized.size(0)))
                    query_vectors.append(normalized)
                    query_weights.append(sanitized)
        return query_vectors, query_weights


def _document_text(document: CodeDocument) -> str:
    return document.code


def _score_query_against_documents(
    query_vectors: torch.Tensor,
    query_weights: torch.Tensor,
    doc_vectors: torch.Tensor,
    doc_mask: torch.Tensor,
) -> torch.Tensor:
    if query_vectors.numel() == 0:
        return doc_vectors.new_zeros((doc_vectors.size(0),), dtype=torch.float32)
    doc_vectors = _finite_eval_tensor(doc_vectors)
    query_vectors = _finite_eval_tensor(query_vectors).to(device=doc_vectors.device)
    query_weights = _sanitize_query_weights(query_weights.to(device=doc_vectors.device), patch_count=int(query_vectors.size(0)))
    doc_mask = doc_mask.to(device=doc_vectors.device, dtype=torch.bool)
    similarities = torch.einsum("qd,bkd->bqk", query_vectors, doc_vectors)
    similarities = _finite_eval_tensor(similarities)
    similarities = similarities.masked_fill(~doc_mask.unsqueeze(1), EVAL_MASKED_SCORE_FLOOR)
    max_per_patch = similarities.max(dim=-1).values
    return _finite_eval_tensor((query_weights.unsqueeze(0) * max_per_patch).sum(dim=-1))


class CustomEncoderRetriever:
    name = "custom-encoder"
    display_name = "Custom encoder"

    def __init__(
        self,
        *,
        checkpoint_path: Path | None = None,
        encoder_backend: Any | None = None,
        batch_size: int = 8,
        score_chunk_size: int = 512,
        device: str | torch.device | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        if self.checkpoint_path is None and encoder_backend is None:
            self.checkpoint_path = _checkpoint_path_from_env()
        elif self.checkpoint_path is None:
            backend_checkpoint_path = getattr(encoder_backend, "checkpoint_path", None)
            self.checkpoint_path = Path(backend_checkpoint_path) if backend_checkpoint_path is not None else None
        self.batch_size = int(batch_size)
        self.score_chunk_size = int(score_chunk_size)
        self.encoder_backend = encoder_backend or CustomEncoderBackend(
            checkpoint_path=self.checkpoint_path,
            device=device,
            batch_size=batch_size,
        )
        self._documents: tuple[CodeDocument, ...] = ()
        self._doc_vectors: torch.Tensor | None = None
        self._doc_mask: torch.Tensor | None = None
        self._latency_samples_ms: list[float] = []

    def embedding_metadata(self) -> dict[str, Any]:
        if hasattr(self.encoder_backend, "metadata"):
            metadata = dict(self.encoder_backend.metadata())
        else:
            metadata = {
                "model_id": str(getattr(self.encoder_backend, "model_id", CUSTOM_ENCODER_MODEL_ID)),
                "checkpoint_path": str(getattr(self.encoder_backend, "checkpoint_path", self.checkpoint_path or "unknown")),
                "checkpoint_sha256": str(getattr(self.encoder_backend, "checkpoint_sha256", "unknown")),
                "param_count": int(getattr(self.encoder_backend, "param_count", 0)),
                "query_seq": int(getattr(self.encoder_backend, "query_seq", CUSTOM_ENCODER_QUERY_SEQ)),
                "doc_seq": int(getattr(self.encoder_backend, "doc_seq", CUSTOM_ENCODER_DOC_SEQ)),
                "event_threshold": float(getattr(self.encoder_backend, "event_threshold", CUSTOM_ENCODER_EVENT_THRESHOLD)),
                "device": str(getattr(self.encoder_backend, "device", "unknown")),
            }
        metadata.setdefault("truncation_strategy", CUSTOM_ENCODER_TRUNCATION_STRATEGY)
        metadata.setdefault("score_function", CUSTOM_ENCODER_SCORE_FUNCTION)
        return metadata

    def index(self, corpus: Sequence[CodeDocument]) -> None:
        self._documents = tuple(corpus)
        if not self._documents:
            raise ValueError("CustomEncoderRetriever requires a non-empty corpus")
        texts = [_document_text(document) for document in self._documents]
        doc_vectors, doc_mask = self.encoder_backend.encode_documents(texts)
        doc_vectors = _normalize_eval_vectors(torch.as_tensor(doc_vectors, dtype=torch.float32))
        doc_mask = torch.as_tensor(doc_mask, dtype=torch.bool)
        if doc_vectors.size(0) != len(self._documents):
            raise ValueError(f"Encoder returned {doc_vectors.size(0)} document vector rows for {len(self._documents)} documents")
        if doc_mask.shape[:2] != doc_vectors.shape[:2]:
            raise ValueError(f"Document mask shape {tuple(doc_mask.shape)} does not match vectors {tuple(doc_vectors.shape)}")
        self._doc_vectors = doc_vectors.cpu()
        self._doc_mask = doc_mask.cpu()
        self._latency_samples_ms.clear()

    def search(self, query: str, k: int) -> list[SearchResult]:
        if self._doc_vectors is None or self._doc_mask is None:
            raise RuntimeError("CustomEncoderRetriever.search() called before index()")
        if k <= 0:
            return []
        started = time.perf_counter()
        query_vectors_list, query_weights_list = self.encoder_backend.encode_queries([query])
        if len(query_vectors_list) != 1 or len(query_weights_list) != 1:
            raise ValueError("Encoder backend must return exactly one query representation for one query")
        query_vectors = _normalize_eval_vectors(torch.as_tensor(query_vectors_list[0], dtype=torch.float32))
        query_weights = _sanitize_query_weights(torch.as_tensor(query_weights_list[0], dtype=torch.float32), patch_count=int(query_vectors.size(0)))
        score_chunks: list[torch.Tensor] = []
        chunk_size = max(1, self.score_chunk_size)
        for start_index in range(0, self._doc_vectors.size(0), chunk_size):
            score_chunks.append(
                _score_query_against_documents(
                    query_vectors,
                    query_weights,
                    self._doc_vectors[start_index : start_index + chunk_size],
                    self._doc_mask[start_index : start_index + chunk_size],
                )
            )
        scores = torch.cat(score_chunks, dim=0) if score_chunks else torch.empty((0,), dtype=torch.float32)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._latency_samples_ms.append(elapsed_ms)
        top_n = min(k, len(self._documents))
        if top_n == 0:
            return []
        candidate_indices = torch.topk(scores, k=top_n).indices.tolist()
        ranked_indices = sorted(candidate_indices, key=lambda index: (-float(scores[index].item()), self._documents[index].document_id))
        metadata = self.embedding_metadata()
        return [
            SearchResult(
                document_id=self._documents[index].document_id,
                path=self._documents[index].path,
                score=float(scores[index].item()),
                code=self._documents[index].code,
                metadata={**metadata, "ranker": self.name, "score_function": CUSTOM_ENCODER_SCORE_FUNCTION},
            )
            for index in ranked_indices
        ]

    def latency_ms(self) -> dict[str, float]:
        if not self._latency_samples_ms:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        return summarize_latency(self._latency_samples_ms)
