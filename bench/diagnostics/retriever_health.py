from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from bench.constants import (
    BENCH_MODEL_CACHE_DIR,
    CODEBERT_MODEL_ID,
    CODEBERT_MODEL_REVISION,
    UNIXCODER_MODEL_ID,
    UNIXCODER_MODEL_REVISION,
)

QUERY = "sort a list of integers"
TOKENIZER_SAMPLE = "def sort_integers(values): return sorted(values)"

DOCS = [
    {
        "name": "relevant_sort",
        "relevant": True,
        "raw_code": "def sort_integers(values: list[int]) -> list[int]:\n    return sorted(values)\n",
        "normalized_index_text": "sort integers values list int list int return sorted values",
    },
    {
        "name": "irrelevant_http",
        "relevant": False,
        "raw_code": "def parse_http_request(raw_request: str) -> dict:\n    return raw_request.headers\n",
        "normalized_index_text": "parse http request raw request str dict return raw request headers",
    },
    {
        "name": "irrelevant_gradient",
        "relevant": False,
        "raw_code": "def gradient_norm(tensor):\n    return tensor.grad.norm()\n",
        "normalized_index_text": "gradient norm tensor return tensor grad norm",
    },
]


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model_id: str
    revision: str
    force_decoder_config: bool = False


MODEL_SPECS = [
    ModelSpec("CodeBERT", CODEBERT_MODEL_ID, CODEBERT_MODEL_REVISION, False),
    ModelSpec("UniXcoder/current-auto-config", UNIXCODER_MODEL_ID, UNIXCODER_MODEL_REVISION, False),
    ModelSpec("UniXcoder/official-decoder-config", UNIXCODER_MODEL_ID, UNIXCODER_MODEL_REVISION, True),
]


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    token_mask = attention_mask
    if token_mask.ndim == 3:
        # Official UniXcoder-style 3D masks use outer products. The row mask recovers
        # which source tokens are real. Diagonal is 1 for non-padding tokens.
        token_mask = torch.diagonal(token_mask, dim1=-2, dim2=-1)
    token_mask = token_mask.to(hidden.dtype)
    mean = (hidden * token_mask.unsqueeze(-1)).sum(dim=1) / token_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    cls = hidden[:, 0, :]
    return {"cls": cls, "mean": mean}


def _cosine(query: torch.Tensor, docs: torch.Tensor) -> list[float]:
    query = torch.nn.functional.normalize(query, p=2, dim=-1)
    docs = torch.nn.functional.normalize(docs, p=2, dim=-1)
    return (docs @ query.unsqueeze(-1)).squeeze(-1).detach().cpu().tolist()


def _manual_unixcoder_encoder_only(tokenizer: Any, texts: Iterable[str], *, max_length: int) -> dict[str, torch.Tensor]:
    rows: list[list[int]] = []
    mode = "<encoder-only>"
    for text in texts:
        tokens = tokenizer.tokenize(text)
        tokens = tokens[: max_length - 4]
        tokens = [tokenizer.cls_token, mode, tokenizer.sep_token] + tokens + [tokenizer.sep_token]
        rows.append(tokenizer.convert_tokens_to_ids(tokens))
    pad_id = tokenizer.pad_token_id
    width = max(len(row) for row in rows)
    input_ids = [row + [pad_id] * (width - len(row)) for row in rows]
    attention_mask_2d = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_2d, dtype=torch.long),
    }


def _encode_plain(tokenizer: Any, texts: list[str], *, max_length: int) -> dict[str, torch.Tensor]:
    return tokenizer(texts, add_special_tokens=True, truncation=True, max_length=max_length, padding=True, return_tensors="pt")


def _batch_for_mode(tokenizer: Any, texts: list[str], *, input_mode: str, max_length: int) -> dict[str, torch.Tensor]:
    if input_mode == "plain-auto-specials":
        return _encode_plain(tokenizer, texts, max_length=max_length)
    if input_mode == "unixcoder-encoder-only-prefix":
        return _manual_unixcoder_encoder_only(tokenizer, texts, max_length=max_length)
    raise ValueError(f"unknown input_mode={input_mode}")


def _token_debug(tokenizer: Any, text: str, *, input_mode: str, max_length: int) -> dict[str, Any]:
    batch = _batch_for_mode(tokenizer, [text], input_mode=input_mode, max_length=max_length)
    ids = batch["input_ids"][0].tolist()
    mask = batch["attention_mask"][0]
    if mask.ndim > 1:
        mask_count = int(torch.diagonal(mask, dim1=-2, dim2=-1).sum().item())
    else:
        mask_count = int(mask.sum().item())
    trimmed = ids[:mask_count]
    return {
        "input_mode": input_mode,
        "token_count": mask_count,
        "ids": trimmed,
        "tokens": tokenizer.convert_ids_to_tokens(trimmed),
    }


def _run_model(spec: ModelSpec, *, device: torch.device, max_length: int) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision, cache_dir=str(BENCH_MODEL_CACHE_DIR))
    config = AutoConfig.from_pretrained(spec.model_id, revision=spec.revision, cache_dir=str(BENCH_MODEL_CACHE_DIR))
    original_is_decoder = bool(getattr(config, "is_decoder", False))
    if spec.force_decoder_config:
        config.is_decoder = True
    model = AutoModel.from_pretrained(spec.model_id, revision=spec.revision, cache_dir=str(BENCH_MODEL_CACHE_DIR), config=config)
    model.to(device)
    model.eval()

    input_modes = ["plain-auto-specials"]
    if "unixcoder" in spec.model_id.lower():
        input_modes.append("unixcoder-encoder-only-prefix")

    model_results: list[dict[str, Any]] = []
    for input_mode in input_modes:
        if input_mode == "unixcoder-encoder-only-prefix":
            encoder_only_id = tokenizer.convert_tokens_to_ids("<encoder-only>")
            if encoder_only_id == tokenizer.unk_token_id:
                mode_note = "<encoder-only> maps to unk_token_id"
            else:
                mode_note = f"<encoder-only> id={encoder_only_id}"
        else:
            mode_note = "default tokenizer special tokens"

        tokenizer_debug = {
            "sample": TOKENIZER_SAMPLE,
            "plain": _token_debug(tokenizer, TOKENIZER_SAMPLE, input_mode="plain-auto-specials", max_length=max_length),
        }
        if "unixcoder" in spec.model_id.lower():
            tokenizer_debug["unixcoder_encoder_only"] = _token_debug(
                tokenizer, TOKENIZER_SAMPLE, input_mode="unixcoder-encoder-only-prefix", max_length=max_length
            )

        query_texts = [QUERY]
        for surface in ("normalized_index_text", "raw_code"):
            doc_texts = [str(doc[surface]) for doc in DOCS]
            texts = query_texts + doc_texts
            batch = _batch_for_mode(tokenizer, texts, input_mode=input_mode, max_length=max_length)
            # Official UniXcoder wrapper feeds a 3D non-causal mask for encoder-only embeddings.
            if spec.force_decoder_config and input_mode == "unixcoder-encoder-only-prefix":
                mask_2d = batch["attention_mask"]
                batch["attention_mask"] = mask_2d.unsqueeze(1) * mask_2d.unsqueeze(2)
            batch = _to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)
                pooled = _pool(outputs.last_hidden_state, batch["attention_mask"])

            surface_result: dict[str, Any] = {
                "surface": surface,
                "query": QUERY,
                "documents": [{"name": doc["name"], "relevant": doc["relevant"], "text": doc[surface]} for doc in DOCS],
                "token_counts": [],
                "pooling": {},
            }
            for index, text in enumerate(texts):
                dbg = _token_debug(tokenizer, text, input_mode=input_mode, max_length=max_length)
                surface_result["token_counts"].append(
                    {"kind": "query" if index == 0 else "document", "name": "query" if index == 0 else DOCS[index - 1]["name"], "token_count": dbg["token_count"]}
                )

            for pooling_name, vectors in pooled.items():
                query_vec = vectors[0]
                doc_vecs = vectors[1:]
                similarities = _cosine(query_vec, doc_vecs)
                norms = [float(vec.norm().detach().cpu()) for vec in vectors]
                relevant_similarity = similarities[0]
                max_irrelevant_similarity = max(similarities[1:])
                surface_result["pooling"][pooling_name] = {
                    "query_norm": norms[0],
                    "doc_norms": {DOCS[i]["name"]: norms[i + 1] for i in range(len(DOCS))},
                    "cosine_similarities": {DOCS[i]["name"]: similarities[i] for i in range(len(DOCS))},
                    "relevant_wins": bool(relevant_similarity > max_irrelevant_similarity),
                    "relevant_margin_vs_best_irrelevant": float(relevant_similarity - max_irrelevant_similarity),
                }
            model_results.append(
                {
                    "model_label": spec.label,
                    "model_id": spec.model_id,
                    "revision": spec.revision,
                    "model_class": model.__class__.__name__,
                    "config_architectures": getattr(config, "architectures", None),
                    "config_is_decoder_original": original_is_decoder,
                    "config_is_decoder_effective": bool(getattr(config, "is_decoder", False)),
                    "special_tokens_map": tokenizer.special_tokens_map,
                    "model_max_length": tokenizer.model_max_length,
                    "input_mode": input_mode,
                    "mode_note": mode_note,
                    "tokenizer_debug": tokenizer_debug,
                    "surface_result": surface_result,
                }
            )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return model_results


def _print_human(results: list[dict[str, Any]]) -> None:
    print("# retriever_health diagnostic")
    print(f"query: {QUERY!r}")
    print(f"tokenizer_sample: {TOKENIZER_SAMPLE!r}")
    print("")
    for result in results:
        surface = result["surface_result"]
        print(f"## {result['model_label']} | {result['input_mode']} | surface={surface['surface']}")
        print(
            f"model={result['model_id']} revision={result['revision']} class={result['model_class']} "
            f"is_decoder={result['config_is_decoder_effective']} note={result['mode_note']}"
        )
        print("token_counts:")
        for item in surface["token_counts"]:
            print(f"  - {item['kind']}:{item['name']} tokens={item['token_count']}")
        for pooling_name, metrics in surface["pooling"].items():
            print(f"pooling={pooling_name} query_norm={metrics['query_norm']:.6f}")
            for doc_name, sim in metrics["cosine_similarities"].items():
                marker = " relevant" if doc_name == "relevant_sort" else ""
                print(f"  cosine({doc_name})={sim:.6f}{marker}")
            print(
                f"  relevant_wins={metrics['relevant_wins']} "
                f"margin={metrics['relevant_margin_vs_best_irrelevant']:.6f}"
            )
        plain_tokens = result["tokenizer_debug"]["plain"]["tokens"]
        print(f"plain_tokens={plain_tokens}")
        if "unixcoder_encoder_only" in result["tokenizer_debug"]:
            print(f"unixcoder_encoder_only_tokens={result['tokenizer_debug']['unixcoder_encoder_only']['tokens']}")
        print("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose CodeBERT/UniXcoder retrieval embedding health.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("bench-results/diagnostics/retriever-health.json"))
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    results: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        results.extend(_run_model(spec, device=device, max_length=args.max_length))

    payload = {
        "query": QUERY,
        "tokenizer_sample": TOKENIZER_SAMPLE,
        "device": str(device),
        "max_length": args.max_length,
        "results": results,
        "interpretation_rule": "A pooling/mode/surface combination is healthy on this toy probe only if relevant_wins is true.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _print_human(results)
    print(f"json_output={args.output}")


if __name__ == "__main__":
    main()
