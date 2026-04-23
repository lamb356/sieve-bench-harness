from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

import orjson
from jinja2 import Environment, FileSystemLoader, select_autoescape


def _environment(template_root: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(template_root)),
        autoescape=select_autoescape(enabled_extensions=("html",)),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def write_results_json(payload: dict[str, Any], *, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n")


def write_hero_table(payload: dict[str, Any], *, template_root: Path, output_path: Path) -> None:
    summary = payload["summary"]
    environment = _environment(template_root)
    template = environment.get_template("hero_table.md.j2")
    rendered = template.render(
        source=summary["source"],
        language=summary["language"],
        retriever=summary["retriever"],
        query_count=summary["query_count"],
        contamination_rejected_count=summary["contamination_rejected_count"],
        recall_at_5=f"{summary['recall@5']:.3f}",
        contamination_flag="yes" if summary["contamination_flag"] else "no",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")


def _format_metric(value: Any) -> str:
    return f"{float(value):.3f}"


def _format_latency(value: Any) -> str:
    return f"{float(value):.2f} ms"


def _format_throughput(value: Any) -> str:
    return f"{float(value):.2f} q/s"


def render_phase_b_hero_table(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        f"# Phase B Python benchmark — {summary['source']} / {summary['language']}",
        "",
        f"Queries: {summary['query_count']}  ",
        f"Corpus documents: {summary['corpus_document_count']}  ",
        f"Contamination rejected: {summary['contamination_rejected_count']}",
        "",
        "| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | p50 latency | p95 latency | Throughput |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["retriever_summaries"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["display_name"]),
                    _format_metric(row["recall@1"]),
                    _format_metric(row["recall@5"]),
                    _format_metric(row["recall@10"]),
                    _format_metric(row["mrr@10"]),
                    _format_metric(row["ndcg@10"]),
                    _format_latency(row["p50"]),
                    _format_latency(row["p95"]),
                    _format_throughput(row["throughput_qps"]),
                ]
            )
            + " |"
        )

    findings = summary.get("findings") or []
    if findings:
        lines.extend(["", "## Findings", ""])
        lines.extend(f"- {finding}" for finding in findings)
    else:
        lines.extend(["", "## Findings", "", "- None."])

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Phase B uses normalized `document.index_text` for all retrievers to keep the benchmark surface aligned with Phase A.",
            "- CodeBERT and UniXcoder use pinned HuggingFace revisions, 512-token max context, document head+tail truncation, query head-only truncation, mean pooling, and cosine similarity.",
            "- The SIEVE row is a deterministic query-hash-seeded random stub, not the real SIEVE integration.",
        ]
    )
    return "\n".join(lines) + "\n"


def _json_cell(value: Any) -> Any:
    if isinstance(value, (str, int, float)) or value is None:
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(value, sort_keys=True)


def write_benchmark_csv(rows: list[dict[str, Any]], *, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_cell(row.get(key)) for key in fieldnames})


def write_interactive_html(payload: dict[str, Any], *, output_path: Path) -> None:
    table = render_phase_b_hero_table(payload)
    escaped_json = html.escape(json.dumps(payload, indent=2, sort_keys=True))
    escaped_table = html.escape(table)
    html_body = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Phase B Python benchmark</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.45; }}
    pre {{ background: #111827; color: #f9fafb; padding: 1rem; overflow-x: auto; border-radius: 0.5rem; }}
  </style>
</head>
<body>
  <h1>Phase B Python benchmark</h1>
  <h2>Markdown table</h2>
  <pre>{escaped_table}</pre>
  <h2>Raw JSON</h2>
  <pre>{escaped_json}</pre>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_body, encoding="utf-8")


def write_phase_b_reports(payload: dict[str, Any], *, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_json = output_dir / "results.json"
    benchmark_table = output_dir / "benchmark-table.md"
    benchmark_csv = output_dir / "benchmark-full.csv"
    interactive_html = output_dir / "interactive.html"

    write_results_json(payload, output_path=results_json)
    benchmark_table.write_text(render_phase_b_hero_table(payload), encoding="utf-8")
    write_benchmark_csv(payload.get("rows", []), output_path=benchmark_csv)
    write_interactive_html(payload, output_path=interactive_html)
    return {
        "results_json": results_json,
        "benchmark_table": benchmark_table,
        "benchmark_csv": benchmark_csv,
        "interactive_html": interactive_html,
    }
