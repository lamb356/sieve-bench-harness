from __future__ import annotations

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
