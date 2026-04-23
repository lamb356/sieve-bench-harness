from __future__ import annotations

from functools import lru_cache

from tree_sitter import Node
from tree_sitter_language_pack import get_parser


_LANGUAGE_ALIASES = {
    "c++": "cpp",
    "cpp": "cpp",
    "go": "go",
    "java": "java",
    "python": "python",
    "rust": "rust",
    "typescript": "typescript",
    "ts": "typescript",
}


@lru_cache(maxsize=None)
def _parser_for(language: str):
    normalized_language = _LANGUAGE_ALIASES.get(language.lower())
    if normalized_language is None:
        raise ValueError(f"Unsupported language for normalization: {language}")
    return get_parser(normalized_language)


def _is_comment(node: Node) -> bool:
    return node.type == "comment"


def _is_python_docstring(node: Node, *, language: str) -> bool:
    if language != "python":
        return False
    if node.type != "expression_statement":
        return False
    if node.named_child_count != 1:
        return False
    child = node.named_children[0]
    return child.type == "string"


def _is_python_definition_name(node: Node, *, language: str) -> bool:
    if language != "python" or node.type != "identifier" or node.parent is None:
        return False
    if node.parent.type not in {"function_definition", "class_definition"}:
        return False
    name_node = node.parent.child_by_field_name("name")
    if name_node is None:
        return False
    return name_node.start_byte == node.start_byte and name_node.end_byte == node.end_byte


def _collect_leaf_tokens(node: Node, source_bytes: bytes, *, language: str, sink: list[str]) -> None:
    if _is_comment(node) or _is_python_docstring(node, language=language):
        return
    if node.child_count == 0:
        snippet = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore").strip()
        if snippet:
            sink.append(snippet)
        return
    for child in node.children:
        _collect_leaf_tokens(child, source_bytes, language=language, sink=sink)


def _collect_search_leaf_tokens(node: Node, source_bytes: bytes, *, language: str, sink: list[str]) -> None:
    if (
        _is_comment(node)
        or _is_python_docstring(node, language=language)
        or _is_python_definition_name(node, language=language)
    ):
        return
    if language == "python" and node.type in {"function_definition", "class_definition"}:
        body = node.child_by_field_name("body")
        if body is not None:
            _collect_search_leaf_tokens(body, source_bytes, language=language, sink=sink)
            return
    if node.child_count == 0:
        snippet = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore").strip()
        if snippet:
            sink.append(snippet)
        return
    for child in node.children:
        _collect_search_leaf_tokens(child, source_bytes, language=language, sink=sink)


def normalize_code(code: str, *, language: str, separator: str = "") -> str:
    if not isinstance(code, str) or not code.strip():
        return ""
    parser = _parser_for(language)
    source_bytes = code.encode("utf-8")
    tree = parser.parse(source_bytes)
    tokens: list[str] = []
    _collect_leaf_tokens(tree.root_node, source_bytes, language=language.lower(), sink=tokens)
    return separator.join(tokens)


def normalize_for_search(code: str, *, language: str) -> str:
    if not isinstance(code, str) or not code.strip():
        return ""
    parser = _parser_for(language)
    source_bytes = code.encode("utf-8")
    tree = parser.parse(source_bytes)
    tokens: list[str] = []
    _collect_search_leaf_tokens(tree.root_node, source_bytes, language=language.lower(), sink=tokens)
    return " ".join(tokens)
