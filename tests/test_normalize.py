from bench.contamination.normalize import normalize_code


PYTHON_SOURCE = '''
# leading comment

def example(value: int) -> int:
    """Docstring should be removed."""
    total = value + 1  # inline comment
    return total
'''


def test_normalize_code_strips_python_comments_docstrings_and_whitespace() -> None:
    normalized = normalize_code(PYTHON_SOURCE, language="python")

    assert "Docstring should be removed" not in normalized
    assert "leading comment" not in normalized
    assert "inline comment" not in normalized
    assert normalized == "defexample(value:int)->int:total=value+1returntotal"
