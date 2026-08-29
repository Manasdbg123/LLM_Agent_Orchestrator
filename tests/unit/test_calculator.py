"""The calculator's safety claim is an allowlist, so these test what it refuses."""

from __future__ import annotations

import pytest

from app.domain.errors import TerminalError
from app.tools.calculator import evaluate


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 2", 4),
        ("(12.5 * 3) + 2 ** 8", 293.5),
        ("-7 + 3", -4),
        ("10 / 4", 2.5),
        ("10 // 4", 2),
        ("10 % 4", 2),
        ("2 ** 10", 1024),
    ],
)
def test_evaluates_arithmetic(expression: str, expected: float) -> None:
    assert evaluate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        # The classic sandbox escapes. None of these are pattern-matched for; they
        # fail because the node types simply are not in the allowlist.
        "().__class__.__bases__",
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "[x for x in range(10)]",
        "print(1)",
        "os.getcwd()",
        "lambda: 1",
        "x + 1",
        "{'a': 1}['a']",
        "'abc' * 3",
        "1 if True else 2",
        "(1).__class__",
    ],
)
def test_refuses_anything_that_is_not_arithmetic(expression: str) -> None:
    with pytest.raises(TerminalError):
        evaluate(expression)


def test_refuses_division_by_zero() -> None:
    for expression in ("1 / 0", "1 // 0", "1 % 0"):
        with pytest.raises(TerminalError, match="division by zero"):
            evaluate(expression)


def test_bounds_exponent_before_evaluating_it() -> None:
    """2 ** 2 ** 64 parses fine and would never return.

    The bound has to be checked *before* the operation, not after, which is why this
    test asserts on a raise rather than on a slow success.
    """
    with pytest.raises(TerminalError, match="exponent too large"):
        evaluate("2 ** 99999")


def test_rejects_non_finite_results() -> None:
    with pytest.raises(TerminalError, match="finite"):
        evaluate("1e308 * 10")


def test_booleans_are_not_numbers() -> None:
    # bool is an int subclass in Python; without an explicit check `True + True`
    # would quietly evaluate to 2.
    with pytest.raises(TerminalError, match="numeric literals"):
        evaluate("True + True")


def test_syntax_errors_are_terminal_not_retryable() -> None:
    # Retrying a malformed expression produces the same malformed expression.
    with pytest.raises(TerminalError, match="could not parse"):
        evaluate("2 +")
