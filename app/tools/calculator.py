"""Arithmetic evaluation over a restricted AST.

This is **not** a code sandbox and is not presented as one. A real code-execution
tool needs a container with seccomp/gVisor, egress rules and cgroup limits — a
security project in its own right — and a `subprocess` with a timeout is not a
substitute. What this is instead: a genuinely safe evaluator for arithmetic, honest
about its scope.

Safety comes from an allowlist, not a denylist. The parsed tree is walked and any
node type not explicitly permitted is rejected, so there is no `eval`, no name
lookup, no attribute access, no calls, no subscripting, and no comprehensions —
which removes the usual escapes (`().__class__.__bases__`, `__import__`, and so on)
by construction rather than by pattern-matching for them.
"""

from __future__ import annotations

import ast
import math
from typing import ClassVar, cast

from pydantic import Field

from app.domain.errors import ErrorCode, TerminalError
from app.tools.base import BaseTool, EffectPolicy, ToolArgs, ToolContext, ToolResult

#: Node types the evaluator will walk. Everything else is refused.
_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)

#: Guards against a single expression burning the worker: 2**2**64 is a valid parse
#: tree that never returns. Bounds are checked before the operation, not after.
_MAX_EXPONENT = 128
_MAX_MAGNITUDE = 10**100


class CalculatorArgs(ToolArgs):
    expression: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "Arithmetic expression using numbers and + - * / // % ** and parentheses. "
            "Example: (12.5 * 3) + 2 ** 8. No variables or function calls."
        ),
    )


def _evaluate(node: ast.AST) -> float | int:
    if not isinstance(node, _ALLOWED_NODES):
        raise TerminalError(
            f"expression contains unsupported syntax: {type(node).__name__}",
            code=ErrorCode.INVALID_INPUT,
        )

    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise TerminalError("only numeric literals are allowed", code=ErrorCode.INVALID_INPUT)
        return node.value

    if isinstance(node, ast.UnaryOp):
        operand = _evaluate(node.operand)
        return -operand if isinstance(node.op, ast.USub) else +operand

    if isinstance(node, ast.BinOp):
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_EXPONENT:
                raise TerminalError(
                    f"exponent too large (max {_MAX_EXPONENT})", code=ErrorCode.INVALID_INPUT
                )
            if abs(left) > _MAX_MAGNITUDE:
                raise TerminalError("base too large", code=ErrorCode.INVALID_INPUT)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise TerminalError("division by zero", code=ErrorCode.INVALID_INPUT)

        result: float | int = _apply(node.op, left, right)
        if isinstance(result, float) and (math.isinf(result) or math.isnan(result)):
            raise TerminalError("result is not a finite number", code=ErrorCode.INVALID_INPUT)
        return result

    raise TerminalError(f"unsupported node {type(node).__name__}", code=ErrorCode.INVALID_INPUT)


def _apply(op: ast.AST, left: float | int, right: float | int) -> float | int:
    match op:
        case ast.Add():
            return left + right
        case ast.Sub():
            return left - right
        case ast.Mult():
            return left * right
        case ast.Div():
            return left / right
        case ast.FloorDiv():
            return left // right
        case ast.Mod():
            return left % right
        case ast.Pow():
            return cast("float | int", left**right)
    raise TerminalError(f"unsupported operator {type(op).__name__}", code=ErrorCode.INVALID_INPUT)


def evaluate(expression: str) -> float | int:
    """Parse and evaluate. Raises TerminalError on anything unsupported."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise TerminalError(
            f"could not parse expression: {exc.msg}", code=ErrorCode.INVALID_INPUT
        ) from exc
    return _evaluate(tree)


class CalculatorTool(BaseTool):
    name: ClassVar[str] = "calculator"
    description: ClassVar[str] = (
        "Evaluate an arithmetic expression exactly. Use this instead of doing "
        "arithmetic yourself whenever precision matters."
    )
    args_model: ClassVar[type[ToolArgs]] = CalculatorArgs
    # Pure computation: replaying it cannot change anything.
    effect_policy: ClassVar[EffectPolicy] = EffectPolicy.SAFE_TO_REPLAY
    requires_approval: ClassVar[bool] = False
    timeout_seconds: ClassVar[float] = 5.0

    async def execute(self, ctx: ToolContext, args: CalculatorArgs) -> ToolResult:
        try:
            value = evaluate(args.expression)
        except TerminalError as exc:
            # Returned to the model as an errored tool_result rather than raised: a
            # bad expression is something the model can see and correct on its next
            # turn, which is cheaper and more useful than failing the run.
            return ToolResult(
                content=f"Error: {exc.message}",
                is_error=True,
                data={"expression": args.expression, "error": exc.message},
            )
        return ToolResult(
            content=str(value),
            data={"expression": args.expression, "result": value},
        )


__all__ = ["CalculatorArgs", "CalculatorTool", "evaluate"]
