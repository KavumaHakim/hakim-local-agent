"""Exact arithmetic via a restricted expression evaluator.

There is no `eval()` here. The expression is parsed with `ast` and walked node
by node against a whitelist, so imports, function definitions, attribute
access, subscripting, comprehensions, assignments and every name outside the
constants below are rejected before anything is evaluated. That leaves no
route to open(), os, subprocess or the network.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable

from tools.base import Tool, ToolError

# Cheap guards against expressions that are quick to type and slow to evaluate.
MAX_EXPRESSION_LENGTH = 500
MAX_EXPONENT = 1000
MAX_FACTORIAL = 500

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

_FUNCTIONS: dict[str, Callable[..., Any]] = {
    # roots and powers
    "sqrt": math.sqrt,
    "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
    "pow": math.pow,
    "exp": math.exp,
    # logarithms
    "log": math.log,  # log(x) or log(x, base)
    "log2": math.log2,
    "log10": math.log10,
    # trigonometry
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "degrees": math.degrees,
    "radians": math.radians,
    "hypot": math.hypot,
    # rounding and sign
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
    # aggregates and integer maths
    "min": min,
    "max": max,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "factorial": math.factorial,
    "comb": math.comb,
    "perm": math.perm,
    "isqrt": math.isqrt,
}


class CalculationError(ToolError):
    """The expression was rejected or could not be evaluated."""


def evaluate(expression: str) -> float | int:
    """Evaluate a single mathematical expression safely."""
    if not isinstance(expression, str) or not expression.strip():
        raise CalculationError("Expression must be a non-empty string.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculationError(
            f"Expression is too long ({len(expression)} characters, "
            f"limit {MAX_EXPRESSION_LENGTH})."
        )

    # Models write powers the way people do. Ministral produced
    # "sqrt(144) + 25^2" where Qwen wrote "25**2", and rejecting that costs a
    # whole round-trip on a machine where a round-trip is tens of seconds.
    #
    # This is a text rewrite, not an AST one, and that matters: in Python `^`
    # binds *looser* than `+`, so treating BitXor as a power operator after
    # parsing would read "sqrt(144) + 25^2" as (12 + 25)**2 = 1369 instead of
    # 637 - silently wrong, which is worse than refusing. Rewriting before the
    # parse gives `**` its correct tighter precedence.
    #
    # Nothing is lost: `^` as bitwise xor was rejected outright before, and
    # strings are not allowed here, so there is no text for this to corrupt.
    expression = expression.replace("^", "**")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculationError(f"Invalid expression: {exc.msg}.") from None

    try:
        result = _eval(tree.body)
    except CalculationError:
        raise
    except ZeroDivisionError:
        raise CalculationError("Division by zero.") from None
    except (ValueError, OverflowError) as exc:
        raise CalculationError(f"Maths error: {exc}.") from None
    except TypeError as exc:
        raise CalculationError(f"Invalid operand: {exc}.") from None

    if not isinstance(result, (int, float)):
        raise CalculationError(
            f"Expression produced a {type(result).__name__}, not a number."
        )
    return result


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculationError("Only numeric literals are allowed.")
        return node.value

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        if node.id in _FUNCTIONS:
            raise CalculationError(f"{node.id!r} must be called, e.g. {node.id}(x).")
        raise CalculationError(
            f"Unknown name {node.id!r}. Allowed constants: "
            f"{', '.join(sorted(_CONSTANTS))}."
        )

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise CalculationError(f"Operator {type(node.op).__name__} is not allowed.")
        left = _eval(node.left)
        right = _eval(node.right)
        if isinstance(node.op, ast.Pow):
            _check_exponent(right)
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise CalculationError(f"Operator {type(node.op).__name__} is not allowed.")
        return op(_eval(node.operand))

    if isinstance(node, ast.Call):
        return _eval_call(node)

    raise CalculationError(
        f"{type(node).__name__} is not allowed. Use arithmetic, the listed "
        f"functions, and numeric constants only."
    )


def _eval_call(node: ast.Call) -> Any:
    # Only a bare whitelisted name may be called. This is what rules out
    # attribute access such as ().__class__.__bases__[0].
    if not isinstance(node.func, ast.Name):
        raise CalculationError("Only direct calls to the listed functions are allowed.")
    if node.keywords:
        raise CalculationError("Keyword arguments are not supported.")

    func = _FUNCTIONS.get(node.func.id)
    if func is None:
        raise CalculationError(
            f"Unknown function {node.func.id!r}. Available: "
            f"{', '.join(sorted(_FUNCTIONS))}."
        )

    args = [_eval(arg) for arg in node.args]

    if node.func.id == "factorial":
        if not args or not isinstance(args[0], int) or args[0] > MAX_FACTORIAL:
            raise CalculationError(
                f"factorial() needs a whole number of at most {MAX_FACTORIAL}."
            )

    try:
        return func(*args)
    except TypeError as exc:
        raise CalculationError(f"{node.func.id}() got bad arguments: {exc}.") from None


def _check_exponent(exponent: Any) -> None:
    if isinstance(exponent, (int, float)) and abs(exponent) > MAX_EXPONENT:
        raise CalculationError(
            f"Exponent {exponent} exceeds the limit of {MAX_EXPONENT}."
        )


def format_number(value: float | int) -> str:
    """Render a result without noisy floating point tails."""
    if isinstance(value, int):
        return str(value)
    if math.isnan(value) or math.isinf(value):
        return str(value)
    if value.is_integer() and abs(value) < 1e16:
        return str(int(value))
    return f"{value:.12g}"


def calculate(expression: str) -> dict[str, Any]:
    """Tool entry point. Returns a structured result."""
    value = evaluate(expression)
    return {
        "success": True,
        "result": value,
        # The model reads text, so give it a clean rendering too.
        "formatted": format_number(value),
    }


CALCULATOR_TOOL = Tool(
    name="calculate",
    category="calculator",
    # Kept compact: prompt tokens are expensive on CPU, but the function list
    # stays because a rejected call costs a whole extra round-trip.
    description=(
        "Evaluate a maths expression exactly. Use instead of calculating "
        "yourself. Operators + - * / // % ** (^ also works for powers); "
        "constants pi, e, tau; functions "
        "sqrt cbrt exp log log2 log10 sin cos tan asin acos atan atan2 sinh "
        "cosh tanh degrees radians hypot abs round floor ceil trunc min max "
        "gcd lcm factorial comb perm isqrt. "
        "Examples: 'sqrt(144)', '2**10', 'log(100, 10)', 'sin(pi/2)', '17*43'. "
        "Calculator only: no variables, imports or statements."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate.",
            }
        },
        "required": ["expression"],
    },
    run=calculate,
)
