"""Calculator: exact arithmetic for the robot (LLMs get sums wrong). Local, no internet.

Evaluates with an `ast` whitelist (no eval): numbers, + - * / // % **, parentheses, unary +/-,
the functions below and the constants pi and e. "15% of 80" / "15 percent of 80" is rewritten
to (15/100)*80, and "x" / "×" / "÷" / "^" are accepted. Exponents and result sizes are capped.
"""

import ast
import logging
import math
import operator
import re
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
          ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
UNOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
FUNCS = {"sqrt": math.sqrt, "sin": lambda x: math.sin(math.radians(x)), "cos": lambda x: math.cos(math.radians(x)),
         "tan": lambda x: math.tan(math.radians(x)), "log": math.log10, "ln": math.log, "exp": math.exp,
         "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil, "min": min, "max": max}
CONSTS = {"pi": math.pi, "e": math.e}
MAX_EXPONENT = 1000
MAX_ABS = 1e100


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        if abs(node.value) > MAX_ABS:
            raise ValueError("number too large")
        return node.value
    if isinstance(node, ast.Name) and node.id in CONSTS:
        return CONSTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNOPS:
        return UNOPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError("exponent too large")
        result = BINOPS[type(node.op)](left, right)
        if isinstance(result, complex) or abs(result) > MAX_ABS:
            raise ValueError("result too large or not a real number")
        return result
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS
            and not node.keywords and 1 <= len(node.args) <= 2):
        return FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError(f"unsupported: {ast.dump(node)[:40]}")


def normalise(expression: str) -> str:
    x = expression.strip().lower().replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", "")
    x = re.sub(r"(\d)\s*x\s*(\d)", r"\1*\2", x)                                           # 3 x 4
    x = re.sub(r"([\d.]+)\s*(%|percent)\s*of\s*", r"(\1/100)*", x)                        # 15% of 80
    x = re.sub(r"\b(times)\b", "*", x)
    x = re.sub(r"\b(divided by|over)\b", "/", x)
    x = re.sub(r"\bplus\b", "+", x)
    x = re.sub(r"\bminus\b", "-", x)
    x = re.sub(r"\bsquared\b", "**2", x)
    return x


def calculate(expression: str) -> float:
    return _eval(ast.parse(normalise(expression), mode="eval"))


def spoken(value: float) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}" if abs(value) < 1e21 else f"{value:.4g}"
    return f"{value:,.6g}" if abs(value) >= 1e-4 else f"{value:.3g}"


class Calculate(Tool):
    """Evaluate an arithmetic expression exactly."""

    name = "calculate"
    description = (
        "Do arithmetic exactly: + - * / ** %, percentages ('15% of 80'), sqrt, log, ln, sin/cos/tan (degrees), "
        "round, min, max, pi. Always use this instead of doing maths yourself; say the 'spoken' result."
    )
    parameters_schema = {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "e.g. '17*23', '15% of 80', 'sqrt(2)*10'."}},
        "required": ["expression"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Evaluate and return exact and spoken results."""
        expression = str(kwargs.get("expression") or "")
        logger.info("Tool call: calculate %r", expression)
        try:
            value = calculate(expression)
        except ZeroDivisionError:
            return {"error": "Division by zero."}
        except (ValueError, SyntaxError, TypeError, OverflowError, RecursionError) as e:
            return {"error": f"Couldn't calculate {expression!r} ({e})."}
        try:
            said = spoken(value)
        except (OverflowError, ValueError):
            return {"error": "Result too large."}
        return {"expression": expression, "result": value, "spoken": said}
