"""Arithmetic problems: generation, few-shot prompts and step grading (pure Python, no model needed).

The model sees a few worked examples, then a target expression; it continues with one ``= ...`` line per
step, each line reducing one operation, and a blank line to finish (``gemma.generate`` stops there and
``gemma.step_final_positions`` finds the ``=`` lines). Each ``=`` line is the whole reduced expression, so a
step is graded by evaluating it:

- global: the line's value equals the problem's true value;
- local:  the line's value equals the previous line's value (the problem itself for the first step).

The two agree up to and including the first error, so only those steps get a label (``1 = incorrect``,
INTERFACE.md); later steps are kept for analysis with ``label=None`` (CRV's stated truncation policy).
"""
from __future__ import annotations

import ast
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

PREC = {"+": 1, "-": 1, "*": 2}
INSTRUCTION = "Simplify step by step, one operation per line."
SHOT_EXPRESSIONS = ("(3 + 4) * 2", "12 - 5 * 2 + 1")


@dataclass(frozen=True)
class Problem:
    problem_id: str
    expression: str
    value: int
    n_ops: int
    prompt: str


@dataclass(frozen=True)
class StepGrade:
    step_idx: int
    line: str
    value: int | None  # None if the line doesn't parse
    local_ok: bool
    global_ok: bool
    label: int | None  # 1 = incorrect, 0 = correct, None = not used (after first error / unparseable / cut off)


# ---- expression trees: a leaf is an int, an internal node is (op, left, right) ----

def _eval_tree(t) -> int:
    if isinstance(t, int):
        return t
    op, a, b = t
    a, b = _eval_tree(a), _eval_tree(b)
    return a + b if op == "+" else a - b if op == "-" else a * b


def _render(t) -> str:
    """Render with minimal parentheses (standard precedence, left-associative)."""
    if isinstance(t, int):
        return str(t)
    op, a, b = t
    sa, sb = _render(a), _render(b)
    if not isinstance(a, int) and PREC[a[0]] < PREC[op]:
        sa = f"({sa})"
    if not isinstance(b, int) and (PREC[b[0]] < PREC[op] or (PREC[b[0]] == PREC[op] and op == "-")):
        sb = f"({sb})"
    return f"{sa} {op} {sb}"


def _parse_tree(expr: str):
    """Parse an expression of ints, + - * and parentheses into a tree (ValueError otherwise)."""
    ops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}

    def conv(n):
        if isinstance(n, ast.Constant) and type(n.value) is int:
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            v = conv(n.operand)
            if isinstance(v, int):
                return -v
            return ("-", 0, v)
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return (ops[type(n.op)], conv(n.left), conv(n.right))
        raise ValueError(f"unsupported syntax: {ast.dump(n)}")

    try:
        return conv(ast.parse(expr, mode="eval").body)
    except SyntaxError as e:
        raise ValueError(str(e)) from e


def _reduce_once(t):
    """Evaluate the leftmost operation whose operands are both numbers."""
    if isinstance(t, int):
        return t
    op, a, b = t
    if isinstance(a, int) and isinstance(b, int):
        return _eval_tree(t)
    if not isinstance(a, int):
        return (op, _reduce_once(a), b)
    return (op, a, _reduce_once(b))


def solve_steps(expression: str) -> list[str]:
    """Reference solution: one ``= ...`` line per operation, ending at the value."""
    t, lines = _parse_tree(expression), []
    while not isinstance(t, int):
        t = _reduce_once(t)
        lines.append(f"= {_render(t)}")
    return lines


# ---- prompts and generation ----

def make_prompt(expression: str, shots=SHOT_EXPRESSIONS, instruction: str = INSTRUCTION) -> str:
    """Instruction, worked examples (from ``solve_steps``), then the target; the model continues with ``=``."""
    blocks = ["\n".join([s, *solve_steps(s)]) for s in shots]
    return "\n\n".join([instruction, *blocks, expression]) + "\n"


def _random_tree(rng: random.Random, n_ops: int, lo: int, hi: int, ops: str):
    if n_ops == 0:
        return rng.randint(lo, hi)
    k = rng.randrange(n_ops)  # operators in the left subtree
    return (rng.choice(ops), _random_tree(rng, k, lo, hi, ops), _random_tree(rng, n_ops - 1 - k, lo, hi, ops))


def _intermediates(t) -> list[int]:
    if isinstance(t, int):
        return [t]
    return _intermediates(t[1]) + _intermediates(t[2]) + [_eval_tree(t)]


def generate_problems(n: int, n_ops: int, seed: int = 0, operand_range: tuple[int, int] = (1, 20),
                      ops: str = "+-*", max_abs: int = 999, allow_negative: bool = False,
                      max_tries: int = 1000) -> list[Problem]:
    """Sample ``n`` distinct expressions with ``n_ops`` operators.

    Rejects expressions with any intermediate value above ``max_abs`` in magnitude, or below zero unless
    ``allow_negative``. IDs are ``op{n_ops}_s{seed}_{i:05d}``, so (n_ops, seed) pairs never collide.
    """
    rng, seen, out = random.Random(seed), set(), []
    for _ in range(n * max_tries):
        if len(out) == n:
            break
        t = _random_tree(rng, n_ops, *operand_range, ops)
        expr, vals = _render(t), _intermediates(t)
        if expr in seen or any(abs(v) > max_abs or (v < 0 and not allow_negative) for v in vals):
            continue
        seen.add(expr)
        pid = f"op{n_ops}_s{seed}_{len(out):05d}"
        out.append(Problem(pid, expr, _eval_tree(t), n_ops, make_prompt(expr)))
    if len(out) < n:
        raise RuntimeError(f"only found {len(out)}/{n} expressions; loosen the constraints")
    return out


def save_problems(problems: list[Problem], path: str | Path) -> None:
    with open(path, "w") as f:
        for p in problems:
            f.write(json.dumps(asdict(p)) + "\n")


def load_problems(path: str | Path) -> list[Problem]:
    with open(path) as f:
        return [Problem(**json.loads(line)) for line in f if line.strip()]


# ---- grading ----

def evaluate(expr: str) -> int | None:
    """Value of an ``int``/``+ - *``/parentheses expression, or None if it isn't one."""
    expr = expr.replace("×", "*").replace("−", "-").strip().rstrip(".").strip()
    try:
        return _eval_tree(_parse_tree(expr))
    except (ValueError, RecursionError):
        return None


def grade_steps(expression: str, lines: list[str], complete: bool = True) -> list[StepGrade]:
    """Grade a solution's ``=`` lines against ``expression``.

    Labels go to steps up to and including the first error. An unparseable line, and everything after it,
    gets ``label=None``; so does the last line when ``complete=False`` (generation hit the token limit and
    the line may be cut off).
    """
    truth = evaluate(expression)
    if truth is None:
        raise ValueError(f"not a valid expression: {expression!r}")
    grades, prev, stop = [], truth, False
    for i, line in enumerate(lines):
        body = line.strip()
        value = evaluate(body[1:]) if body.startswith("=") else None
        local_ok, global_ok = value is not None and value == prev, value == truth
        usable = not stop and value is not None and (complete or i < len(lines) - 1)
        grades.append(StepGrade(i, line, value, local_ok, global_ok, int(not global_ok) if usable else None))
        stop = stop or value is None or not global_ok
        prev = value if value is not None else prev
    return grades
