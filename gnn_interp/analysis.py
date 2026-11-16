"""Text-only analyses of a ``class_balance`` run (CPU, no model): what each step computed, what kinds of
errors the model makes, and whether trivial step features already predict errors.

    python -m gnn_interp.analysis --steps runs/balance --sizes runs/sizes --out runs/analysis

Every step is aligned with the expression before it (the problem, or the previous line): the common
token prefix and suffix are copied, the middle of the previous expression (``before``) is what the step
computed and the middle of the step (``after``) is the model's result.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .problems import evaluate

_TOKEN = re.compile(r"\d+|[-+*()]")
LEVEL_ORDER = ("easy", "medium", "hard", "xhard")


def tokens(expr: str) -> list[str]:
    """Numbers and operators of an expression (thousands separators removed, leading ``=`` dropped)."""
    expr = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expr.strip().lstrip("=").replace("×", "*"))
    return _TOKEN.findall(expr)


def align(prev: str, line: str) -> tuple[list[str], list[str]]:
    """``(before, after)``: the middle of ``prev`` that ``line`` replaced, and what replaced it."""
    a, b = tokens(prev), tokens(line)
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    j = 0
    while j < min(len(a), len(b)) - i and a[-1 - j] == b[-1 - j]:
        j += 1
    return a[i:len(a) - j], b[i:len(b) - j]


def _value(toks: list[str]) -> int | None:
    return evaluate(" ".join(toks)) if toks else None


def step_info(prev: str, line: str) -> dict:
    """What the step computed: operators collapsed, operand sizes, true and written result, error kind.

    ``kind`` (meaningful for incorrect steps): ``arithmetic`` (the step wrote a new number: it computed
    something), ``copy`` (no new number: terms were dropped or rearranged), or ``unknown`` (unparseable).
    Digit statistics are added when the step replaced the sub-expression by a single number.
    """
    before, after = align(prev, line)
    nums = [t for t in before if t.isdigit()]
    ops = [t for t in before if t in "+-*"]
    true, written = _value(before), _value(after)
    new = Counter(t for t in after if t.isdigit()) - Counter(nums)
    kind = "copy" if not new else "unknown" if true is None or written is None else "arithmetic"
    info = {"before": " ".join(before), "after": " ".join(after), "n_ops": len(ops), "ops": "".join(sorted(set(ops))),
            "max_digits": max((len(t) for t in nums), default=0), "true": true, "written": written, "kind": kind,
            "prev_ops": sum(t in "+-*" for t in tokens(prev)), "prev_max_digits": max(
                (len(t) for t in tokens(prev) if t.isdigit()), default=0)}
    if kind == "arithmetic" and len(after) == 1 and written != true:
        s, t = str(abs(written)), str(abs(true))
        info |= {"same_length": len(s) == len(t), "last_digit_ok": s[-1] == t[-1],
                 "digits_wrong": sum(x != y for x, y in zip(s.zfill(len(t)), t.zfill(len(s)))),
                 "rel_error": abs(written - true) / max(abs(true), 1)}
    return info


def load_steps(steps_dir: Path) -> list[dict]:
    """Labelled steps of a ``class_balance`` run, each with ``prev`` (the expression before it) and ``step_info``."""
    gens = {g["problem_id"]: g for g in map(json.loads, open(steps_dir / "generations.jsonl"))}
    by_problem = defaultdict(list)
    for r in map(json.loads, open(steps_dir / "steps.jsonl")):
        by_problem[r["problem_id"]].append(r)
    out = []
    for pid, rows in by_problem.items():
        prev = gens[pid]["expression"]
        for r in sorted(rows, key=lambda r: r["step_idx"]):
            if r.get("label") is not None:
                out.append(r | {"prev": prev, "n_steps": len(rows)} | step_info(prev, r["line"]))
            prev = r["line"]
    return out


def _rate(rows) -> dict:
    return {"n": len(rows), "incorrect": sum(r["label"] for r in rows),
            "error_rate": round(float(np.mean([r["label"] for r in rows])), 3) if rows else None}


def error_breakdown(steps: list[dict], n_examples: int = 25, seed: int = 0) -> dict:
    """Error rates by level, step index, operation and operand size; kinds of incorrect steps; examples."""
    bad = [s for s in steps if s["label"] == 1]
    arith = [s for s in bad if "digits_wrong" in s]
    group = lambda key: {str(k): _rate([s for s in steps if key(s) == k])  # noqa: E731
                         for k in sorted({key(s) for s in steps}, key=str)}
    examples = random.Random(seed).sample(bad, min(n_examples, len(bad)))
    return {
        "by_level": group(lambda s: s["level"]),
        "by_step_idx": group(lambda s: min(s["step_idx"], 4)),  # 4 = 4+
        "by_ops_in_step": group(lambda s: s["ops"] if s["n_ops"] == 1 else f"ops={s['n_ops']}"),
        "by_max_digits": group(lambda s: min(s["max_digits"], 4)),
        "by_level_and_op": group(lambda s: f"{s['level']}:{s['ops'] if s['n_ops'] == 1 else 'multi'}"),
        "incorrect_kinds": dict(Counter(s["kind"] for s in bad)),
        "single_result_errors": {  # incorrect steps that wrote one wrong number
            "n": len(arith),
            "same_length": float(np.mean([s["same_length"] for s in arith])),
            "last_digit_ok": float(np.mean([s["last_digit_ok"] for s in arith])),
            "one_digit_wrong": float(np.mean([s["digits_wrong"] == 1 for s in arith])),
            "rel_error_p50": float(np.median([s["rel_error"] for s in arith])),
        } if arith else {},
        "examples": [{k: s[k] for k in ("level", "step_idx", "prev", "line", "before", "after", "true", "kind")}
                     for s in examples],
    }


FEATURE_SETS = {
    # known before the model writes the step
    "problem": ["level", "step_idx", "prev_ops", "prev_max_digits", "prev_has_mul", "pos"],
    # + what the step chose to compute (not its result)
    "step": ["level", "step_idx", "prev_ops", "prev_max_digits", "prev_has_mul", "pos",
             "n_ops", "max_digits", "has_mul", "true_digits"],
}


def _features(s: dict, names: list[str]) -> list[float]:
    f = {"level": LEVEL_ORDER.index(s["level"]), "step_idx": s["step_idx"], "prev_ops": s["prev_ops"],
         "prev_max_digits": s["prev_max_digits"], "prev_has_mul": "*" in s["prev"], "pos": s["pos"],
         "n_ops": s["n_ops"], "max_digits": s["max_digits"], "has_mul": "*" in s["ops"],
         "true_digits": len(str(abs(s["true"]))) if s["true"] is not None else 0}
    return [float(f[n]) for n in names]


def trivial_baseline(steps: list[dict], levels=LEVEL_ORDER, n_folds: int = 5, seeds=range(5)) -> dict:
    """Problem-grouped CV AUROC/AUPR of logistic regression and gradient boosting on ``FEATURE_SETS``.

    The bar a graph-based classifier has to clear: errors predictable from problem difficulty alone.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .train import grouped_cv, summarize

    rows = [s for s in steps if s["level"] in levels]
    y, groups = np.array([s["label"] for s in rows]), np.array([s["problem_id"] for s in rows])
    out = {"levels": list(levels), **_rate(rows)}
    models = {"logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced",
                                                                                    max_iter=2000)),
              "boosting": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=100,
                                                                 class_weight="balanced")}
    for fs, names in FEATURE_SETS.items():
        X = np.array([_features(s, names) for s in rows])
        for m, make in models.items():
            def fit_predict(tr, te, seed, make=make, X=X):
                return make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]

            res, _ = grouped_cv(y, groups, fit_predict, n_folds, seeds)
            metrics = summarize(res)
            out[f"{fs}/{m}"] = {k: [round(v, 3) for v in metrics[k]] for k in ("auroc", "aupr")}
    return out


def graph_stats_by_label(sizes_rows: list[dict], cap: str = "4096", threshold: str = "0.8/0.98") -> dict:
    """Per graph statistic: mean for correct / incorrect steps and AUROC of the statistic as a predictor.

    AUROC far from 0.5 means the statistic alone separates the classes (a confound for a graph classifier).
    """
    from sklearn.metrics import roc_auc_score

    rows = [r for r in sizes_rows if cap in r["by_cap"]]
    y = np.array([r["label"] for r in rows])
    stats = {
        "replacement_score": lambda c: c["replacement_score"],
        "completeness_score": lambda c: c["completeness_score"],
        "n_pos": lambda c: c["n_pos"],
        "n_active": lambda c: c["n_active"],
        "pruned_nodes": lambda c: c["pruned"][threshold]["nodes"],
        "pruned_error_nodes": lambda c: c["pruned"][threshold]["error"],
        "pruned_error_frac": lambda c: c["pruned"][threshold]["error"] / c["pruned"][threshold]["nodes"],
        "pruned_edges": lambda c: c["pruned"][threshold]["edges"],
        "n_logits": lambda c: c["pruned"][threshold]["logit"],
    }
    out = {"n": len(rows), "incorrect": int(y.sum())}
    for name, f in stats.items():
        v = np.array([f(r["by_cap"][cap]) for r in rows], dtype=float)
        out[name] = {"correct": round(float(v[y == 0].mean()), 4), "incorrect": round(float(v[y == 1].mean()), 4),
                     "auroc": round(float(roc_auc_score(y, v)), 3)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=Path, required=True, help="class_balance output directory")
    ap.add_argument("--sizes", type=Path, help="graph_sizes output directory (for graph_stats_by_label)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    steps = load_steps(a.steps)
    result = {"errors": error_breakdown(steps),
              "baseline_all": trivial_baseline(steps),
              "baseline_no_easy": trivial_baseline(steps, LEVEL_ORDER[1:])}
    if a.sizes:
        result["graph_stats_by_label"] = graph_stats_by_label(
            [json.loads(line) for line in open(a.sizes / "sizes.jsonl")])
    (a.out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "errors"}, indent=1))


if __name__ == "__main__":
    main()
