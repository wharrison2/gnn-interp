import random

import pytest

from gnn_interp import problems as P


@pytest.mark.parametrize("tree, text", [
    (("*", ("+", 3, 4), 2), "(3 + 4) * 2"),
    (("+", 1, ("*", 2, 3)), "1 + 2 * 3"),
    (("-", ("-", 5, 3), 1), "5 - 3 - 1"),
    (("-", 5, ("-", 3, 1)), "5 - (3 - 1)"),
    (("-", 5, ("+", 3, 1)), "5 - (3 + 1)"),
    (("*", 2, ("*", 3, 4)), "2 * 3 * 4"),
])
def test_render_minimal_parens(tree, text):
    assert P._render(tree) == text


def test_render_preserves_value():
    rng = random.Random(0)
    for _ in range(500):
        t = P._random_tree(rng, rng.randint(1, 8), 0, 30, "+-*")
        assert P.evaluate(P._render(t)) == P._eval_tree(t)


def test_solve_steps():
    assert P.solve_steps("(3 + 4) * 2") == ["= 7 * 2", "= 14"]
    assert P.solve_steps("12 - 5 * 2 + 1") == ["= 12 - 10 + 1", "= 2 + 1", "= 3"]
    assert P.solve_steps("7") == []


def test_solve_steps_one_op_per_line_and_value_preserved():
    for p in P.generate_problems(50, 5, seed=1):
        lines = P.solve_steps(p.expression)
        assert len(lines) == p.n_ops
        assert all(P.evaluate(line[1:]) == p.value for line in lines)


def test_make_prompt_format():
    prompt = P.make_prompt("8 + 9 * 2")
    assert prompt.startswith(P.INSTRUCTION) and prompt.endswith("\n\n8 + 9 * 2\n")
    assert "(3 + 4) * 2\n= 7 * 2\n= 14\n\n" in prompt


def test_generate_problems():
    ps = P.generate_problems(200, 4, seed=3, operand_range=(1, 20), max_abs=999)
    assert ps == P.generate_problems(200, 4, seed=3, operand_range=(1, 20), max_abs=999)
    assert len({p.expression for p in ps}) == len({p.problem_id for p in ps}) == 200
    for p in ps:
        assert sum(p.expression.count(c) for c in "+-*") == 4
        assert p.value == P.evaluate(p.expression) and 0 <= p.value <= 999
        assert p.prompt == P.make_prompt(p.expression)
    assert ps[0].problem_id.startswith("op4_") and ps[0].problem_id.endswith("_s3_00000")


def test_problem_ids_unique_across_settings():
    """Same n_ops and seed, different operand ranges (e.g. two difficulty levels): IDs must not collide."""
    a = P.generate_problems(5, 3, seed=0, operand_range=(1, 20))
    b = P.generate_problems(5, 3, seed=0, operand_range=(1, 50))
    assert not {p.problem_id for p in a} & {p.problem_id for p in b}
    assert [p.problem_id for p in a] == [p.problem_id for p in P.generate_problems(5, 3, seed=0)]  # stable
    assert P.generate_problems(2, 3, id_prefix="easy")[1].problem_id == "easy_00001"


def test_generate_problems_impossible_constraints():
    with pytest.raises(RuntimeError):
        P.generate_problems(10, 3, operand_range=(50, 60), ops="*", max_abs=100, max_tries=5)


def test_save_load_roundtrip(tmp_path):
    ps = P.generate_problems(5, 2)
    P.save_problems(ps, tmp_path / "p.jsonl")
    assert P.load_problems(tmp_path / "p.jsonl") == ps


@pytest.mark.parametrize("text, value", [
    ("3 + 4 * 2", 11), ("(3 + 4) * 2", 14), ("-3 * 4", -12), ("12 - (2 - 5)", 15),
    (" 14. ", 14), ("6 × 7", 42), ("9 − 2", 7),
    ("2 ** 3", None), ("8 / 2", None), ("abc", None), ("", None), ("1, 2", None),
    ("__import__('os')", None), ("1.5 + 1", None), ("True + 1", None),
])
def test_evaluate(text, value):
    assert P.evaluate(text) == value


def labels(grades):
    return [g.label for g in grades]


def test_grade_correct_solution():
    gs = P.grade_steps("(3 + 4) * 2", ["= 7 * 2", "= 14"])
    assert labels(gs) == [0, 0] and all(g.local_ok and g.global_ok for g in gs)


def test_grade_first_error_labeled_rest_excluded():
    # 42 * 3 - 48: step 2 slips (126 -> 125); step 3 is locally right but globally wrong.
    gs = P.grade_steps("(17 + 25) * 3 - 48", ["= 42 * 3 - 48", "= 125 - 48", "= 77", "= 78"])
    assert labels(gs) == [0, 1, None, None]
    assert [g.local_ok for g in gs] == [True, False, True, False]
    assert [g.global_ok for g in gs] == [True, False, False, True]


def test_grade_unparseable_stops_labels():
    gs = P.grade_steps("2 + 3 * 4", ["= 2 + 12", "= fourteen", "= 14"])
    assert labels(gs) == [0, None, None] and gs[1].value is None


def test_grade_line_without_equals_is_unparseable():
    assert labels(P.grade_steps("2 + 3", ["5"])) == [None]


def test_grade_incomplete_drops_last_line():
    lines = ["= 42 * 3 - 48", "= 126 - 48", "= 7"]  # last line cut off mid-number
    assert labels(P.grade_steps("(17 + 25) * 3 - 48", lines, complete=False)) == [0, 0, None]


def test_grade_rejects_bad_expression():
    with pytest.raises(ValueError):
        P.grade_steps("2 +", ["= 2"])
