import json

from gnn_interp import analysis


def test_align_and_step_info():
    assert analysis.align("7 * 17 - 5", "= 119 - 5") == (["7", "*", "17"], ["119"])
    assert analysis.align("43 * (89 - 9)", "= 43 * 80") == (["(", "89", "-", "9", ")"], ["80"])
    s = analysis.step_info("732 * 839 + 5", "= 619,108 + 5")
    assert s["kind"] == "arithmetic" and s["true"] == 614148 and s["n_ops"] == 1 and s["max_digits"] == 3
    assert s["same_length"] and s["last_digit_ok"] and s["digits_wrong"] == 2
    assert analysis.step_info("43 * (89 - 9) + 5", "= 43 * (89 - 9)")["kind"] == "copy"  # dropped a term
    assert analysis.step_info("263 * 31 - (756 - 539)", "= 7993 - 221")["kind"] == "arithmetic"
    assert analysis.step_info("2 * 3 + 4 * 5", "= 6 + 20")["n_ops"] == 3  # two products in one line


def test_load_steps_and_breakdown(tmp_path):
    gens = [{"problem_id": "p0", "level": "easy", "expression": "2 * 3 + 1"},
            {"problem_id": "p1", "level": "medium", "expression": "9 * 9 - 1"}]
    steps = [{"problem_id": "p0", "level": "easy", "step_idx": 0, "line": "= 6 + 1", "label": 0, "pos": 10},
             {"problem_id": "p0", "level": "easy", "step_idx": 1, "line": "= 7", "label": 0, "pos": 12},
             {"problem_id": "p1", "level": "medium", "step_idx": 0, "line": "= 82 - 1", "label": 1, "pos": 10},
             {"problem_id": "p1", "level": "medium", "step_idx": 1, "line": "= 81", "label": None, "pos": 12}]
    for name, rows in (("generations", gens), ("steps", steps)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    rows = analysis.load_steps(tmp_path)
    assert [(r["problem_id"], r["step_idx"]) for r in rows] == [("p0", 0), ("p0", 1), ("p1", 0)]
    assert rows[1]["prev"] == "= 6 + 1" and rows[1]["before"] == "6 + 1"
    b = analysis.error_breakdown(rows)
    assert b["incorrect_kinds"] == {"arithmetic": 1} and b["by_level"]["medium"]["error_rate"] == 1.0
