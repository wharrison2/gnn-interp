"""Text/difficulty baselines: number parsing, arithmetic check, oracle."""
import numpy as np
import pytest

from gnn_interp.baselines import (all_baselines, arithmetic_check, difficulty_features, oracle_baseline,
                                  text_features)


def test_text_features():
    f = text_features("12 + 30 = 42", "5 * 2 = 10")
    assert f[0] == 12 and f[4] == 5 and f[-1] == 2 and len(f) == 9
    assert np.array_equal(text_features("12 + 30 = 43")[:2], [12, 30]) and text_features("12 + 30 = 43")[2] == 43
    assert np.isnan(text_features("no math here")[-1])
    assert text_features("12-3 = 9")[:3].tolist() == [12, 3, 9]


@pytest.mark.parametrize("line, prev, expected", [
    ("= 42 + 5", "12 + 30 + 5", 1.0), ("= 47", "= 42 + 5", 1.0), ("= 48", "= 42 + 5", 0.0),
    ("12 * (3 + 1) = 48", "", 1.0), ("12 × 2 = 25", "1 + 1 = 2", 0.0), ("= 5", "10 / 0", np.nan),
])
def test_arithmetic_check_chained_steps(line, prev, expected):
    out = arithmetic_check(line, prev)
    assert (np.isnan(out) and np.isnan(expected)) or out == expected


def test_oracle_baseline_is_the_label(crv_dataset):
    y = np.array([d.y.item() for d in crv_dataset])
    p = oracle_baseline(crv_dataset)(np.arange(40), np.arange(40, 60), 0)
    assert (p == y[40:]).all()
    assert "oracle_check" in all_baselines(crv_dataset)
    assert not np.isin(text_features("1 + 1 = 3"), [1.0]).all()  # no check column


def test_difficulty_features_ignore_answer():
    f = difficulty_features("48 + 39 = 87", "", 2)
    assert np.array_equal(f, difficulty_features("48 + 39 = 99", "", 2), equal_nan=True)  # answer-blind
    assert f[0] == 48 and f[1] == 39 and f[4] == 2 and f[8] == 1 and f[12] == 1  # operands, digits, '+', carry
    assert f[-1] == 2 and difficulty_features("48 + 31 = 79")[12] == 0
    assert difficulty_features("52 - 38 = 14")[12] == 1  # borrow
    g = difficulty_features("= 42 + 5", "12 + 30 + 5")  # chained: operands from prev_line, 1 op remaining
    assert g[:3].tolist() == [12, 30, 5] and g[-2] == 1
    assert np.array_equal(g, difficulty_features("= 99 + 5", "12 + 30 + 5"), equal_nan=True)
