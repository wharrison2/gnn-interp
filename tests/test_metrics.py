import numpy as np
import pytest

from gnn_interp.metrics import compute_metrics, fpr_at_tpr

Y = np.array([0, 0, 0, 1, 1, 1])


def test_perfect_and_inverted():
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = compute_metrics(Y, p)
    assert m["auroc"] == m["aupr"] == m["accuracy"] == m["precision"] == m["recall"] == 1.0
    assert m["fpr@95"] == 0.0
    assert compute_metrics(Y, 1 - p)["auroc"] == 0.0


def test_constant_scores_are_chance():
    m = compute_metrics(Y, np.full(6, 0.5))
    assert m["auroc"] == 0.5 and m["recall"] == 1.0 and m["precision"] == 0.5 and m["fpr@95"] == 1.0


def test_threshold_and_zero_division():
    m = compute_metrics(Y, np.full(6, 0.1))
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["accuracy"] == 0.5
    assert compute_metrics(Y, np.full(6, 0.1), threshold=0.05)["recall"] == 1.0


def test_fpr_at_tpr_known_value():
    # positives need threshold 0.4 to reach TPR 1; one negative (0.5) scores above it
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.5, 0.4, 0.9])
    assert fpr_at_tpr(y, p, 0.95) == pytest.approx(0.25)
    assert fpr_at_tpr(y, p, 0.5) == 0.0


def test_fpr_at_tpr_keeps_collinear_roc_points():
    # tied (pos, neg) pairs at 0.5 and 0.4 make collinear ROC points; TPR 0.95 is reached at FPR 0.1
    y = np.array([1] * 19 + [0] + [1, 0] + [0] * 8)
    p = np.array([0.9] * 18 + [0.5, 0.5] + [0.4, 0.4] + [0.1] * 8)
    assert fpr_at_tpr(y, p) == pytest.approx(0.1)


def test_single_class_ranking_metrics_are_nan():
    m = compute_metrics(np.zeros(4), np.linspace(0, 1, 4))
    assert np.isnan(m["auroc"]) and np.isnan(m["aupr"]) and np.isnan(m["fpr@95"]) and m["accuracy"] == 0.5
