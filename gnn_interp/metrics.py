"""Classification metrics. Positive class (label 1) = the step is INCORRECT."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def fpr_at_tpr(y, p, tpr_target: float = 0.95) -> float:
    """Lowest false-positive rate achieving at least ``tpr_target`` true-positive rate."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(fpr[np.searchsorted(tpr, tpr_target, side="left")])


def compute_metrics(y, p, threshold: float = 0.5) -> dict[str, float]:
    """AUROC, AUPR, FPR@95TPR, and accuracy/precision/recall at ``threshold``."""
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= threshold).astype(int)
    return {
        "auroc": roc_auc_score(y, p),
        "aupr": average_precision_score(y, p),
        "fpr@95": fpr_at_tpr(y, p),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
    }
