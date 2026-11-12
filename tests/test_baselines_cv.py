"""Baselines under grouped cross-validation (fitting runs)."""
import numpy as np
import pytest

from gnn_interp.baselines import all_baselines, crv_baseline, hybrid_baseline, top_prob_baseline
from gnn_interp.train import grouped_cv

pytestmark = pytest.mark.slow


def test_crv_baseline_models(crv_dataset):
    y = [d.y.item() for d in crv_dataset]
    groups = [d.problem_id for d in crv_dataset]
    for model in ("hgb", "gbc"):
        rows, preds = grouped_cv(y, groups, crv_baseline(crv_dataset, model=model), n_folds=3, seeds=[0])
        assert np.isfinite(preds[0]).all() and np.mean([r["auroc"] for r in rows]) > 0.8, model


def test_grouped_cv_all_baselines(crv_dataset):
    y, groups = [d.y.item() for d in crv_dataset], [d.problem_id for d in crv_dataset]
    _, oof = grouped_cv(y, groups, all_baselines(crv_dataset)["top_prob"], n_folds=3, seeds=[0])
    fits = all_baselines(crv_dataset, oof)
    assert "hybrid" in fits
    for name, fp in fits.items():
        rows, preds = grouped_cv(y, groups, fp, n_folds=3, seeds=[0])
        assert np.isfinite(preds[0]).all() and ((preds[0] >= 0) & (preds[0] <= 1)).all(), name
        if name in ("top_prob", "crv", "hybrid"):  # confidence is informative by construction
            assert np.mean([r["auroc"] for r in rows]) > 0.8, name


def test_hybrid_accepts_array(crv_dataset):
    p = np.linspace(0, 1, len(crv_dataset))
    out = hybrid_baseline(crv_dataset, p)(np.arange(40), np.arange(40, 60), 0)
    assert out.shape == (20,)


def test_hybrid_nested_crossfit_is_grouped(crv_dataset):
    """With a GNN FitPredict, the hybrid never trains the GNN on its outer test problems."""
    groups = np.array([d.problem_id for d in crv_dataset])
    inner = top_prob_baseline(crv_dataset)
    calls = []

    def gnn(tr, te, seed):
        calls.append((tr, te))
        assert not set(groups[tr]) & set(groups[te])
        return inner(tr, te, seed)

    tr, te = np.arange(40), np.arange(40, 60)
    out = hybrid_baseline(crv_dataset, gnn, n_inner=3)(tr, te, 0)
    assert out.shape == (20,) and np.isfinite(out).all() and len(calls) == 4
    assert all(not set(t) & set(te) for t, _ in calls)  # outer test rows never used for GNN training
    assert sorted(np.concatenate([v for _, v in calls[:3]])) == list(tr)  # inner OOF covers train rows once
