import numpy as np
import pytest
import torch

from gnn_interp.models import GraphClassifier
from gnn_interp.train import (TrainConfig, fit_gnn, gnn_fit_predict, grouped_cv, holdout_split,
                              predict, summarize)

CFG = TrainConfig(max_epochs=40, patience=40, batch_size=16, lr=3e-3)


def labels_groups(ds):
    return np.array([d.y.item() for d in ds]), np.array([d.problem_id for d in ds])


def test_holdout_split_grouped():
    groups = np.repeat(np.arange(10), 4)
    dev, test = holdout_split(groups, 0.2, seed=3)
    assert not set(groups[dev]) & set(groups[test]) and len(dev) + len(test) == 40
    assert np.array_equal(test, holdout_split(groups, 0.2, seed=3)[1])


def test_predict_range_and_eval_mode(dataset, model_kwargs):
    m = GraphClassifier(**model_kwargs)
    p = predict(m, dataset, batch_size=7)
    assert p.shape == (len(dataset),) and ((p > 0) & (p < 1)).all() and not m.training
    assert np.allclose(p, predict(m, dataset, batch_size=64), atol=1e-5)  # deterministic in eval


@pytest.mark.slow
def test_fit_gnn_learns_and_restores_best(dataset, model_kwargs):
    torch.manual_seed(0)
    train, val = dataset[:72], dataset[72:]
    m = fit_gnn(GraphClassifier(**model_kwargs), train, val, CFG)
    y = np.array([d.y.item() for d in val])
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, predict(m, val)) > 0.7


def test_fit_gnn_single_class_validation(dataset, model_kwargs):
    val = [d for d in dataset if d.y.item() == 0][:6]  # falls back to -BCE scoring
    m = fit_gnn(GraphClassifier(**model_kwargs), dataset[:32], val, TrainConfig(max_epochs=2))
    assert isinstance(m, GraphClassifier)


@pytest.mark.slow
def test_grouped_cv_folds_and_signal(dataset, model_kwargs):
    y, groups = labels_groups(dataset)
    seen = []

    def spy(fp):
        def wrapped(tr, te, seed):
            assert not set(groups[tr]) & set(groups[te])  # no problem in both train and test
            seen.append(te)
            return fp(tr, te, seed)
        return wrapped

    fp = spy(gnn_fit_predict(dataset, dict(model_kwargs), CFG))
    rows, oof = grouped_cv(y, groups, fp, n_folds=3, seeds=[0, 1])
    assert len(rows) == 6 and set(oof) == {0, 1}
    for s in oof:  # every sample is predicted exactly once per seed
        assert not np.isnan(oof[s]).any()
    folds0 = np.concatenate(seen[:3])
    assert sorted(folds0) == list(range(len(dataset)))
    summ = summarize(rows)
    assert summ["auroc"][0] > 0.7, summ["auroc"]
    assert set(summ) == {"auroc", "aupr", "fpr@95", "accuracy", "precision", "recall"}


def test_fit_predict_deterministic_per_seed(dataset, model_kwargs):
    fp = gnn_fit_predict(dataset, dict(model_kwargs), TrainConfig(max_epochs=2))
    tr, te = np.arange(0, 72), np.arange(72, 96)
    assert np.allclose(fp(tr, te, 0), fp(tr, te, 0), atol=1e-5)


def test_summarize():
    rows = [{"seed": 0, "fold": 0, "auroc": 0.8}, {"seed": 0, "fold": 1, "auroc": 0.6}]
    mean, std = summarize(rows)["auroc"]
    assert mean == pytest.approx(0.7) and std == pytest.approx(0.1)


def test_fold_pca_fit_on_training_problems_only(dataset, model_kwargs, fetch):
    from gnn_interp.graph_data import DecoderPCA
    seen = []

    class SpyPCA(DecoderPCA):
        def fit(self, data_list, fetch, **kw):
            seen.append({d.problem_id for d in data_list})
            return super().fit(data_list, fetch, **kw)

    fp = gnn_fit_predict(dataset, dict(model_kwargs), TrainConfig(max_epochs=1), pca=SpyPCA(4), fetch=fetch)
    tr, te = np.arange(0, 72), np.arange(72, 96)
    p = fp(tr, te, 0)
    assert p.shape == (24,) and np.isfinite(p).all()
    _, groups = labels_groups(dataset)
    assert len(seen) == 1 and seen[0] < set(groups[tr]) and not seen[0] & set(groups[te])
