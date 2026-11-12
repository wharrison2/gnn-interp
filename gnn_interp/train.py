"""Training loop and grouped cross-validation harness.

Every model (GNN or baseline) is wrapped as ``fit_predict(train_idx, test_idx, seed) -> probs``
over a fixed dataset, so ``grouped_cv`` runs all of them on identical problem-grouped folds.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch_geometric.loader import DataLoader

from .graph_data import DecoderPCA, FeatureNormalizer
from .metrics import compute_metrics
from .models import GraphClassifier

FitPredict = Callable[[np.ndarray, np.ndarray, int], np.ndarray]


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 20  # epochs without validation improvement before stopping
    device: str = "cpu"


def _labels(data_list) -> torch.Tensor:
    return torch.tensor([d.y.item() for d in data_list])


@torch.no_grad()
def _logits(model, data_list, batch_size: int, device: str) -> torch.Tensor:
    model.eval().to(device)
    return torch.cat([model(b.to(device)).cpu() for b in DataLoader(data_list, batch_size)])


def predict(model, data_list, batch_size: int = 64, device: str = "cpu") -> np.ndarray:
    """Sigmoid P(incorrect) for each graph."""
    return torch.sigmoid(_logits(model, data_list, batch_size, device)).numpy()


def _val_score(model, val, cfg: TrainConfig, pos_weight: torch.Tensor) -> float:
    """Validation AUROC, or -(class-weighted BCE) when the validation set holds a single class."""
    z, y = _logits(model, val, cfg.batch_size, cfg.device), _labels(val)
    if len(y.unique()) == 2:
        return roc_auc_score(y.numpy(), z.numpy())
    return -F.binary_cross_entropy_with_logits(z, y, pos_weight=pos_weight).item()


def fit_gnn(model, train, val, cfg: TrainConfig = TrainConfig()):
    """Train with class-weighted BCE and AdamW; early-stop on ``val`` and restore the best weights.

    Sets ``model.epochs_run`` (epochs trained before stopping).
    """
    y = _labels(train)
    pos_weight = (len(y) - y.sum()) / y.sum().clamp_min(1)  # n_neg / n_pos (positive = incorrect)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(cfg.device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(train, cfg.batch_size, shuffle=True)
    model.to(cfg.device)

    best, best_state, stale = -np.inf, copy.deepcopy(model.state_dict()), 0
    for model.epochs_run in range(1, cfg.max_epochs + 1):
        model.train()
        for b in loader:
            b = b.to(cfg.device)
            opt.zero_grad()
            loss_fn(model(b), b.y).backward()
            opt.step()
        score = _val_score(model, val, cfg, pos_weight)
        if score > best:
            best, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
        elif (stale := stale + 1) >= cfg.patience:
            break
    model.load_state_dict(best_state)
    return model


def gnn_fit_predict(data_list, model_kwargs: dict, cfg: TrainConfig = TrainConfig(),
                    val_frac: float = 0.15, pca: DecoderPCA | None = None, fetch=None) -> FitPredict:
    """Wrap ``GraphClassifier`` training as a ``FitPredict``.

    Per fold: an inner label-stratified, problem-grouped validation split (~``val_frac``) for early stopping; a copy of ``pca``
    (if given, with the decoder ``fetch``er) and a ``FeatureNormalizer``, both fit on the inner
    training graphs only. Without ``pca`` the decoder-PCA columns are used as they are in ``data_list``.
    ``fit_predict.epochs`` collects each fit's ``epochs_run``.
    """
    assert (pca is None) == (fetch is None), "pass pca and fetch together"
    groups = np.array([d.problem_id for d in data_list])
    y = np.array([d.y.item() for d in data_list])

    def fit_predict(train_idx, test_idx, seed):
        torch.manual_seed(seed)
        split = StratifiedGroupKFold(round(1 / val_frac), shuffle=True, random_state=seed)
        tr, va = next(split.split(train_idx, y[train_idx], groups[train_idx]))
        tr, va = train_idx[tr], train_idx[va]
        train, val, test = ([data_list[i] for i in idx] for idx in (tr, va, test_idx))
        if pca is not None:
            fold_pca = copy.deepcopy(pca).fit(train, fetch, seed=seed)
            train, val, test = (fold_pca.apply(part, fetch) for part in (train, val, test))
        norm = FeatureNormalizer().fit(train)
        train, val, test = (norm.apply(part) for part in (train, val, test))
        model = fit_gnn(GraphClassifier(**model_kwargs), train, val, cfg)
        fit_predict.epochs.append(model.epochs_run)
        return predict(model, test, cfg.batch_size, cfg.device)

    fit_predict.epochs = []
    return fit_predict


def holdout_split(groups, test_frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Problem-grouped (dev_idx, test_idx) split; the test set is touched once, at the end."""
    split = GroupShuffleSplit(1, test_size=test_frac, random_state=seed)
    return next(split.split(np.zeros(len(groups)), groups=groups))


def grouped_cv(y, groups, fit_predict: FitPredict, n_folds: int = 5,
               seeds=range(5)) -> tuple[list[dict], dict[int, np.ndarray]]:
    """Stratified, problem-grouped k-fold CV repeated over ``seeds``.

    Returns per-(seed, fold) metric rows and out-of-fold probabilities per seed.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    rows, oof = [], {}
    for seed in seeds:
        oof[seed] = np.full(len(y), np.nan)
        cv = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            p = fit_predict(tr, te, seed)
            oof[seed][te] = p
            rows.append({"seed": seed, "fold": fold, **compute_metrics(y[te], p)})
    return rows, oof


def summarize(rows: list[dict]) -> dict[str, tuple[float, float]]:
    """Mean and std of each metric across folds and seeds."""
    keys = [k for k in rows[0] if k not in ("seed", "fold")]
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in keys}
