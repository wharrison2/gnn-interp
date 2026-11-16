"""Interruption safety: truncated jsonl lines, atomic saves, resumable grouped CV."""
import numpy as np
import torch

from gnn_interp import jsonl
from gnn_interp.train import grouped_cv


def test_read_drops_truncated_last_line(tmp_path):
    p = tmp_path / "rows.jsonl"
    jsonl.append(p, {"a": 1}, {"a": 2})
    with p.open("a") as f:
        f.write('{"a": 3, "b"')  # interrupted mid-write
    assert jsonl.read(p) == [{"a": 1}, {"a": 2}]
    jsonl.append(p, {"a": 4})  # next append starts on a clean line
    assert jsonl.read(p) == [{"a": 1}, {"a": 2}, {"a": 4}]
    assert jsonl.read(tmp_path / "missing.jsonl") == []


def test_save_atomic(tmp_path):
    p = tmp_path / "data.pt"
    jsonl.save_atomic([1, 2], p)
    jsonl.save_atomic([1, 2, 3], p)
    assert torch.load(p) == [1, 2, 3] and not (tmp_path / "data.pt.tmp").exists()


def test_grouped_cv_resumes_from_checkpoint(tmp_path):
    rng = np.random.default_rng(0)
    y, groups = np.array([0, 1] * 20), np.repeat(np.arange(20), 2)
    calls = []

    def fit_predict(tr, te, seed):
        calls.append((seed, len(te)))
        return rng.random(len(te))

    ckpt = tmp_path / "folds.jsonl"
    rows, oof = grouped_cv(y, groups, fit_predict, n_folds=4, seeds=[0, 1], checkpoint=ckpt)
    assert len(calls) == 8
    lines = ckpt.read_text().splitlines()
    ckpt.write_text("\n".join(lines[:5]) + "\n" + lines[5][:10])  # interrupted during the 6th fold's write
    rows2, oof2 = grouped_cv(y, groups, fit_predict, n_folds=4, seeds=[0, 1], checkpoint=ckpt)
    assert len(calls) == 8 + 3  # only the 3 unfinished folds were refit
    np.testing.assert_allclose(oof2[0], oof[0])  # seed 0 fully reused
    assert [(r["seed"], r["fold"]) for r in rows2] == [(r["seed"], r["fold"]) for r in rows]
