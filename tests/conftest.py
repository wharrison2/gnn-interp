"""Shared pytest fixtures (builders live in ``tests/synthetic.py``)."""
import os

import pytest
import torch

# Workaround, tests only: sklearn HistGradientBoosting segfaults when torch is loaded with multi-threaded
# OpenMP (sklearn 1.7 + torch 2.12 on macOS/anaconda). Harmless elsewhere (e.g. Colab: tests are tiny).
# Real training is unaffected because this only applies inside the pytest process.
os.environ.setdefault("OMP_NUM_THREADS", "1")

from gnn_interp import graph_data  # noqa: E402  (after the env var)
from tests.synthetic import D_MODEL, MAX_POS, N_LAYERS, N_PCA, make_crv_dataset, make_dataset  # noqa: E402


@pytest.fixture(scope="session")
def dataset():
    """Synthetic circuit-graph dataset with a planted signal (incorrect steps have higher activations)."""
    return make_dataset()


@pytest.fixture(scope="session")
def crv_dataset():
    """Small hand-built graphs with logit probs and arithmetic lines, for the baselines."""
    return make_crv_dataset()


@pytest.fixture(scope="session", params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request):
    """``cpu`` always; ``cuda`` only when available (skipped otherwise)."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


@pytest.fixture
def fetch():
    """Fake decoder fetcher: deterministic random decoder rows per (layer, feature id)."""
    def f(layer, ids):
        return torch.stack([torch.randn(D_MODEL, generator=torch.Generator().manual_seed(int(layer) * 1000 + int(i)))
                            for i in ids])
    return f


@pytest.fixture
def model_kwargs():
    return dict(type_dims=graph_data.type_dims(N_PCA), hidden=16, n_model_layers=N_LAYERS, max_pos=MAX_POS)
