"""Synthetic builders shared by the tests (no circuit-tracer or model needed)."""
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import Data

from gnn_interp import graph_data
from gnn_interp.graph_data import ERROR, FEATURE, LOGIT, TOKEN

N_LAYERS, N_POS, N_LOGIT, N_PCA, MAX_POS, VOCAB, D_MODEL = 3, 4, 3, 4, 8, 10, 16
TOKEN_CLASSES = torch.arange(VOCAB) % graph_data.N_TOKEN_CLASSES


def make_graph(rng: np.random.Generator, shift: float = 0.0, n_feat: int = 12) -> SimpleNamespace:
    """Lightweight fake of ``circuit_tracer.graph.Graph`` with the attributes graph_data reads.

    Node order: features | errors (n_layers x n_pos) | tokens | logits. Edges go from lower to
    higher node index (rows = targets). Feature activations are shifted by ``shift``.
    """
    n = n_feat + N_LAYERS * N_POS + N_POS + N_LOGIT
    adj = torch.from_numpy(rng.normal(size=(n, n)) * (rng.random((n, n)) < 0.15)).float().tril(-1)
    active = torch.stack([torch.from_numpy(rng.integers(0, N_LAYERS, n_feat)),
                          torch.from_numpy(rng.integers(0, N_POS, n_feat)),
                          torch.from_numpy(rng.integers(0, 20, n_feat))], 1)
    probs = torch.tensor(np.sort(rng.dirichlet(np.ones(N_LOGIT)))[::-1].copy()).float()
    return SimpleNamespace(
        adjacency_matrix=adj, cfg=SimpleNamespace(n_layers=N_LAYERS), n_pos=N_POS,
        selected_features=torch.arange(n_feat), logit_targets=list(range(N_LOGIT)),
        active_features=active, activation_values=torch.from_numpy(rng.normal(shift, 1, n_feat)).float(),
        input_tokens=torch.from_numpy(rng.integers(0, VOCAB, N_POS)), logit_probabilities=probs,
        logit_token_ids=torch.arange(N_LOGIT))


def make_layered_graph(rng: np.random.Generator, n_feat: int = 200, density: float = 0.3) -> SimpleNamespace:
    """``make_graph`` with real attribution-graph structure: tokens feed features/errors/logits, and a
    feature or error at (layer l, pos p) feeds features at a later layer and position >= p, and the logits."""
    g = make_graph(rng, n_feat=n_feat)
    feat_layer, feat_pos = g.active_features[:, 0], g.active_features[:, 1]
    err_layer, err_pos = torch.arange(N_LAYERS).repeat_interleave(N_POS), torch.arange(N_POS).repeat(N_LAYERS)
    layer = torch.cat([feat_layer, err_layer, torch.full((N_POS,), -1), torch.full((N_LOGIT,), N_LAYERS)])
    pos = torch.cat([feat_pos, err_pos, torch.arange(N_POS), torch.full((N_LOGIT,), N_POS - 1)])
    is_feat_tgt = torch.arange(len(layer)) < n_feat
    is_logit = torch.arange(len(layer)) >= len(layer) - N_LOGIT
    allowed = ((is_feat_tgt | is_logit)[:, None] & (layer[:, None] > layer[None, :])
               & (pos[:, None] >= pos[None, :]) & ~is_logit[None, :])
    w = torch.from_numpy(rng.normal(size=allowed.shape) * (rng.random(allowed.shape) < density)).float()
    g.adjacency_matrix = w * allowed
    return g


def to_data(graph, label: int, rng: np.random.Generator, **kw):
    errs = torch.from_numpy(rng.random((N_LAYERS, N_POS))).float()
    return graph_data.graph_to_data(graph, label, errs, TOKEN_CLASSES, N_PCA, max_pos=MAX_POS, **kw)


def make_dataset(n_problems: int = 24, steps: int = 4, seed: int = 0) -> list:
    """``n_problems`` problem groups of ``steps`` graphs; incorrect steps (y=1) have higher activations."""
    rng = np.random.default_rng(seed)
    out = []
    for pid in range(n_problems):
        for s in range(steps):
            label = int(rng.random() < 0.5) if s else pid % 2  # both classes appear across problems
            d = to_data(make_graph(rng, shift=3.0 * label), label, rng)
            d.problem_id, d.step_idx = pid, s
            out.append(d)
    return out


def make_crv_data(i: int, label: int, rng) -> Data:
    """Hand-built graph: 3 feature, 2 error, 2 token, 1 logit nodes. Wrong steps are noisier."""
    types = torch.tensor([FEATURE] * 3 + [ERROR] * 2 + [TOKEN] * 2 + [LOGIT])
    n = len(types)
    src, dst = torch.randint(0, n - 1, (12,)), torch.randint(0, n, (12,))
    top = 0.9 - 0.4 * label + 0.05 * rng.standard_normal()
    a, b = int(rng.integers(10, 50)), int(rng.integers(10, 50))
    c = a + b + (int(rng.integers(1, 5)) if label else 0)
    return Data(
        x=torch.randn(n, 5), edge_index=torch.stack([src, dst]), edge_weight=torch.randn(12),
        node_type=types, layer_idx=torch.tensor([1, 5, 5, 2, 3, 0, 0, 27]), pos_idx=torch.zeros(n).long(),
        logit_probs=torch.tensor([[top, 0.05, 0.02, 0.01, 0.01]]), y=torch.tensor([float(label)]),
        problem_id=f"p{i // 2}", line=f"{a} + {b} = {c}", prev_line=f"{a} + 1 = {a + 1}",
        anchor_index=torch.tensor([n - 1]), w_scale=torch.tensor([2.0]),
    )


def make_crv_dataset(n: int = 60, seed: int = 0) -> list:
    """``n`` hand-built graphs, alternating labels, two per problem."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    return [make_crv_data(i, i % 2, rng) for i in range(n)]
