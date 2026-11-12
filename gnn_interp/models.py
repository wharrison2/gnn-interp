"""Graph classifiers over attribution graphs (outputs one logit per graph; positive = incorrect).

``conv`` options:
    graphconv -- Morris et al. GraphConv with signed edge weights, sum aggregation (main model).
    gine      -- GINEConv; edge features [w, layer gap].
    nnconv    -- NNConv: per-edge weight matrix generated from edge features [w, layer gap].
    sgc       -- linear GraphConv stack (no norm/nonlinearity). PyG's SGConv is avoided because
                 its GCN normalisation assumes non-negative weights. Still learned, so not the
                 parameter-free SGC + logistic-regression fallback of the spec.
    deepsets  -- node-wise linear layers, no message passing (tests whether structure matters).
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINEConv, GraphConv, NNConv, global_max_pool, global_mean_pool
from torch_geometric.utils import dropout_edge

from .graph_data import N_TYPES

CONVS = ("graphconv", "gine", "nnconv", "sgc", "deepsets")
READOUTS = ("anchor", "mean", "max", "all")
EDGE_DIM = 2  # [w, layer gap] for gine / nnconv


class NodeEncoder(nn.Module):
    """Type-specific linear projection of raw features plus type, layer and position embeddings."""

    def __init__(self, type_dims: list[int], hidden: int, n_model_layers: int, max_pos: int):
        super().__init__()
        self.dims = type_dims
        self.proj = nn.ModuleList(nn.Linear(d, hidden) for d in type_dims)
        self.type_emb = nn.Embedding(N_TYPES, hidden)
        self.layer_emb = nn.Embedding(n_model_layers + 2, hidden)  # embedding, layers, logits
        self.pos_emb = nn.Embedding(max_pos, hidden)

    def forward(self, data) -> torch.Tensor:
        h = data.x.new_zeros(data.num_nodes, self.type_emb.embedding_dim)
        for t, (lin, d) in enumerate(zip(self.proj, self.dims)):
            m = data.node_type == t
            if m.any():
                h[m] = lin(data.x[m, :d])
        return h + self.type_emb(data.node_type) + self.layer_emb(data.layer_idx) + self.pos_emb(data.pos_idx)


def _make_conv(kind: str, h: int) -> nn.Module:
    if kind in ("graphconv", "sgc"):
        return GraphConv(h, h, aggr="add")
    if kind == "gine":
        return GINEConv(nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)), edge_dim=EDGE_DIM)
    if kind == "nnconv":
        return NNConv(h, h, nn.Sequential(nn.Linear(EDGE_DIM, 16), nn.ReLU(), nn.Linear(16, h * h)), aggr="add")
    if kind == "deepsets":
        return nn.Linear(h, h)  # a GraphConv layer without the neighbour term
    raise ValueError(kind)


class GraphClassifier(nn.Module):
    """Encoder -> ``n_conv`` layers -> readout -> LayerNorm -> linear head.

    Each layer: conv -> LayerNorm -> ReLU -> dropout (``sgc``: conv only). ``readout`` is the
    top-1 logit node ("anchor"), mean, max, or the concatenation of all three ("all").
    """

    def __init__(self, type_dims: list[int], conv: str = "graphconv", hidden: int = 64, n_conv: int = 2,
                 dropout: float = 0.2, edge_dropout: float = 0.1, readout: str = "all",
                 n_model_layers: int = 26, max_pos: int = 64):
        super().__init__()
        assert conv in CONVS and readout in READOUTS, (conv, readout)
        self.kind, self.readout, self.edge_dropout = conv, readout, edge_dropout
        self.gap_scale = 1.0 / (n_model_layers + 1)
        self.encoder = NodeEncoder(type_dims, hidden, n_model_layers, max_pos)
        self.convs = nn.ModuleList(_make_conv(conv, hidden) for _ in range(n_conv))
        self.post = nn.ModuleList(
            nn.Identity() if conv == "sgc" else
            nn.Sequential(nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout)) for _ in range(n_conv))
        out = hidden * (3 if readout == "all" else 1)
        self.head = nn.Sequential(nn.LayerNorm(out), nn.Linear(out, 1))

    def _conv(self, conv, h, ei, w, data):
        if self.kind == "deepsets":
            return conv(h)
        if self.kind in ("graphconv", "sgc"):
            return conv(h, ei, w)
        gap = (data.layer_idx[ei[1]] - data.layer_idx[ei[0]]).to(w.dtype) * self.gap_scale
        return conv(h, ei, torch.stack([w, gap], -1))

    def forward(self, data) -> torch.Tensor:
        h = self.encoder(data)
        ei, w = data.edge_index, data.edge_weight
        if self.training and self.edge_dropout > 0:
            ei, keep = dropout_edge(ei, p=self.edge_dropout)
            w = w[keep]
        for conv, post in zip(self.convs, self.post):
            h = post(self._conv(conv, h, ei, w, data))

        batch = data.batch if data.batch is not None else h.new_zeros(h.shape[0], dtype=torch.long)
        pooled = {"anchor": lambda: h[data.anchor_index],
                  "mean": lambda: global_mean_pool(h, batch),
                  "max": lambda: global_max_pool(h, batch)}
        names = ("anchor", "mean", "max") if self.readout == "all" else (self.readout,)
        return self.head(torch.cat([pooled[k]() for k in names], -1)).squeeze(-1)
