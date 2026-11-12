"""Convert circuit-tracer attribution ``Graph``s into PyG ``Data`` objects.

Node order in a circuit-tracer adjacency matrix (rows = targets, columns = sources):
selected features | error nodes (layer-major, n_layers x n_pos) | input tokens | logits (top-1 first).

Per-type raw feature columns of ``x`` (zero-padded to a common width):
    FEATURE: [activation, influence, decoder-PCA (n_pca, filled later by ``DecoderPCA``)]
    ERROR:   [influence, ||error||]
    TOKEN:   [influence, coarse token-class one-hot (digit / operator / other)]
    LOGIT:   [logit-rank one-hot (TOP_K)]
Token identity is never a feature; the operands reach the GNN only through the graph structure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

FEATURE, ERROR, TOKEN, LOGIT = 0, 1, 2, 3
N_TYPES, N_TOKEN_CLASSES, TOP_K = 4, 3, 5
CONTINUOUS = {FEATURE: [0, 1], ERROR: [0, 1], TOKEN: [0]}  # columns z-scored by FeatureNormalizer


def type_dims(n_pca: int) -> list[int]:
    """Raw input width for each node type, indexed by type id."""
    return [2 + n_pca, 2, 1 + N_TOKEN_CLASSES, TOP_K]


def _node_influence(adj: torch.Tensor, logit_probs: torch.Tensor) -> torch.Tensor:
    """circuit-tracer's node influence on the logits (as used by ``prune_graph``), on CPU."""
    from circuit_tracer.graph import compute_node_influence

    weights = torch.zeros(adj.shape[0])
    weights[-len(logit_probs):] = logit_probs
    return compute_node_influence(adj, weights)


def graph_to_data(
    graph,
    label: int,
    error_norms: torch.Tensor,
    token_classes: torch.Tensor,
    n_pca: int = 8,
    node_mask: torch.Tensor | None = None,
    edge_mask: torch.Tensor | None = None,
    prune: bool = False,
    max_pos: int = 64,
) -> Data:
    """Build a PyG ``Data`` from a circuit-tracer ``Graph``.

    Args:
        label: 1 if the step is incorrect, else 0.
        error_norms: (n_layers, n_pos) transcoder error norms for the graph's input.
        token_classes: vocab-sized table from ``gemma.token_class_table``.
        node_mask, edge_mask: a saved ``PruneResult``'s masks. If absent, ``prune=True`` prunes
            here with circuit-tracer defaults; otherwise every node and edge is kept.
        max_pos: positions are counted back from the last token and clamped to ``max_pos - 1``.

    Extra attributes: ``node_type``, ``layer_idx`` (0 = embedding, l+1 = layer l, L+1 = logits),
    ``pos_idx``, ``edge_weight`` (scaled to max |w| = 1; scale kept in ``w_scale``),
    ``anchor_index`` (top-1 logit node), ``feat_key`` ((layer, feature id) per feature node, in
    node order), ``logit_probs`` (1, TOP_K) and ``y``.
    """
    adj = graph.adjacency_matrix.float().cpu()
    n = adj.shape[0]
    n_layers, n_pos = graph.cfg.n_layers, graph.n_pos
    n_feat, n_logit = len(graph.selected_features), len(graph.logit_targets)
    n_err = n_layers * n_pos
    assert tuple(error_norms.shape) == (n_layers, n_pos), (error_norms.shape, n_layers, n_pos)
    assert n == n_feat + n_err + n_pos + n_logit

    if node_mask is None and prune:
        from circuit_tracer.graph import prune_graph

        node_mask, edge_mask, _ = prune_graph(graph)
    node_mask = torch.ones(n, dtype=torch.bool) if node_mask is None else node_mask.cpu().bool()
    logit_probs = graph.logit_probabilities.float().cpu()
    influence = _node_influence(adj, logit_probs)

    # Per-node metadata over the full node set.
    ntype = torch.cat([torch.full((k,), t) for k, t in
                       [(n_feat, FEATURE), (n_err, ERROR), (n_pos, TOKEN), (n_logit, LOGIT)]])
    feats = graph.active_features[graph.selected_features].cpu().long()  # (n_feat, 3): layer, pos, id
    acts = graph.activation_values.float().cpu()[graph.selected_features.cpu()]  # indexed like active_features
    err_layer = torch.arange(n_layers).repeat_interleave(n_pos)
    err_pos = torch.arange(n_pos).repeat(n_layers)
    layer = torch.cat([feats[:, 0] + 1, err_layer + 1, torch.zeros(n_pos, dtype=torch.long),
                       torch.full((n_logit,), n_layers + 1)])
    pos = torch.cat([feats[:, 1], err_pos, torch.arange(n_pos), torch.full((n_logit,), n_pos - 1)])

    x = torch.zeros(n, max(type_dims(n_pca)))
    f, e, t, lg = (ntype == k for k in range(N_TYPES))
    x[f, 0], x[f, 1] = acts, influence[f]
    x[e, 0], x[e, 1] = influence[e], error_norms.float().flatten()
    x[t, 0] = influence[t]
    x[t, 1:1 + N_TOKEN_CLASSES] = torch.eye(N_TOKEN_CLASSES)[token_classes[graph.input_tokens.cpu().long()]]
    x[lg, :TOP_K] = torch.eye(TOP_K)[torch.arange(n_logit).clamp(max=TOP_K - 1)]

    # Restrict to kept nodes and edges; edges point source (column) -> target (row).
    keep = node_mask.nonzero().squeeze(1)
    assert node_mask[n - n_logit], "the top-1 logit node (readout anchor) must be kept"
    w = adj[keep][:, keep]
    if edge_mask is not None:
        w = w * edge_mask.cpu().bool()[keep][:, keep]
    dst, src = w.nonzero(as_tuple=True)
    weight = w[dst, src]
    scale = weight.abs().max().item() if weight.numel() else 1.0
    probs = torch.zeros(1, TOP_K)
    probs[0, : min(n_logit, TOP_K)] = logit_probs[:TOP_K]

    kept_feat = keep[keep < n_feat]
    return Data(
        x=x[keep],
        edge_index=torch.stack([src, dst]),
        edge_weight=weight / scale,
        w_scale=torch.tensor([scale]),
        node_type=ntype[keep],
        layer_idx=layer[keep],
        pos_idx=(n_pos - 1 - pos[keep]).clamp(0, max_pos - 1),
        anchor_index=(keep == n - n_logit).nonzero().view(1),
        feat_key=torch.stack([feats[kept_feat, 0], feats[kept_feat, 2]], 1),
        logit_probs=probs,
        y=torch.tensor([float(label)]),
    )


# --------------------------------------------------------------------------------------------
# Decoder-vector PCA (fit on the training split only)
# --------------------------------------------------------------------------------------------

def decoder_fetcher(transcoders):
    """Return ``fetch(layer, feat_ids) -> (k, d_model)`` decoder rows from a ``TranscoderSet``."""
    return lambda layer, ids: transcoders._get_decoder_vectors(int(layer), ids).float().cpu()


def _fetch_unit(fetch, keys: np.ndarray) -> dict[tuple[int, int], torch.Tensor]:
    """Fetch unit-normalised decoder rows for unique (layer, feat_id) ``keys``, batched by layer."""
    out = {}
    for layer in np.unique(keys[:, 0]):
        ids = np.unique(keys[keys[:, 0] == layer, 1])
        vecs = fetch(layer, torch.from_numpy(ids))
        vecs = vecs / vecs.norm(dim=1, keepdim=True).clamp_min(1e-8)
        out.update({(int(layer), int(i)): v for i, v in zip(ids, vecs)})
    return out


def _unique_keys(data_list) -> np.ndarray:
    keys = torch.cat([d.feat_key for d in data_list]) if data_list else torch.zeros(0, 2)
    return np.unique(keys.long().numpy(), axis=0)


class DecoderPCA:
    """Project feature nodes' unit decoder vectors onto ``n_components`` directions.

    Fit on the distinct (layer, feature) pairs of the training graphs, each counted once.
    ``random_baseline=True`` uses random orthonormal directions instead of PCA (control for
    whether the learned directions matter). Before relying on the PCA, compare
    ``explained_variance_ratio`` with ``random_unit_evr(n_fit, d_model, k)``.
    """

    def __init__(self, n_components: int = 8, random_baseline: bool = False):
        self.k, self.random_baseline = n_components, random_baseline
        self.components: torch.Tensor | None = None  # (k, d_model)
        self.mean: torch.Tensor | None = None
        self.explained_variance_ratio: np.ndarray | None = None
        self.n_fit = 0  # number of decoder vectors the fit used

    def fit(self, data_list, fetch, n_sample: int = 20000, seed: int = 0) -> "DecoderPCA":
        rng = np.random.default_rng(seed)
        keys = _unique_keys(data_list)
        keys = keys[rng.choice(len(keys), min(n_sample, len(keys)), replace=False)]
        X = torch.stack(list(_fetch_unit(fetch, keys).values()))
        self.n_fit = len(X)
        if self.random_baseline:
            g = torch.Generator().manual_seed(seed)
            q, _ = torch.linalg.qr(torch.randn(X.shape[1], self.k, generator=g))
            self.components, self.mean = q.T, torch.zeros(X.shape[1])
        else:
            from sklearn.decomposition import PCA

            pca = PCA(self.k, random_state=seed).fit(X.numpy())
            self.components = torch.from_numpy(pca.components_).float()
            self.mean = torch.from_numpy(pca.mean_).float()
            self.explained_variance_ratio = pca.explained_variance_ratio_
        return self

    def apply(self, data_list, fetch) -> list[Data]:
        """Return copies of ``data_list`` with PCA columns of feature nodes filled in."""
        vecs = _fetch_unit(fetch, _unique_keys(data_list))
        proj = {key: (v - self.mean) @ self.components.T for key, v in vecs.items()}
        out = []
        for d in data_list:
            d = d.clone()
            rows = (d.node_type == FEATURE).nonzero().squeeze(1)
            if len(rows):
                d.x[rows, 2: 2 + self.k] = torch.stack([proj[tuple(k)] for k in d.feat_key.tolist()])
            out.append(d)
        return out


def random_unit_evr(n: int, d: int, k: int, seed: int = 0) -> np.ndarray:
    """Explained-variance ratios of a ``k``-component PCA on ``n`` random unit vectors in R^d.

    Null curve for ``DecoderPCA.explained_variance_ratio``: if they match, the decoder vectors have
    no low-dimensional structure and the PCA columns carry almost nothing.
    """
    from sklearn.decomposition import PCA

    X = np.random.default_rng(seed).standard_normal((n, d))
    return PCA(k, random_state=seed).fit(X / np.linalg.norm(X, axis=1, keepdims=True)).explained_variance_ratio_


# --------------------------------------------------------------------------------------------
# Normalisation and ablations
# --------------------------------------------------------------------------------------------

class FeatureNormalizer:
    """Per-node-type z-scoring of the ``CONTINUOUS`` columns, fit on training graphs only."""

    def fit(self, data_list) -> "FeatureNormalizer":
        x, t = torch.cat([d.x for d in data_list]), torch.cat([d.node_type for d in data_list])
        self.stats = {}
        for typ, cols in CONTINUOUS.items():
            v = x[t == typ][:, cols]
            if len(v):
                self.stats[typ] = (cols, v.mean(0), v.std(0, unbiased=False).clamp_min(1e-6))
        return self

    def apply(self, data_list) -> list[Data]:
        out = []
        for d in data_list:
            d = d.clone()
            for typ, (cols, mu, sd) in self.stats.items():
                m = d.node_type == typ
                d.x[m.nonzero(), torch.tensor(cols)] = (d.x[m][:, cols] - mu) / sd
            out.append(d)
        return out


ABLATIONS = ("shuffle_edges", "binarize", "drop_errors", "bidirectional")


def ablate(data: Data, mode: str, seed: int = 0) -> Data:
    """Structural ablation of one graph (see ``ABLATIONS``)."""
    d = data.clone()
    if mode == "shuffle_edges":  # keep sources and weights, randomise targets
        g = torch.Generator().manual_seed(seed)
        d.edge_index = torch.stack([d.edge_index[0], torch.randint(d.num_nodes, (d.num_edges,), generator=g)])
    elif mode == "binarize":
        d.edge_weight = d.edge_weight.sign()
    elif mode == "drop_errors":
        keep = d.node_type != ERROR
        d.edge_index, d.edge_weight = subgraph(keep, d.edge_index, d.edge_weight,
                                               relabel_nodes=True, num_nodes=d.num_nodes)
        new_id = keep.long().cumsum(0) - 1
        d.anchor_index = new_id[d.anchor_index]
        for key in ("x", "node_type", "layer_idx", "pos_idx"):
            d[key] = d[key][keep]
    elif mode == "bidirectional":
        d.edge_index = torch.cat([d.edge_index, d.edge_index.flip(0)], 1)
        d.edge_weight = torch.cat([d.edge_weight, d.edge_weight])
    else:
        raise ValueError(f"unknown ablation {mode!r}; choose from {ABLATIONS}")
    return d


# --------------------------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------------------------

def load_dataset(manifest: str | Path, records_dir: str | Path, token_classes: torch.Tensor,
                 n_pca: int = 8, prune: bool = False) -> list[Data]:
    """Load every graph listed in a manifest (jsonl; see INTERFACE.md).

    Rows: ``{graph_path, problem_id, step_idx, label[, mask_path]}``, paths relative to the
    manifest. ``mask_path`` holds ``{"node_mask", "edge_mask"}``. Error norms and the step text
    come from ``records_dir/{problem_id}.pt`` (written by ``gemma.run_problems``).
    """
    from circuit_tracer.graph import Graph

    manifest, records_dir = Path(manifest), Path(records_dir)
    records, out = {}, []
    for row in map(json.loads, manifest.read_text().splitlines()):
        pid = row["problem_id"]
        if pid not in records:
            records[pid] = torch.load(records_dir / f"{pid}.pt", weights_only=False)
        step = records[pid]["steps"][row["step_idx"]]
        masks = (torch.load(manifest.parent / row["mask_path"], weights_only=False)
                 if row.get("mask_path") else {})
        d = graph_to_data(Graph.from_pt(str(manifest.parent / row["graph_path"])), row["label"],
                          step["error_norms"], token_classes, n_pca,
                          masks.get("node_mask"), masks.get("edge_mask"), prune)
        d.problem_id, d.step_idx = pid, row["step_idx"]
        d.line, d.prev_line = step["line"], step.get("prev_line", "")
        out.append(d)
    return out
