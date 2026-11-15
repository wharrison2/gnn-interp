import numpy as np
import pytest
import torch

pytest.importorskip("circuit_tracer")
from circuit_tracer.graph import compute_node_influence, find_threshold, normalize_matrix, prune_graph  # noqa: E402

from gnn_interp import attribution  # noqa: E402
from gnn_interp import graph_data as gd  # noqa: E402
from tests.synthetic import N_LAYERS, N_POS, TOKEN_CLASSES, make_layered_graph  # noqa: E402

pytestmark = pytest.mark.needs_circuit_tracer
NODE_TS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
EDGE_TS = (0.5, 0.8, 0.9, 0.95, 0.98, 0.99)


def reference_prune(g, t, e):
    """circuit-tracer's ``prune_graph``, verbatim except edges are scored on the full graph."""
    A = g.adjacency_matrix
    n_tok, n_log, n_feat = len(g.input_tokens), len(g.logit_targets), len(g.selected_features)
    lw = torch.zeros(A.shape[0])
    lw[-n_log:] = g.logit_probabilities
    infl = compute_node_influence(A, lw)
    node_mask = infl >= find_threshold(infl, t)
    node_mask[-n_log - n_tok:] = True
    scores = normalize_matrix(A.double()) * (infl.double() + lw.double())[:, None]  # full-graph scoring
    scores[~node_mask] = 0
    scores[:, ~node_mask] = 0
    edge_mask = scores >= find_threshold(scores.flatten(), e)
    old = node_mask.clone()
    node_mask[: -n_log - n_tok] &= edge_mask[:, : -n_log - n_tok].any(0)
    node_mask[:n_feat] &= edge_mask[:n_feat].any(1)
    while not torch.all(node_mask == old):
        old[:] = node_mask
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False
        node_mask[: -n_log - n_tok] &= edge_mask[:, : -n_log - n_tok].any(0)
        node_mask[:n_feat] &= edge_mask[:n_feat].any(1)
    return node_mask, edge_mask & (A != 0)


def to_global(sp, node_mask, edge_mask):
    gi, (src, dst), n = sp["node_index"].long(), sp["edge_index"].long(), sp["n_nodes"]
    nodes, edges = torch.zeros(n, dtype=torch.bool), torch.zeros(n, n, dtype=torch.bool)
    nodes[gi] = node_mask
    edges[gi[dst[edge_mask]], gi[src[edge_mask]]] = True
    return nodes, edges


@pytest.fixture(params=range(4))
def graph(request):
    return make_layered_graph(np.random.default_rng(request.param))


def test_reprune_matches_reference(graph):
    sp = attribution.sparsify(graph)
    for t in NODE_TS:
        for e in EDGE_TS:
            ref_nodes, ref_edges = reference_prune(graph, t, e)
            nodes, edges = to_global(sp, *attribution.reprune(sp, t, e))
            assert torch.equal(nodes, ref_nodes), (t, e)
            assert torch.equal(edges, ref_edges), (t, e)


def test_prune_matches_reference(graph):
    masks = attribution.prune(graph, 0.8, 0.98)
    ref_nodes, ref_edges = reference_prune(graph, 0.8, 0.98)
    assert torch.equal(masks.node_mask, ref_nodes) and torch.equal(masks.edge_mask, ref_edges)


def test_node_set_matches_circuit_tracer(graph):
    """Before the edge step, kept nodes are exactly prune_graph's (only edge scoring differs)."""
    sp = attribution.sparsify(graph, 0.8, 0.98, 0.8)
    A = graph.adjacency_matrix
    lw = torch.zeros(A.shape[0])
    lw[-len(graph.logit_targets):] = graph.logit_probabilities
    infl = compute_node_influence(A, lw)
    expected = (infl >= find_threshold(infl, 0.8)).nonzero().squeeze(1)
    fixed = len(graph.input_tokens) + len(graph.logit_targets)
    expected = torch.unique(torch.cat([expected, torch.arange(A.shape[0] - fixed, A.shape[0])]))
    assert torch.equal(sp["node_index"].long(), expected)
    # final masks stay close to circuit-tracer's (renormalised) edge scoring
    ct_nodes = prune_graph(graph, 0.8, 0.98).node_mask
    ours = attribution.prune(graph, 0.8, 0.98).node_mask
    assert (ct_nodes & ours).sum() / (ct_nodes | ours).sum() > 0.5


def test_sparse_storage_is_smaller(graph):
    sp = attribution.sparsify(graph)
    gi = sp["node_index"].long()
    all_edges = int((graph.adjacency_matrix[gi][:, gi] != 0).sum())
    assert 0 < sp["edge_index"].shape[1] < all_edges
    assert sp["edge_index"].dtype == torch.int16 and len(sp["edge_mass"]) <= len(gi) + 1


def test_out_of_range_raises(graph):
    sp = attribution.sparsify(graph, 0.9, 0.98, 0.6)
    for t, e in [(0.95, 0.9), (0.5, 0.9), (0.8, 0.99)]:
        with pytest.raises(ValueError):
            attribution.reprune(sp, t, e)


def test_compact_roundtrip_matches_graph_to_data(graph, tmp_path):
    path = tmp_path / "g.pt"
    attribution.save_compact(graph, path)
    sp = torch.load(path, weights_only=False)
    errs = torch.rand(N_LAYERS, N_POS)
    for t, e in [(0.8, 0.98), (0.6, 0.9)]:
        a = gd.graph_to_data(graph, 1, errs, TOKEN_CLASSES, 4, *attribution.prune(graph, t, e))
        b = gd.compact_to_data(sp, 1, errs, TOKEN_CLASSES, 4, t, e)
        for key in ("x", "edge_index", "edge_weight", "w_scale", "node_type", "layer_idx", "pos_idx",
                    "anchor_index", "feat_key", "logit_probs", "y"):
            assert torch.allclose(a[key].float(), b[key].float()), key


def test_load_dataset_reads_compact_rows(graph, tmp_path):
    import json

    attribution.save_compact(graph, tmp_path / "g.pt")
    step = {"error_norms": torch.rand(N_LAYERS, N_POS), "line": "= 5", "prev_line": ""}
    torch.save({"steps": [step]}, tmp_path / "p0.pt")
    row = {"graph_path": "g.pt", "mask_path": None, "problem_id": "p0", "step_idx": 0, "label": 1,
           "storage": "compact"}
    (tmp_path / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    d, = gd.load_dataset(tmp_path / "manifest.jsonl", tmp_path, TOKEN_CLASSES, 4, node_threshold=0.7)
    ref = gd.graph_to_data(graph, 1, step["error_norms"], TOKEN_CLASSES, 4, *attribution.prune(graph, 0.7, 0.98))
    assert torch.equal(d.edge_index, ref.edge_index) and torch.allclose(d.x, ref.x) and d.line == "= 5"
