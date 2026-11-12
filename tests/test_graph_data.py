import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from gnn_interp import graph_data as gd
from gnn_interp.graph_data import ABLATIONS, ERROR, FEATURE, LOGIT, TOKEN
from tests.synthetic import N_LAYERS, N_LOGIT, N_PCA, N_POS, make_graph, to_data


@pytest.fixture
def graph_and_data():
    rng = np.random.default_rng(1)
    g = make_graph(rng)
    return g, to_data(g, 1, rng)


def test_shapes_and_types(graph_and_data):
    g, d = graph_and_data
    n = g.adjacency_matrix.shape[0]
    assert d.num_nodes == n and d.x.shape == (n, max(gd.type_dims(N_PCA)))
    counts = torch.bincount(d.node_type, minlength=4).tolist()
    assert counts == [12, N_LAYERS * N_POS, N_POS, N_LOGIT]
    assert d.edge_index.shape == (2, d.num_edges) and d.edge_weight.shape == (d.num_edges,)
    assert d.y.item() == 1 and d.logit_probs.shape == (1, gd.TOP_K)
    assert d.layer_idx.max() == N_LAYERS + 1 and (d.layer_idx[d.node_type == TOKEN] == 0).all()
    assert d.feat_key.shape == (12, 2)


def test_edge_direction_and_weight_scaling(graph_and_data):
    g, d = graph_and_data
    src, dst = d.edge_index
    adj = g.adjacency_matrix
    assert (adj[dst, src] != 0).all() and d.num_edges == (adj != 0).sum()  # column = source, row = target
    assert torch.allclose(d.edge_weight * d.w_scale, adj[dst, src])
    assert d.edge_weight.abs().max() == pytest.approx(1.0)
    assert (d.edge_weight.sign() == adj[dst, src].sign()).all()  # sign preserved


def test_anchor_and_feature_columns(graph_and_data):
    g, d = graph_and_data
    a = d.anchor_index
    assert a.shape == (1,) and d.node_type[a].item() == LOGIT and a.item() == d.num_nodes - N_LOGIT
    assert (d.x[d.node_type == LOGIT, :gd.TOP_K] == torch.eye(gd.TOP_K)[:N_LOGIT]).all()
    f = d.node_type == FEATURE
    assert torch.allclose(d.x[f, 0], g.activation_values)
    assert d.pos_idx.max() < 8 and (d.pos_idx[a] == 0).all()  # logits sit at the last position
    assert torch.allclose(d.logit_probs[0, :N_LOGIT], g.logit_probabilities)


@pytest.mark.needs_circuit_tracer
def test_node_influence_feature_matches_circuit_tracer(graph_and_data):
    compute_node_influence = pytest.importorskip("circuit_tracer.graph").compute_node_influence
    g, d = graph_and_data
    f = d.node_type == FEATURE
    w = torch.zeros(g.adjacency_matrix.shape[0])
    w[-N_LOGIT:] = g.logit_probabilities
    assert torch.allclose(d.x[f, 1], compute_node_influence(g.adjacency_matrix, w)[:12])


def test_masks_prune_nodes_and_edges():
    rng = np.random.default_rng(2)
    g = make_graph(rng)
    n = g.adjacency_matrix.shape[0]
    node_mask = torch.rand(n) < 0.6
    node_mask[-N_LOGIT:] = True
    edge_mask = torch.rand(n, n) < 0.5
    d = to_data(g, 0, rng, node_mask=node_mask, edge_mask=edge_mask)
    assert d.num_nodes == node_mask.sum() and d.node_type[d.anchor_index].item() == LOGIT
    keep = node_mask.nonzero().squeeze(1)
    expected = (g.adjacency_matrix * edge_mask)[keep][:, keep]
    assert d.num_edges == (expected != 0).sum()
    assert d.feat_key.shape[0] == (d.node_type == FEATURE).sum()


@pytest.mark.needs_circuit_tracer
def test_prune_with_circuit_tracer():
    pytest.importorskip("circuit_tracer")
    rng = np.random.default_rng(3)
    g = make_graph(rng)
    d = to_data(g, 0, rng, prune=True)
    assert d.num_nodes < g.adjacency_matrix.shape[0] and d.node_type[d.anchor_index].item() == LOGIT
    assert (d.node_type == TOKEN).sum() == N_POS  # prune_graph always keeps tokens and logits


def test_decoder_pca_fit_apply(dataset, fetch):
    pca = gd.DecoderPCA(n_components=3).fit(dataset, fetch)
    assert pca.components.shape == (3, 16) and pca.mean.shape == (16,)
    out = pca.apply(dataset[:4], fetch)
    for d0, d1 in zip(dataset[:4], out):
        f = d1.node_type == FEATURE
        assert (d0.x[f, 2:5] == 0).all() and d1.x[f, 2:5].abs().sum() > 0  # filled in on a copy
        assert torch.equal(d0.x[~f], d1.x[~f]) and torch.equal(d0.x[f, :2], d1.x[f, :2])
        assert (d1.x[f, 5:] == 0).all()
        key = tuple(d1.feat_key[0].tolist())
        v = fetch(key[0], [key[1]])[0]
        v = v / v.norm()
        assert torch.allclose(d1.x[f][0, 2:5], (v - pca.mean) @ pca.components.T, atol=1e-5)


def test_random_unit_evr_is_flat():
    evr = gd.random_unit_evr(2000, 64, 4)
    assert evr.shape == (4,) and ((evr > 1 / 64) & (evr < 1.6 / 64)).all()  # ~1/d plus sampling noise


def test_decoder_pca_random_baseline(dataset, fetch):
    pca = gd.DecoderPCA(3, random_baseline=True).fit(dataset, fetch)
    assert torch.allclose(pca.components @ pca.components.T, torch.eye(3), atol=1e-5)


def test_normalizer_fit_on_train_only(dataset):
    train, test = dataset[:32], dataset[32:]
    norm = gd.FeatureNormalizer().fit(train)
    tr = norm.apply(train)
    x, t = torch.cat([d.x for d in tr]), torch.cat([d.node_type for d in tr])
    for typ, cols in gd.CONTINUOUS.items():
        v = x[t == typ][:, cols]
        assert torch.allclose(v.mean(0), torch.zeros(len(cols)), atol=1e-4)
        assert torch.allclose(v.std(0, unbiased=False), torch.ones(len(cols)), atol=1e-3)
    # the test split is scaled with train statistics, so it is generally not exactly standardised
    xt = torch.cat([d.x for d in norm.apply(test)])[torch.cat([d.node_type for d in test]) == FEATURE][:, 0]
    mu, sd = norm.stats[FEATURE][1][0], norm.stats[FEATURE][2][0]
    raw = torch.cat([d.x for d in test])[torch.cat([d.node_type for d in test]) == FEATURE][:, 0]
    assert torch.allclose(xt, (raw - mu) / sd, atol=1e-5)
    # non-continuous columns and the inputs are untouched
    assert torch.equal(tr[0].x[tr[0].node_type == LOGIT], train[0].x[train[0].node_type == LOGIT])
    assert not torch.equal(tr[0].x, train[0].x)


@pytest.mark.parametrize("mode", ABLATIONS)
def test_ablations(dataset, mode):
    d = dataset[0]
    a = gd.ablate(d, mode)
    assert a is not d and d.num_edges == dataset[0].num_edges  # input untouched
    if mode == "shuffle_edges":
        assert a.num_edges == d.num_edges and torch.equal(a.edge_index[0], d.edge_index[0])
        assert not torch.equal(a.edge_index[1], d.edge_index[1])
        assert torch.equal(a.edge_weight, d.edge_weight)
        assert torch.equal(a.edge_index, gd.ablate(d, mode, seed=0).edge_index)
    elif mode == "binarize":
        assert set(a.edge_weight.tolist()) <= {-1.0, 1.0} and (a.edge_weight == d.edge_weight.sign()).all()
    elif mode == "drop_errors":
        n_err = (d.node_type == ERROR).sum().item()
        assert n_err > 0 and (a.node_type != ERROR).all() and a.num_nodes == d.num_nodes - n_err
        assert a.x.shape[0] == a.layer_idx.shape[0] == a.pos_idx.shape[0] == a.num_nodes
        assert a.edge_index.max() < a.num_nodes and a.node_type[a.anchor_index].item() == LOGIT
        assert a.edge_index.shape[1] == a.edge_weight.shape[0]
        # surviving edges are exactly those between non-error nodes
        keep = d.node_type[d.edge_index[0]].ne(ERROR) & d.node_type[d.edge_index[1]].ne(ERROR)
        assert a.num_edges == keep.sum()
    else:
        assert a.num_edges == 2 * d.num_edges
        assert torch.equal(a.edge_index[:, d.num_edges:], d.edge_index.flip(0))
        assert torch.equal(a.edge_weight[d.num_edges:], d.edge_weight)


def test_ablate_unknown_mode(dataset):
    with pytest.raises(ValueError):
        gd.ablate(dataset[0], "nope")


def test_batching_offsets_anchor_index(dataset):
    batch = Batch.from_data_list([gd.ablate(d, "drop_errors") for d in dataset[:5]])
    assert batch.anchor_index.shape == (5,)
    assert (batch.node_type[batch.anchor_index] == LOGIT).all()
    assert torch.equal(batch.batch[batch.anchor_index], torch.arange(5))
