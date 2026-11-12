import pytest
import torch
from torch_geometric.data import Batch

from gnn_interp import graph_data as gd
from gnn_interp.models import CONVS, READOUTS, GraphClassifier
from tests.synthetic import N_LAYERS


def build(model_kwargs, conv, readout="all", **kw):
    torch.manual_seed(0)
    return GraphClassifier(conv=conv, readout=readout, **model_kwargs, **kw)


@pytest.mark.parametrize("conv", CONVS)
@pytest.mark.parametrize("readout", READOUTS)
def test_forward_batched_and_single(dataset, model_kwargs, conv, readout):
    m = build(model_kwargs, conv, readout).eval()
    batch = Batch.from_data_list(dataset[:6])
    out = m(batch)
    assert out.shape == (6,) and torch.isfinite(out).all()
    single = m(dataset[0])
    assert single.shape == (1,)
    # batching must not leak across graphs: per-graph outputs match single-graph outputs
    # (this also verifies that anchor_index is offset by PyG batching)
    solo = torch.cat([m(d) for d in dataset[:6]])
    assert torch.allclose(out, solo, atol=1e-4)


@pytest.mark.parametrize("conv", CONVS)
def test_gradient_flow(dataset, model_kwargs, conv):
    m = build(model_kwargs, conv, dropout=0.0, edge_dropout=0.0).train()
    m(Batch.from_data_list(dataset[:6])).sum().backward()
    for name, p in m.named_parameters():  # every parameter is used (e.g. no idle LayerNorms for sgc)
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    for part in (m.encoder, m.convs, m.head):
        assert sum(p.grad.abs().sum() for p in part.parameters()) > 0


def test_deepsets_ignores_edges_graphconv_does_not(dataset, model_kwargs):
    batch = Batch.from_data_list(dataset[:4])
    shuffled = Batch.from_data_list([gd.ablate(d, "shuffle_edges") for d in dataset[:4]])
    ds, gc = build(model_kwargs, "deepsets").eval(), build(model_kwargs, "graphconv").eval()
    assert all(isinstance(c, torch.nn.Linear) for c in ds.convs)
    assert torch.allclose(ds(batch), ds(shuffled), atol=1e-5)
    assert not torch.allclose(gc(batch), gc(shuffled), atol=1e-4)


def test_edge_weight_sign_matters(dataset, model_kwargs):
    m = build(model_kwargs, "graphconv").eval()
    d = dataset[0].clone()
    flipped = d.clone()
    flipped.edge_weight = -d.edge_weight
    assert not torch.allclose(m(d), m(flipped), atol=1e-5)


def test_edge_dropout_only_in_training(dataset, model_kwargs):
    m = build(model_kwargs, "graphconv", edge_dropout=0.5, dropout=0.0)
    b = Batch.from_data_list(dataset[:4])
    assert torch.equal(m.eval()(b), m.eval()(b))
    assert not torch.allclose(m.train()(b), m.train()(b))


def test_invalid_options(model_kwargs):
    with pytest.raises(AssertionError):
        GraphClassifier(conv="bogus", **model_kwargs)
    with pytest.raises(AssertionError):
        GraphClassifier(readout="bogus", **model_kwargs)


@pytest.mark.parametrize("conv", ["gine", "nnconv"])
def test_edge_features_use_layer_gap(dataset, model_kwargs, conv):
    m = build(model_kwargs, conv).eval()
    with torch.no_grad():  # identical layer embeddings, so only the edge layer gaps differ
        m.encoder.layer_emb.weight[:] = m.encoder.layer_emb.weight[0]
    d = dataset[0].clone()
    shifted = d.clone()
    shifted.layer_idx = d.layer_idx.flip(0)
    assert not torch.allclose(m(d), m(shifted), atol=1e-5)


@pytest.mark.parametrize("conv", CONVS)
def test_forward_on_device(dataset, model_kwargs, device, conv):
    """Runs on cpu always, on cuda when available (marker ``gpu``)."""
    m = build(model_kwargs, conv).to(device).eval()
    out = m(Batch.from_data_list(dataset[:4]).to(device))
    assert out.device.type == device.type and torch.isfinite(out).all()
