"""CRV topology baselines: graph features on hand-built graphs."""
import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from gnn_interp.baselines import crv_features, topo_features
from gnn_interp.graph_data import FEATURE, LOGIT, TOKEN
from tests.synthetic import make_crv_data


def chain(edges, types, n_extra=0) -> Data:
    ei = torch.tensor(edges).T
    return Data(edge_index=ei, edge_weight=torch.ones(len(edges)), node_type=torch.tensor(types))


def test_topo_features_known_graph():
    # token0 -> feat1 -> feat2 -> logit3 ; token0 -> logit3 is absent; isolated feature node 4
    d = chain([[0, 1], [1, 2], [2, 3]], [TOKEN, FEATURE, FEATURE, LOGIT, FEATURE])
    dens, dmean, dmax, bmean, bmax, ncomp, avgpath, io = topo_features(d)
    assert dens == pytest.approx(3 / 20) and ncomp == 2 and io == 3
    assert dmax == pytest.approx(2 / 4) and dmean == pytest.approx(6 / 5 / 4)
    assert avgpath == pytest.approx((1 + 2 + 3 + 1 + 2 + 1) / 6)  # path graph on 4 nodes
    assert bmax > 0 and bmean > 0


def test_topo_features_no_path():
    d = chain([[3, 0], [1, 2]], [TOKEN, FEATURE, FEATURE, LOGIT])  # logit -> token only
    assert np.isnan(topo_features(d)[-1]) and topo_features(d)[5] == 2


def test_crv_features(crv_dataset):
    d = crv_dataset[0]
    with_l, without = crv_features(d), crv_features(d, with_logit_stats=False)
    assert len(with_l) == len(without) + 2 and np.isfinite(with_l[:-2]).all()
    assert with_l[0] == 3 and with_l[1] == 2  # feature / error node counts
    assert with_l[-2] == pytest.approx(d.logit_probs[0, 0].item())


def test_crv_handles_empty_graph():
    d = make_crv_data(0, 0, np.random.default_rng(0))
    d.edge_index, d.edge_weight = torch.zeros(2, 0).long(), torch.zeros(0)
    f = crv_features(d)
    assert np.isnan(f).sum() <= 2  # only avg path length / input->logit path may be undefined
