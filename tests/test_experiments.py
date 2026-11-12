"""Experiment helpers on synthetic data (the model-dependent parts run only on a GPU VM)."""
import numpy as np
import pytest

from gnn_interp import experiments
from tests.synthetic import N_PCA, make_graph


def _row(pid, level, step_idx, label, value=1, local_ok=True, global_ok=True):
    return {"problem_id": pid, "level": level, "step_idx": step_idx, "label": label, "value": value,
            "local_ok": local_ok, "global_ok": global_ok, "pos": 10}


def test_summarize_balance():
    rows = [_row("a", "easy", 0, 0), _row("a", "easy", 1, 1, local_ok=False, global_ok=False),
            _row("a", "easy", 2, None, global_ok=False),  # after the first error: unlabelled
            _row("b", "easy", 0, None, value=None, local_ok=False, global_ok=False), _row("c", "easy", None, None)]
    s = experiments.summarize_balance(rows)["easy"]
    assert s["problems"] == 3 and s["problems_without_steps"] == 1
    assert s["steps"] == 4 and s["unparseable_steps"] == 1 and s["labelled_steps"] == 2
    assert s["incorrect_steps"] == 1 and s["error_rate"] == pytest.approx(1 / 2)
    assert s["local_error_rate"] == pytest.approx(1 / 3) and s["final_answer_acc"] == 0


def test_sample_steps_balances_classes():
    rows = [_row(str(i), "easy", 0, int(i < 3)) for i in range(20)] + [_row("x", "easy", 0, None)]
    picked = experiments.sample_steps(rows, 10)
    assert len(picked) == 10 and sum(r["label"] for r in picked) == 3
    assert all(r["label"] is not None for r in picked)


@pytest.mark.needs_circuit_tracer
def test_graph_stats_counts():
    pytest.importorskip("circuit_tracer")
    g = make_graph(np.random.default_rng(0))
    s = experiments.graph_stats(g)
    assert s["n_nodes"] == g.adjacency_matrix.shape[0] and s["nnz"] == int((g.adjacency_matrix != 0).sum())
    loose, default = s["pruned"]["0.95/0.995"], s["pruned"]["0.8/0.98"]
    for p in (loose, default):
        assert p["feature"] + p["error"] + p["token"] + p["logit"] == p["nodes"]
        assert p["token"] == g.n_pos and p["edges"] <= p["edges_node_pruned"]
    assert default["nodes"] <= loose["nodes"]


@pytest.mark.slow
def test_gnn_timing_and_cv_dry_run(dataset):
    t = experiments.gnn_epoch_timing(dataset, n_target=40, device="cpu", n_pca=N_PCA)
    assert t["graphconv"]["epoch_seconds"] > 0 and t["deepsets"]["epoch_seconds"] > 0
    r = experiments.cv_dry_run(dataset, n_folds=2, seeds=[0], device="cpu", n_pca=N_PCA)
    assert r["epochs_run"]["max"] >= 1 and 0 <= r["metrics"]["auroc"][0] <= 1


@pytest.mark.needs_circuit_tracer
def test_balance_and_sizes_plumbing(tmp_path, monkeypatch):
    """``class_balance`` -> ``graph_sizes`` -> ``data.pt`` with generation and attribution faked."""
    pytest.importorskip("circuit_tracer")
    import torch
    from types import SimpleNamespace

    from gnn_interp import attribution, gemma, problems
    from tests.synthetic import N_LAYERS, N_POS, VOCAB

    rng = np.random.default_rng(0)
    lines = [("= 5 + 1", 0), ("= 7", 1)]  # second step wrong for every problem with answer 6

    def fake_run_problems(model, prompts, out_dir, max_new_tokens):
        (pid, _), = prompts.items()
        steps = [{"pos": N_POS, "line": l, "prev_line": "2 + 3 + 1" if i == 0 else lines[0][0],
                  "error_norms": torch.rand(N_LAYERS, N_POS)} for i, (l, _) in enumerate(lines)]
        rec = {"problem_id": pid, "tokens": torch.zeros(N_POS + 1, dtype=torch.long), "n_prompt": 1, "steps": steps}
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(rec, out_dir / f"{pid}.pt")
        return [rec]

    def fake_build_graph(model, tokens, pos, max_feature_nodes, **kw):
        g = make_graph(rng, n_feat=min(max_feature_nodes, 12))
        g.logit_token_ids = torch.arange(len(g.logit_targets))
        return g

    monkeypatch.setattr(gemma, "run_problems", fake_run_problems)
    monkeypatch.setattr(gemma, "token_class_table", lambda tok: torch.arange(VOCAB) % 3)
    monkeypatch.setattr(attribution, "build_graph", fake_build_graph)
    monkeypatch.setattr(problems, "generate_problems", lambda n, seed, **kw: [
        problems.Problem(f"p{i}", "2 + 3 + 1", 6, 2, "") for i in range(n)])
    model = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids: "= 5 + 1\n= 7\n\n"))

    bal = experiments.class_balance(model, tmp_path / "bal", 3, levels=["easy"])
    assert bal["easy"]["labelled_steps"] == 6 and bal["easy"]["incorrect_steps"] == 3
    sizes = experiments.graph_sizes(model, tmp_path / "bal", tmp_path / "sizes", 4,
                                    max_feature_nodes=(12, 6))
    assert sizes["n_graphs"] == 4 and "jaccard_vs_ref" in sizes["6"]
    data = torch.load(tmp_path / "sizes" / "data.pt", weights_only=False)
    assert len(data) == 4 and sum(int(d.y.item()) for d in data) == 2
    # rerun resumes: nothing new is attributed
    assert experiments.graph_sizes(model, tmp_path / "bal", tmp_path / "sizes", 4,
                                   max_feature_nodes=(12, 6))["n_graphs"] == 4
