"""Cheap calibration experiments that size the real run: memory, class balance, attribution cost, GNN time.

Run from the repo root, e.g. on a GPU VM or Colab::

    python -m gnn_interp.experiments memory  --out runs/memory                       # 1. memory fit
    python -m gnn_interp.experiments balance --out runs/balance --n 70               # 3. class balance
    python -m gnn_interp.experiments sizes   --out runs/sizes --steps runs/balance --n 100  # 2. timing + size
    python -m gnn_interp.experiments gnn     --out runs/gnn --data runs/sizes/data.pt      # 4. GNN epoch time
    python -m gnn_interp.experiments cv      --out runs/cv --data runs/sizes/data.pt       # 5. CV dry run

``balance`` runs before ``sizes`` because it generates and labels the steps that ``sizes`` attributes.
Problems, grading and graph building are Ananth's (``problems``, ``attribution``); this module only measures.
Each command writes a ``summary.json``; ``balance`` and ``sizes`` also write per-row ``.jsonl`` files and
skip rows already written, so a preempted spot VM can rerun the same command.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from . import attribution, gemma, problems
from .graph_data import graph_to_data, type_dims

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
THRESHOLDS = [(0.6, 0.9), (0.8, 0.98), (0.95, 0.995)]  # (node, edge); the middle one is circuit-tracer's default
DEFAULT_THRESHOLD = (0.8, 0.98)
# difficulty levels: kwargs for ``problems.generate_problems`` (tune after the first ``balance`` run)
LEVELS = {"easy": dict(n_ops=2, operand_range=(1, 20), max_abs=999),
          "medium": dict(n_ops=3, operand_range=(1, 50), max_abs=9999),
          "hard": dict(n_ops=4, operand_range=(1, 99), max_abs=99999)}


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------

def _cuda() -> bool:
    return torch.cuda.is_available()


def _reset_peak():
    if _cuda():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _peak_gb() -> float:
    if not _cuda():
        return float("nan")
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def _free():
    gc.collect()
    if _cuda():
        torch.cuda.empty_cache()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _append_jsonl(path: Path, row: dict):
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=float))
    print(json.dumps(obj, indent=2, default=float))


def _quantiles(values) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    return {"mean": v.mean(), "p50": np.median(v), "p90": np.quantile(v, 0.9), "max": v.max()}


# --------------------------------------------------------------------------------------------
# 1. Memory fit
# --------------------------------------------------------------------------------------------

def memory_fit(configs=(("float32", False), ("bfloat16", False), ("bfloat16", True)),
               max_feature_nodes: int = 4096, batch_size: int = 256) -> list[dict]:
    """Peak GPU memory to load the model and build one attribution graph, per (dtype, lazy_encoder).

    The prompt is a medium problem with its reference solution, attributed at its final token. An
    out-of-memory error is recorded instead of raised.
    """
    p = problems.generate_problems(1, **LEVELS["medium"])[0]
    text = p.prompt + "\n".join(problems.solve_steps(p.expression))
    rows = []
    for dtype, lazy in configs:
        row = {"dtype": dtype, "lazy_encoder": lazy, "max_feature_nodes": max_feature_nodes}
        model = None
        try:
            _reset_peak()
            model = gemma.load_model("cuda" if _cuda() else None, DTYPES[dtype], lazy_encoder=lazy)
            row["load_peak_gb"] = _peak_gb()
            tokens = model.ensure_tokenized(text)
            row["n_pos"] = len(tokens)
            _reset_peak()
            t = time.perf_counter()
            attribution.build_graph(model, tokens, len(tokens), max_feature_nodes=max_feature_nodes,
                                    batch_size=batch_size)
            row["attr_peak_gb"], row["attr_seconds"] = _peak_gb(), time.perf_counter() - t
        except torch.OutOfMemoryError as e:
            row["oom"] = str(e).splitlines()[0]
        del model
        _free()
        rows.append(row)
        print(row)
    return rows


# --------------------------------------------------------------------------------------------
# 3. Class balance (generation + labels; no attribution)
# --------------------------------------------------------------------------------------------

def class_balance(model, out_dir: Path, n_per_level: int, levels=tuple(LEVELS),
                  seed: int = 0, max_new_tokens: int = 128) -> dict:
    """Generate ``n_per_level`` problems per level, grade every step, and summarise error rates.

    Writes ``records/{problem_id}.pt`` (``gemma.run_problems``) and ``steps.jsonl`` (one row per step:
    the ``problems.StepGrade`` fields plus level and ``pos``; a problem without steps gets one row with
    ``step_idx=None``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_dir, steps_path = out_dir / "records", out_dir / "steps.jsonl"
    done = {r["problem_id"] for r in _read_jsonl(steps_path)}
    for level in levels:
        for p in problems.generate_problems(n_per_level, seed=seed, **LEVELS[level]):
            if p.problem_id in done:
                continue
            rec = gemma.run_problems(model, {p.problem_id: p.prompt}, rec_dir, max_new_tokens)[0]
            grades = attribution.label_record(rec, p, model.tokenizer)
            rows = [{**asdict(g), "pos": s["pos"]} for g, s in zip(grades, rec["steps"])] or [{"step_idx": None}]
            for r in rows:
                _append_jsonl(steps_path, {"problem_id": p.problem_id, "level": level, **r})
    return summarize_balance(_read_jsonl(steps_path))


def summarize_balance(rows: list[dict]) -> dict:
    """Per level: problems, steps, labelled steps (up to the first error), incorrect count and rates.

    ``error_rate`` = incorrect / labelled (the dataset's class balance); ``local_error_rate`` = parsed
    steps whose value differs from the previous line's (slips anywhere, not only the first).
    """
    out = {}
    for level in sorted({r["level"] for r in rows}):
        lv = [r for r in rows if r["level"] == level]
        steps = [r for r in lv if r["step_idx"] is not None]
        parsed = [r for r in steps if r["value"] is not None]
        labelled = [r for r in steps if r["label"] is not None]
        pids = {r["problem_id"] for r in lv}
        last = {r["problem_id"]: r for r in steps}  # rows are in step order
        n_bad = sum(r["label"] for r in labelled)
        out[level] = {
            "problems": len(pids),
            "problems_without_steps": len(pids - set(last)),
            "steps": len(steps),
            "unparseable_steps": len(steps) - len(parsed),
            "labelled_steps": len(labelled),
            "incorrect_steps": n_bad,
            "error_rate": n_bad / max(len(labelled), 1),
            "local_error_rate": float(np.mean([not r["local_ok"] for r in parsed])) if parsed else float("nan"),
            "final_answer_acc": float(np.mean([r["global_ok"] for r in last.values()])) if last else float("nan"),
            "mean_pos": float(np.mean([r["pos"] for r in steps])) if steps else float("nan"),
            "labelled_steps_per_250_incorrect": 250 * len(labelled) / max(n_bad, 1),
        }
    return out


# --------------------------------------------------------------------------------------------
# 2. Attribution timing and graph size
# --------------------------------------------------------------------------------------------

def graph_stats(graph, thresholds=THRESHOLDS) -> dict:
    """Unpruned size, and kept nodes (by type) / edges after ``prune_graph`` at each threshold.

    ``edges_node_pruned``: all edges among the kept nodes (at the loosest node threshold this is
    ``attribution.save_compact``'s edge set);
    ``edges``: edges after the edge prune too. ``*_mb`` assume sparse COO (int64 x2 + fp32 = 20 B/edge).
    """
    from circuit_tracer.graph import prune_graph

    adj = graph.adjacency_matrix.cpu()
    nz = adj != 0
    n_feat, n_tok, n_logit = len(graph.selected_features), graph.n_pos, len(graph.logit_targets)
    n_err = graph.cfg.n_layers * graph.n_pos
    out = {"n_pos": graph.n_pos, "n_active": len(graph.active_features), "n_feat": n_feat, "n_nodes": adj.shape[0],
           "nnz": int(nz.sum()), "dense_mb": adj.numel() * adj.element_size() / 1e6, "pruned": {}}
    bounds = np.cumsum([0, n_feat, n_err, n_tok, n_logit])
    for nt, et in thresholds:
        node_mask, edge_mask = (m.cpu() for m in prune_graph(graph, nt, et)[:2])
        kept = node_mask.nonzero().squeeze(1)
        types = {name: int(node_mask[a:b].sum()) for name, a, b in
                 zip(["feature", "error", "token", "logit"], bounds[:-1], bounds[1:])}
        n_node_pruned = int(nz[kept][:, kept].sum())
        n_edges = int((edge_mask & nz).sum())
        out["pruned"][f"{nt}/{et}"] = {"nodes": int(node_mask.sum()), **types, "edges_node_pruned": n_node_pruned,
                                       "edges": n_edges, "node_pruned_mb": n_node_pruned * 20 / 1e6,
                                       "pruned_mb": n_edges * 20 / 1e6}
    return out


def _kept_feature_keys(graph, threshold=DEFAULT_THRESHOLD) -> set:
    from circuit_tracer.graph import prune_graph

    mask = prune_graph(graph, *threshold).node_mask.cpu()[: len(graph.selected_features)]
    feats = graph.active_features[graph.selected_features].cpu()[mask]
    return {tuple(f) for f in feats.tolist()}


def sample_steps(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Up to ``n`` labelled steps (``label`` not None), half incorrect where possible, shuffled."""
    rng = random.Random(seed)
    rows = [r for r in rows if r.get("label") is not None]
    bad, good = [r for r in rows if r["label"] == 1], [r for r in rows if r["label"] == 0]
    rng.shuffle(bad), rng.shuffle(good)
    k = min(len(bad), n // 2)
    picked = bad[:k] + good[: n - k]
    rng.shuffle(picked)
    return picked


def graph_sizes(model, steps_dir: Path, out_dir: Path, n: int, max_feature_nodes=(8192, 4096, 2048),
                batch_size: int = 256, save_data: bool = True, seed: int = 0) -> dict:
    """Attribute ``n`` sampled steps at each ``max_feature_nodes``; record time, memory and sizes.

    ``max_feature_nodes[0]`` is the reference: smaller caps report the Jaccard overlap of their
    kept features (default pruning) with it. With ``save_data``, the reference graph, pruned at the
    default threshold, is converted to PyG ``Data`` and saved to ``data.pt`` for experiments 4-5
    (decoder-PCA columns left at zero).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path, data_path = out_dir / "sizes.jsonl", out_dir / "data.pt"
    done = {(r["problem_id"], r["step_idx"]) for r in _read_jsonl(rows_path)}
    data = torch.load(data_path, weights_only=False) if data_path.exists() else []
    token_classes = gemma.token_class_table(model.tokenizer) if save_data else None
    for step in sample_steps(_read_jsonl(steps_dir / "steps.jsonl"), n, seed):
        key = (step["problem_id"], step["step_idx"])
        if key in done:
            continue
        rec = torch.load(steps_dir / "records" / f"{key[0]}.pt", weights_only=False)
        tokens, pos = rec["tokens"], step["pos"]
        row, ref = {"problem_id": key[0], "step_idx": key[1], "level": step["level"], "label": step["label"],
                    "by_cap": {}}, None
        for cap in max_feature_nodes:
            _reset_peak()
            t = time.perf_counter()
            graph = attribution.build_graph(model, tokens, pos, max_feature_nodes=cap, batch_size=batch_size)
            stats = {"seconds": time.perf_counter() - t, "peak_gb": _peak_gb(), **graph_stats(graph)}
            if ref is None:
                ref = _kept_feature_keys(graph)
                row["top1_is_generated"] = int(graph.logit_token_ids[0]) == int(tokens[pos])
                if save_data:
                    node_mask, edge_mask, _ = attribution.prune(graph, *DEFAULT_THRESHOLD)
                    rs = rec["steps"][key[1]]
                    d = graph_to_data(graph, step["label"], rs["error_norms"], token_classes,
                                      node_mask=node_mask, edge_mask=edge_mask)
                    d.problem_id, d.step_idx, d.line, d.prev_line = key[0], key[1], rs["line"], rs["prev_line"]
                    data.append(d)
                    torch.save(data, data_path)
            else:
                kept = _kept_feature_keys(graph)
                stats["jaccard_vs_ref"] = len(kept & ref) / max(len(kept | ref), 1)
            row["by_cap"][str(cap)] = stats
            del graph
            _free()
        _append_jsonl(rows_path, row)
        print({k: row[k] for k in ("problem_id", "step_idx", "label")},
              {c: round(s["seconds"], 1) for c, s in row["by_cap"].items()})
    return summarize_sizes(_read_jsonl(rows_path))


def summarize_sizes(rows: list[dict]) -> dict:
    """Per ``max_feature_nodes`` cap: time, memory, unpruned size, and pruned size per threshold."""
    out = {"n_graphs": len(rows), "top1_is_generated": float(np.mean([r["top1_is_generated"] for r in rows]))}
    for cap in rows[0]["by_cap"]:
        s = [r["by_cap"][cap] for r in rows]
        out[cap] = {k: _quantiles([x[k] for x in s])
                    for k in ("seconds", "peak_gb", "n_pos", "n_active", "n_feat", "n_nodes", "nnz", "dense_mb")}
        if "jaccard_vs_ref" in s[0]:
            out[cap]["jaccard_vs_ref"] = _quantiles([x["jaccard_vs_ref"] for x in s])
        out[cap]["pruned"] = {th: {k: _quantiles([x["pruned"][th][k] for x in s]) for k in s[0]["pruned"][th]}
                              for th in s[0]["pruned"]}
    return out


# --------------------------------------------------------------------------------------------
# 4-5. GNN timing and CV dry run (on the graphs saved by ``graph_sizes``)
# --------------------------------------------------------------------------------------------

def _model_kwargs(conv: str, n_pca: int) -> dict:
    return dict(type_dims=type_dims(n_pca), conv=conv, n_model_layers=26, max_pos=64)


def gnn_epoch_timing(data_list, convs=("graphconv", "deepsets"), n_target: int = 1000,
                     device: str = "cuda" if torch.cuda.is_available() else "cpu",
                     batch_size: int = 32, epochs_assumed: int = 50, n_pca: int = 8) -> dict:
    """Seconds per training epoch on ``data_list`` tiled to ``n_target`` graphs, per conv.

    ``cv_hours_est``: 5 folds x 5 seeds x ``epochs_assumed`` epochs on 80% of the data.
    """
    from torch_geometric.loader import DataLoader

    from .models import GraphClassifier

    tiled = (data_list * math.ceil(n_target / len(data_list)))[:n_target]
    loader = DataLoader(tiled, batch_size, shuffle=True)
    out = {"n_graphs": n_target, "device": device,
           "nodes": _quantiles([d.num_nodes for d in data_list]),
           "edges": _quantiles([d.edge_index.shape[1] for d in data_list])}
    for conv in convs:
        model = GraphClassifier(**_model_kwargs(conv, n_pca)).to(device)
        opt = torch.optim.AdamW(model.parameters())
        times = []
        for epoch in range(2):  # epoch 0 is warm-up (CUDA init, kernel caching)
            if device == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            for b in loader:
                b = b.to(device)
                opt.zero_grad()
                torch.nn.functional.binary_cross_entropy_with_logits(model(b), b.y).backward()
                opt.step()
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        out[conv] = {"epoch_seconds": times[1], "cv_hours_est": times[1] * 0.8 * epochs_assumed * 25 / 3600}
        print(conv, out[conv])
    return out


def cv_dry_run(data_list, conv: str = "graphconv", n_folds: int = 5, seeds=range(5),
               device: str = "cuda" if torch.cuda.is_available() else "cpu", n_pca: int = 8) -> dict:
    """One full grouped CV run: wall time, epochs to early stopping, and (meaningless at this size) metrics."""
    from .train import TrainConfig, grouped_cv, gnn_fit_predict, summarize

    fp = gnn_fit_predict(data_list, _model_kwargs(conv, n_pca), TrainConfig(device=device))
    y, groups = [d.y.item() for d in data_list], [d.problem_id for d in data_list]
    t = time.perf_counter()
    rows, _ = grouped_cv(y, groups, fp, n_folds, seeds)
    return {"conv": conv, "n_graphs": len(data_list), "n_incorrect": int(sum(y)), "device": device,
            "wall_seconds": time.perf_counter() - t, "epochs_run": _quantiles(fp.epochs),
            "metrics": summarize(rows)}


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["memory", "balance", "sizes", "gnn", "cv"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100, help="balance: problems per level; sizes: steps to attribute")
    ap.add_argument("--levels", nargs="+", default=list(LEVELS))
    ap.add_argument("--steps", type=Path, help="sizes: the balance output directory")
    ap.add_argument("--data", type=Path, help="gnn/cv: data.pt written by sizes")
    ap.add_argument("--dtype", default="float32", choices=list(DTYPES))
    ap.add_argument("--caps", type=int, nargs="+", default=[8192, 4096, 2048], help="sizes: max_feature_nodes")
    ap.add_argument("--batch-size", type=int, default=256, help="attribution batch size (lower if OOM)")
    ap.add_argument("--conv", default="graphconv")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    if a.experiment == "memory":
        result = memory_fit(batch_size=a.batch_size)
    elif a.experiment in ("balance", "sizes"):
        model = gemma.load_model("cuda" if _cuda() else None, DTYPES[a.dtype])
        result = (class_balance(model, a.out, a.n, a.levels) if a.experiment == "balance" else
                  graph_sizes(model, a.steps, a.out, a.n, a.caps, a.batch_size))
    else:
        data = torch.load(a.data, weights_only=False)
        result = gnn_epoch_timing(data) if a.experiment == "gnn" else cv_dry_run(data, a.conv)
    _write_json(a.out / "summary.json", result)


if __name__ == "__main__":
    main()
