"""Build and prune attribution graphs for graded steps, and write the dataset manifest.

Convention (INTERFACE.md): for a step whose final token sits at position ``p``, the graph is built on
``tokens[:p]`` and, under greedy decoding, its rank-1 logit target is ``tokens[p]`` (the anchor).

Pruning (``prune`` / ``reprune``) follows circuit-tracer's ``prune_graph`` (node influence -> node mask ->
edge scores -> edge mask -> remove features/errors left without in/out edges), except that edges are
scored on the **full** graph: ``score[t, s] = |A[t, s]| / sum_s' |A[t, s']| * (influence[t] + logit_weight[t])``
instead of renormalising and recomputing influence on the node-pruned graph. Node masks are identical to
circuit-tracer's; edge thresholds are fractions of the edge score among the kept nodes. Scores then do not
depend on the node threshold, which is what lets ``sparsify`` store a small file that re-prunes exactly.

Storage (``storage=``):
    "full"    -- circuit-tracer ``Graph.to_pt`` (dense N x N float32; ~50-200 MB per graph) plus ``prune``
                 masks at ``node_threshold``/``edge_threshold`` in ``masks/``.
    "compact" -- ``save_compact``: the sparse ``sparsify`` dict (~0.3-2.5 MB estimated); ``reprune``
                 gives exact masks for any node threshold in [min_node_threshold, node_threshold] and
                 any edge threshold <= edge_threshold.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import NamedTuple

import torch

from .problems import Problem, StepGrade, grade_steps

ATTRIBUTION_KW = dict(
    max_n_logits=5,          # fixed top-5 logit nodes (count must not leak confidence)
    desired_logit_prob=1.0,  # never stop early on cumulative probability
    max_feature_nodes=4096,  # as CRV; the main cost lever
    batch_size=64,
    offload=None,
    verbose=False,
)
# Re-prunable range stored by "compact": node thresholds in [min_node_threshold, node_threshold], edge <= edge_threshold.
SPARSE_RANGE = dict(node_threshold=0.95, edge_threshold=0.99, min_node_threshold=0.5)


class Masks(NamedTuple):
    node_mask: torch.Tensor  # (n,) bool
    edge_mask: torch.Tensor  # (n, n) bool, [target, source] like the adjacency matrix


def build_graph(model, tokens: torch.Tensor, pos: int, **kw):
    """Attribution graph for the step ending at ``pos`` (input ``tokens[:pos]``, which starts with BOS)."""
    from circuit_tracer import attribute

    return attribute(tokens[:pos], model, **{**ATTRIBUTION_KW, **kw})


# --------------------------------------------------------------------------------------------
# Sparse storage and exact re-pruning
# --------------------------------------------------------------------------------------------

def sparsify(graph, node_threshold: float = SPARSE_RANGE["node_threshold"],
             edge_threshold: float = SPARSE_RANGE["edge_threshold"],
             min_node_threshold: float = SPARSE_RANGE["min_node_threshold"]) -> dict:
    """Sparse graph from which ``reprune`` exactly reproduces ``prune`` within the given range.

    Stored nodes: those passing the node prune at ``node_threshold`` (before the no-edge cleanup), plus
    all tokens and logits, in circuit-tracer's global order ``[features | errors | tokens | logits]``
    (``node_index``). Stored edges (``edge_index`` = [source, target] in stored-node indices, raw
    ``edge_weight``): every edge whose score is high enough to be kept at some node threshold in
    the range with edge threshold ``edge_threshold``. Per node: full-graph ``influence``, ``row_abs_sum``
    (sum of |incoming weights| over the full graph) and ``target_weight`` (influence + logit weight).
    ``edge_mass[k]``: total score of all edges (stored or not) among the tokens, logits and the ``k``
    most influential other nodes, i.e. the denominator of the edge threshold.
    """
    from circuit_tracer.graph import compute_node_influence, find_threshold

    if not 0 < min_node_threshold <= node_threshold < 1 or not 0 < edge_threshold < 1:
        raise ValueError("need 0 < min_node_threshold <= node_threshold < 1 and 0 < edge_threshold < 1")
    A = graph.adjacency_matrix.cpu()
    n, n_logit, n_tok = A.shape[0], len(graph.logit_targets), len(graph.input_tokens)
    logit_w = torch.zeros(n, dtype=A.dtype)
    logit_w[-n_logit:] = graph.logit_probabilities.cpu().to(A.dtype)
    influence = compute_node_influence(A, logit_w)  # exactly as prune_graph's node step
    fixed = torch.arange(n) >= n - n_logit - n_tok
    cut = find_threshold(influence, node_threshold)
    keep = ((influence >= cut) | fixed).nonzero().squeeze(1)

    row_abs_sum = A.abs().double().sum(1)[keep]
    target_weight = influence[keep].double() + logit_w[keep].double()
    sub = A[keep][:, keep]
    dst, src = sub.nonzero(as_tuple=True)
    weight = sub[dst, src]
    score = _edge_score(weight, row_abs_sum, target_weight, dst)

    # Prefix rank: 0 for tokens/logits, else 1 + position in descending influence order.
    rank = torch.zeros(len(keep), dtype=torch.long)
    other = (~fixed[keep]).nonzero().squeeze(1)
    rank[other[influence[keep][other].argsort(descending=True, stable=True)]] = torch.arange(1, len(other) + 1)
    edge_mass = torch.zeros(len(other) + 1, dtype=torch.float64)
    edge_mass.scatter_add_(0, torch.maximum(rank[src], rank[dst]), score).cumsum_(0)

    # Keep the strongest edges, dropping at most (1 - edge_threshold) of the smallest edge mass in range.
    # For any node set in range and e <= edge_threshold, the dropped edges then all fall below the cutoff.
    k_min = int((influence[keep][other] >= find_threshold(influence, min_node_threshold)).sum())
    budget = (1 - edge_threshold) * edge_mass[k_min]
    asc = score.sort().values
    n_drop = int(torch.searchsorted(asc.cumsum(0), budget, right=True))
    stored = score >= asc[n_drop] if n_drop < len(asc) else torch.zeros_like(score, dtype=torch.bool)

    n_feat = len(graph.selected_features)
    feat = keep[keep < n_feat]
    idx_dtype = torch.int16 if len(keep) < 2 ** 15 else torch.int32
    return {
        "node_index": keep.to(torch.int32),
        "influence": influence[keep],
        "influence_total": influence.sort(descending=True).values.sum(),  # as find_threshold sums
        "influence_cut": cut,  # nodes with influence >= this are the full top of the influence order
        "row_abs_sum": row_abs_sum,
        "target_weight": target_weight,
        "edge_mass": edge_mass,
        "edge_index": torch.stack([src[stored], dst[stored]]).to(idx_dtype),
        "edge_weight": weight[stored],
        "range": {"node_threshold": node_threshold, "edge_threshold": edge_threshold,
                  "min_node_threshold": min_node_threshold},
        "n_nodes": n,
        "n_features": n_feat,
        "n_layers": graph.cfg.n_layers,
        "feature_ids": graph.active_features[graph.selected_features[feat]].cpu(),  # (layer, pos, feature)
        "activations": graph.activation_values[graph.selected_features[feat]].cpu(),
        "input_tokens": graph.input_tokens.cpu(),
        "logit_token_ids": graph.logit_token_ids.cpu(),
        "logit_probabilities": graph.logit_probabilities.cpu(),
        "scan_name": getattr(graph, "scan_name", None),
    }


def _edge_score(weight, row_abs_sum, target_weight, dst):
    return weight.abs().double() / row_abs_sum[dst].clamp(min=1e-10) * target_weight[dst]


def reprune(sp: dict, node_threshold: float = 0.8, edge_threshold: float = 0.98):
    """Masks over a ``sparsify`` dict's stored nodes and edges, identical to ``prune`` on the full graph.

    Returns ``(node_mask, edge_mask)``: (n_stored,) and (n_stored_edges,) bools.
    """
    r = sp["range"]
    if not r["min_node_threshold"] <= node_threshold <= r["node_threshold"] or not 0 < edge_threshold <= r["edge_threshold"]:
        raise ValueError(f"thresholds ({node_threshold}, {edge_threshold}) outside the stored range {r}")
    n = sp["n_nodes"]
    n_fixed = len(sp["input_tokens"]) + len(sp["logit_probabilities"])
    fixed = sp["node_index"].long() >= n - n_fixed
    infl = sp["influence"]

    # Node step: find_threshold over the top of the influence order (all stored) and the stored total.
    top = infl[infl >= sp["influence_cut"]].sort(descending=True).values
    i = int(torch.searchsorted(top.cumsum(0) / sp["influence_total"], node_threshold))
    node_mask = (infl >= top[min(i, len(top) - 1)]) | fixed

    # Edge step: full-graph scores; denominator = edge mass among the kept nodes.
    src, dst = sp["edge_index"].long()
    score = _edge_score(sp["edge_weight"], sp["row_abs_sum"], sp["target_weight"], dst)
    inside = node_mask[src] & node_mask[dst]
    desc = score[inside].sort(descending=True).values
    mass = sp["edge_mass"][int((node_mask & ~fixed).sum())]
    j = int(torch.searchsorted(desc.cumsum(0) / mass, edge_threshold))
    assert j < len(desc), "edge cutoff fell outside the stored edges"
    edge_mask = inside & (score >= desc[j])

    # Cleanup as prune_graph: features and errors need an outgoing edge, features an incoming one.
    gi = sp["node_index"].long()
    needs_out, needs_in = ~fixed, gi < sp["n_features"]
    while True:
        edge_mask &= node_mask[src] & node_mask[dst]
        has_out = torch.zeros_like(node_mask).index_fill_(0, src[edge_mask], True)
        has_in = torch.zeros_like(node_mask).index_fill_(0, dst[edge_mask], True)
        new = node_mask & (has_out | ~needs_out) & (has_in | ~needs_in)
        if torch.equal(new, node_mask):
            return node_mask, edge_mask
        node_mask = new


def prune(graph, node_threshold: float = 0.8, edge_threshold: float = 0.98) -> Masks:
    """Dense node/edge masks over the full graph (full-graph edge scoring; see the module docstring)."""
    sp = sparsify(graph, node_threshold, edge_threshold, node_threshold)
    node_mask, edge_mask = reprune(sp, node_threshold, edge_threshold)
    gi = sp["node_index"].long()
    src, dst = sp["edge_index"].long()
    n = sp["n_nodes"]
    full_nodes = torch.zeros(n, dtype=torch.bool)
    full_nodes[gi] = node_mask
    full_edges = torch.zeros(n, n, dtype=torch.bool)
    full_edges[gi[dst[edge_mask]], gi[src[edge_mask]]] = True
    return Masks(full_nodes, full_edges)


def save_compact(graph, path: str | Path, **sparse_kw) -> None:
    """``torch.save`` the ``sparsify`` dict (default range ``SPARSE_RANGE``)."""
    torch.save(sparsify(graph, **{**SPARSE_RANGE, **sparse_kw}), path)


def nbytes(obj) -> int:
    """Serialized size of ``obj`` under ``torch.save``."""
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.tell()


# --------------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------------

def label_record(record: dict, problem: Problem, tokenizer) -> list[StepGrade]:
    """Grade a ``gemma.run_problems`` record; the last step is unusable if generation never hit a blank line."""
    completion = tokenizer.decode(record["tokens"][record["n_prompt"]:])
    lines = [s["line"] for s in record["steps"]]
    return grade_steps(problem.expression, lines, complete="\n\n" in completion)


def _done(manifest: Path) -> set[tuple[str, int]]:
    if not manifest.exists():
        return set()
    with open(manifest) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {(r["problem_id"], r["step_idx"]) for r in rows}


def build_dataset(model, records, problems: dict[str, Problem], out_dir: str | Path, storage: str = "full",
                  node_threshold: float = 0.8, edge_threshold: float = 0.98, **attr_kw) -> Path:
    """Build, prune and save a graph for every labeled step; append one manifest row per step.

    ``records`` are ``gemma.run_problems`` records (dicts or paths to them). Resumable: steps already in
    ``out_dir/manifest.jsonl`` are skipped. Row: ``graph_path, mask_path, problem_id, step_idx, label,
    storage, anchor_ok`` with paths relative to ``out_dir`` (``mask_path`` is None for "compact", which
    is re-pruned at load time); ``anchor_ok`` says the rank-1 logit is ``tokens[pos]``, as greedy
    decoding implies.
    """
    if storage not in ("full", "compact"):
        raise ValueError(f"storage must be 'full' or 'compact', got {storage!r}")
    out_dir = Path(out_dir)
    (out_dir / "graphs").mkdir(parents=True, exist_ok=True)
    if storage == "full":
        (out_dir / "masks").mkdir(exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    done = _done(manifest)
    for rec in records:
        rec = torch.load(rec, weights_only=False) if isinstance(rec, (str, Path)) else rec
        pid, tokens = rec["problem_id"], rec["tokens"]
        for grade, step in zip(label_record(rec, problems[pid], model.tokenizer), rec["steps"]):
            if grade.label is None or (pid, grade.step_idx) in done:
                continue
            name = f"{pid}_s{grade.step_idx}"
            graph = build_graph(model, tokens, step["pos"], **attr_kw)
            graph_path = f"graphs/{name}.pt"
            if storage == "full":
                mask_path = f"masks/{name}.pt"
                graph.to_pt(out_dir / graph_path)
                torch.save(prune(graph, node_threshold, edge_threshold)._asdict(), out_dir / mask_path)
            else:  # masks live inside the compact file
                mask_path = None
                r = SPARSE_RANGE  # widen the range to cover the requested thresholds
                save_compact(graph, out_dir / graph_path, node_threshold=max(r["node_threshold"], node_threshold),
                             edge_threshold=max(r["edge_threshold"], edge_threshold),
                             min_node_threshold=min(r["min_node_threshold"], node_threshold))
            row = {"graph_path": graph_path, "mask_path": mask_path, "problem_id": pid,
                   "step_idx": grade.step_idx, "label": grade.label, "storage": storage,
                   "anchor_ok": int(graph.logit_token_ids[0]) == int(tokens[step["pos"]])}
            with open(manifest, "a") as f:
                f.write(json.dumps(row) + "\n")
            del graph
    return manifest
