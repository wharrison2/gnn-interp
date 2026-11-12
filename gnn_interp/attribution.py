"""Build and prune attribution graphs for graded steps, and write the dataset manifest.

Convention (INTERFACE.md): for a step whose final token sits at position ``p``, the graph is built on
``tokens[:p]`` and, under greedy decoding, its rank-1 logit target is ``tokens[p]`` (the anchor).

Storage (``storage=``):
    "full"    -- circuit-tracer ``Graph.to_pt`` (dense N x N float32; ~50-200 MB per graph).
    "compact" -- nodes kept by a loose prune plus the sparse edges among them, node metadata and
                 prune masks for several thresholds (``save_compact``); roughly MBs per graph.
"full" also saves the ``PruneResult`` at ``node_threshold``/``edge_threshold`` to ``masks/``; "compact"
stores its masks (including ``node_threshold``) inside the graph file.
"""
from __future__ import annotations

import json
from pathlib import Path

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
COMPACT_THRESHOLDS = (0.6, 0.7, 0.8, 0.9, 0.95)


def build_graph(model, tokens: torch.Tensor, pos: int, **kw):
    """Attribution graph for the step ending at ``pos`` (input ``tokens[:pos]``, which starts with BOS)."""
    from circuit_tracer import attribute

    return attribute(tokens[:pos], model, **{**ATTRIBUTION_KW, **kw})


def prune(graph, node_threshold: float = 0.8, edge_threshold: float = 0.98):
    """circuit-tracer's ``prune_graph``: node/edge masks keeping the given fraction of logit influence."""
    from circuit_tracer.graph import prune_graph

    return prune_graph(graph, node_threshold=node_threshold, edge_threshold=edge_threshold)


def save_compact(graph, path: str | Path, thresholds=COMPACT_THRESHOLDS, edge_threshold: float = 0.98) -> None:
    """Save the subgraph kept at ``max(thresholds)`` with per-threshold masks over it.

    Global node order is circuit-tracer's ``[features | errors (layer * n_pos + pos) | tokens | logits]``;
    ``node_index`` maps stored nodes into it. Edges are ``edge_index[0] -> edge_index[1]`` (source ->
    target) in stored-node indices, from the dense adjacency (``A[target, source]``).
    """
    results = {t: prune(graph, t, edge_threshold) for t in thresholds}
    keep = results[max(thresholds)].node_mask.nonzero().squeeze(1)
    A = graph.adjacency_matrix[keep][:, keep]
    tgt, src = A.nonzero(as_tuple=True)
    n_feat = len(graph.selected_features)
    feat_keep = keep[keep < n_feat]
    torch.save({
        "node_index": keep,
        "edge_index": torch.stack([src, tgt]),
        "edge_weight": A[tgt, src],
        "influence": results[max(thresholds)].cumulative_scores[keep],  # cumulative-rank score, lower = more
        "masks": {t: {"node": r.node_mask[keep], "edge": r.edge_mask[keep][:, keep][tgt, src]}
                  for t, r in results.items()},
        "feature_ids": graph.active_features[graph.selected_features[feat_keep]],  # (layer, pos, feature)
        "activations": graph.activation_values[graph.selected_features[feat_keep]],
        "n_features": n_feat,
        "n_layers": graph.cfg.n_layers,
        "input_tokens": graph.input_tokens.cpu(),
        "logit_token_ids": graph.logit_token_ids.cpu(),
        "logit_probabilities": graph.logit_probabilities.cpu(),
        "scan_name": graph.scan_name,
    }, path)


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
    anchor_ok`` with paths relative to ``out_dir`` (``mask_path`` is None for "compact"); ``anchor_ok``
    says the rank-1 logit is ``tokens[pos]``, as greedy decoding implies.
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
                thresholds = tuple(sorted({*COMPACT_THRESHOLDS, node_threshold}))
                save_compact(graph, out_dir / graph_path, thresholds, edge_threshold)
            row = {"graph_path": graph_path, "mask_path": mask_path, "problem_id": pid,
                   "step_idx": grade.step_idx, "label": grade.label,
                   "anchor_ok": int(graph.logit_token_ids[0]) == int(tokens[step["pos"]])}
            with open(manifest, "a") as f:
                f.write(json.dumps(row) + "\n")
            del graph
    return manifest
