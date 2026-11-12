"""Non-GNN baselines, each a ``train.FitPredict`` factory over a list of PyG ``Data``.

Features are extracted once into a matrix; every factory then only slices rows by index, so
``grouped_cv`` runs all baselines on identical problem-grouped folds. Positive class = incorrect.
"""
from __future__ import annotations

import ast
import operator
import re

import networkx as nx
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .graph_data import ERROR, FEATURE, LOGIT, TOKEN
from .train import FitPredict


# --------------------------------------------------------------------------------------------
# Shared wrappers
# --------------------------------------------------------------------------------------------

def _labels(data_list) -> np.ndarray:
    return np.array([d.y.item() for d in data_list]).astype(int)


def _matrix_fit_predict(X: np.ndarray, y: np.ndarray, make_model) -> FitPredict:
    """FitPredict over precomputed rows ``X``: fit ``make_model(seed)`` on train rows, return P(y=1)."""
    def fit_predict(train_idx, test_idx, seed):
        model = make_model(seed).fit(X[train_idx], y[train_idx])
        return model.predict_proba(X[test_idx])[:, 1]
    return fit_predict


def _logreg(seed: int):
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000,
                                                              random_state=seed))


def _gbm(seed: int):
    # no internal early stopping: its random validation split would not be grouped by problem
    return HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False, random_state=seed)


def _gbc(seed: int):
    """The CRV paper's classifier: sklearn ``GradientBoostingClassifier`` with default hyperparameters
    (no NaN support, so NaNs are median-imputed)."""
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         GradientBoostingClassifier(random_state=seed))


_MODELS = {"hgb": _gbm, "gbc": _gbc}


def logreg_on_columns(data_list, columns) -> FitPredict:
    """Logistic regression on ``columns(d) -> sequence of floats`` computed per graph."""
    X = np.array([np.asarray(columns(d), dtype=float) for d in data_list])
    return _matrix_fit_predict(X, _labels(data_list), _logreg)


# --------------------------------------------------------------------------------------------
# 1-2. Sanity-check baselines
# --------------------------------------------------------------------------------------------

def graph_size_baseline(data_list) -> FitPredict:
    """Logistic regression on (log) node and edge counts of the pruned graph."""
    return logreg_on_columns(data_list, lambda d: np.log1p([d.num_nodes, d.edge_index.shape[1]]))


def top_prob_baseline(data_list) -> FitPredict:
    """Logistic regression on the top-1 token probability (model confidence only)."""
    return logreg_on_columns(data_list, lambda d: [d.logit_probs[0, 0].item()])


# --------------------------------------------------------------------------------------------
# 3. Operand + answer tokens only
# --------------------------------------------------------------------------------------------

_NUM = re.compile(r"(?<![\d.])-?\d+(?:\.\d+)?")  # "12-3" -> 12, 3 (not -3)
_ARITH = r"[\d.+\-*/×÷() ]*"  # arithmetic characters (no newline)
_LEAD, _TRAIL = re.compile("^" + _ARITH), re.compile(_ARITH + "$")
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _eval(expr: str) -> float:
    """Value of a ``+ - * / ( )`` arithmetic expression (``×`` / ``÷`` allowed); NaN if not one."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -ev(n.operand) if isinstance(n.op, ast.USub) else ev(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return _BINOPS[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError(expr)
    try:
        return float(ev(ast.parse(expr.replace("×", "*").replace("÷", "/").strip(), mode="eval").body))
    except (SyntaxError, ValueError, ZeroDivisionError):
        return np.nan


def arithmetic_check(line: str, prev_line: str = "") -> float:
    """1 if the step's last equality holds, 0 if not, NaN if it cannot be parsed.

    The equality is the expression just before the last ``=`` (in ``prev_line`` + ``line``, so a
    chained step ``"= 42 + 5"`` is checked against the previous line) vs. the expression after it.
    """
    parts = (prev_line + "\n" + line).split("=")
    if len(parts) < 2:
        return np.nan
    lhs, rhs = parts[-2:]
    a, b = _eval(_TRAIL.search(lhs.strip()).group()), _eval(_LEAD.search(rhs.strip()).group())
    return np.nan if np.isnan(a) or np.isnan(b) else float(np.isclose(a, b))


def _numbers(text: str, k: int) -> list[float]:
    """First ``k`` numbers in ``text``, NaN-padded."""
    nums = [float(s) for s in _NUM.findall(text)][:k]
    return nums + [np.nan] * (k - len(nums))


def text_features(line: str, prev_line: str = "", k: int = 4) -> np.ndarray:
    """Numbers-only features INCLUDING the answer: the first ``k`` numbers of the step and of the
    previous step, and the digit length of the last number in the step. Excludes ``arithmetic_check``
    (if labels come from checking the arithmetic, it would be the label itself)."""
    last = _NUM.findall(line)
    return np.array(_numbers(line, k) + _numbers(prev_line, k) + [len(last[-1].lstrip("-")) if last else np.nan])


def _carry(a: float, b: float, op: str) -> float:
    """1 if a digit-wise ``a + b`` carries / ``a - b`` borrows (non-negative integers), else 0; NaN otherwise."""
    if op not in "+-" or np.isnan(a) or np.isnan(b) or a % 1 or b % 1 or a < 0 or b < 0:
        return np.nan
    a, b = int(a), int(b)
    c = anyc = 0  # running carry / borrow, and whether one ever occurred
    while a or b:
        x, y = a % 10, b % 10
        c = int(x + y + c >= 10) if op == "+" else int(x - y - c < 0)
        anyc |= c
        a, b = a // 10, b // 10
    return float(anyc)


def difficulty_features(line: str, prev_line: str = "", step_idx: int = 0, k: int = 4) -> np.ndarray:
    """Problem-difficulty features that never see the step's answer.

    Uses only the expression before the step's last ``=`` (``arithmetic_check``'s left side, so a
    chained step reads its operands from ``prev_line``): the first ``k`` operands and their digit
    lengths, operator counts (+ - * /), carry / borrow flag of the first binary operation, the number
    of operations remaining on the right of the ``=`` (count only, not its numbers), and ``step_idx``.
    """
    parts = (prev_line + "\n" + line).split("=")
    lhs = _TRAIL.search(parts[-2].strip()).group() if len(parts) > 1 else line
    rhs = _LEAD.search(parts[-1].strip()).group() if len(parts) > 1 else ""
    nums = _numbers(lhs, k)
    ops = lhs.replace("×", "*").replace("÷", "/")
    counts = [len(re.findall(r"(?<=[\d)\s])\s*" + re.escape(o) + r"(?=\s*[\d(])", ops)) for o in "+-*/"]
    first_op = next((o for o in "+-" if o in ops and counts["+-".index(o)] == sum(counts)), "")
    return np.array(nums + [len(str(abs(int(x)))) if not np.isnan(x) and x % 1 == 0 else np.nan for x in nums] +
                    counts + [_carry(nums[0], nums[1], first_op) if first_op else np.nan,
                              len(re.findall(r"(?<=[\d)\s])\s*[-+*/×÷]\s*(?=[\d(])", rhs)), step_idx], dtype=float)


def operand_baseline(data_list) -> FitPredict:
    """Difficulty control (``difficulty_features``, no answer, no graph) + GBM.

    The real confound control: does problem difficulty alone predict the model's errors?
    """
    X = np.array([difficulty_features(d.line, getattr(d, "prev_line", ""), getattr(d, "step_idx", 0))
                  for d in data_list])
    return _matrix_fit_predict(X, _labels(data_list), _gbm)


def operand_answer_baseline(data_list) -> FitPredict:
    """Lookup-table CEILING, not a comparison: GBM on ``text_features`` (numbers incl. the answer).

    A GBM can memorise small-number arithmetic, so this approaches the label when labels are
    arithmetic correctness. Shows how much of any result is recoverable from the text alone.
    """
    X = np.array([text_features(d.line, getattr(d, "prev_line", "")) for d in data_list])
    return _matrix_fit_predict(X, _labels(data_list), _gbm)


def oracle_baseline(data_list) -> FitPredict:
    """ORACLE, not a comparison: P(incorrect) = 1 - ``arithmetic_check`` (0.5 if unparseable); no fitting.

    A ceiling / label-sanity check: if labels come from checking the arithmetic, its AUROC is ~1
    on parseable steps, and any shortfall is label noise or parse failures.
    """
    p = np.array([1 - arithmetic_check(d.line, getattr(d, "prev_line", "")) for d in data_list])
    p = np.nan_to_num(p, nan=0.5)
    return lambda train_idx, test_idx, seed: p[test_idx]


# --------------------------------------------------------------------------------------------
# 4. CRV-style graph statistics
# --------------------------------------------------------------------------------------------

def _stats(v) -> list[float]:
    """[mean, max, std] of a 1-d tensor (zeros if empty)."""
    return [v.mean().item(), v.max().item(), v.std(unbiased=False).item()] if len(v) else [0.0] * 3


def logit_stats(d) -> list[float]:
    """Top-token probability and entropy of the (renormalised) top-k logit distribution."""
    p = d.logit_probs[0].double()
    q = p / p.sum().clamp_min(1e-12)
    return [p[0].item(), -(q * q.clamp_min(1e-12).log()).sum().item()]


def topo_features(d, n_sample: int = 50, seed: int = 0) -> list[float]:
    """Topological statistics of the pruned directed graph (CRV App. C.1), 8 values:

    density; degree centrality (mean, max; in + out degree / (n - 1)); betweenness centrality (mean,
    max; edge length 1 / |weight|); number of weakly connected components; average shortest path
    length in the largest weakly connected component (undirected, hops); shortest directed path
    (hops) from any input-token node to any logit node (NaN if none).

    For speed, betweenness uses ``n_sample`` sampled source nodes (``nx`` k-sampling, fixed ``seed``,
    an approximation) and the average path length averages BFS distances from ``n_sample`` sampled
    nodes (seeded).
    """
    n = d.num_nodes
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    src, dst = d.edge_index.tolist()
    G.add_weighted_edges_from(((u, v, 1 / max(abs(w), 1e-12)) for u, v, w in zip(src, dst, d.edge_weight.tolist())
                               if u != v), weight="length")
    deg = np.array([deg for _, deg in G.degree()]) / max(n - 1, 1)
    bc = np.array(list(nx.betweenness_centrality(G, k=min(n_sample, n), weight="length", seed=seed).values()))
    comps = list(nx.weakly_connected_components(G))
    big = G.subgraph(max(comps, key=len)).to_undirected()
    rng = np.random.default_rng(seed)
    nodes = list(big)
    srcs = rng.choice(nodes, min(n_sample, len(nodes)), replace=False)
    dists = [x for s in srcs for x in nx.single_source_shortest_path_length(big, int(s)).values() if x > 0]
    t = d.node_type.tolist()
    ends = {i for i, k in enumerate(t) if k == LOGIT}
    hops = nx.multi_source_dijkstra_path_length(G, {i for i, k in enumerate(t) if k == TOKEN}, weight="hop")
    return [nx.density(G), deg.mean(), deg.max(), bc.mean(), bc.max(), len(comps),
            float(np.mean(dists)) if dists else np.nan,
            float(min((hops[i] for i in ends if i in hops), default=np.nan))]


def crv_features(d, n_layers: int = 26, with_logit_stats: bool = True) -> np.ndarray:
    """Hand-built CRV-style statistics of one graph.

    CRV's feature list (Zhao et al., arXiv 2510.09312, App. C.1): node-type counts; activation and
    influence statistics; per-layer feature counts; edge-weight statistics; global topology (density,
    degree and betweenness centrality, connected components, path lengths; ``topo_features``); and,
    optionally, logit statistics (top-token probability, entropy). Ours, per piece:

    Per node type: count. Feature nodes: stats of activation and influence, counts per layer
    (``n_layers + 2`` bins: embedding, layers, logits). Error nodes: stats of influence and
    error norm. Token nodes: influence stats. Edges: the max |raw weight| (``w_scale``), then per
    source node type count, mean signed weight and stats of |weight|, and the in-weight of the top-1
    logit node (weights scaled to max |w| = 1). Then ``topo_features`` (NaN where undefined), and
    optionally ``logit_stats``.
    """
    t, x, w = d.node_type, d.x, d.edge_weight
    f = [(t == k).sum().item() for k in (FEATURE, ERROR, TOKEN, LOGIT)]
    feat, err, tok = (x[t == k] for k in (FEATURE, ERROR, TOKEN))
    f += _stats(feat[:, 0]) + _stats(feat[:, 1]) + _stats(err[:, 0]) + _stats(err[:, 1]) + _stats(tok[:, 0])
    layers = d.layer_idx[t == FEATURE].clamp(max=n_layers + 1)
    f += np.bincount(layers.numpy(), minlength=n_layers + 2).tolist()
    f.append(d.w_scale.item())
    src_type = t[d.edge_index[0]]
    for k in (FEATURE, ERROR, TOKEN):
        wk = w[src_type == k]
        f += [len(wk), wk.mean().item() if len(wk) else 0.0] + _stats(wk.abs())
    into_anchor = w[d.edge_index[1] == d.anchor_index[0]]
    f += [into_anchor.abs().sum().item(), into_anchor.sum().item(), len(into_anchor)]
    f += topo_features(d)
    if with_logit_stats:
        f += logit_stats(d)
    return np.array(f, dtype=float)


def crv_matrix(data_list, with_logit_stats: bool = True, n_layers: int = 26) -> np.ndarray:
    return np.stack([crv_features(d, n_layers, with_logit_stats) for d in data_list])


def crv_baseline(data_list, with_logit_stats: bool = True, n_layers: int = 26, model: str = "hgb") -> FitPredict:
    """CRV-style features (matrix built once) + a gradient-boosted classifier.

    ``with_logit_stats=False`` drops top-token probability and entropy ("without logit stats").
    ``model``: ``"hgb"`` (``HistGradientBoostingClassifier``, balanced, native NaN) or ``"gbc"``
    (the paper's ``GradientBoostingClassifier``, defaults, median-imputed; slower, unweighted).
    """
    return _matrix_fit_predict(crv_matrix(data_list, with_logit_stats, n_layers), _labels(data_list),
                               _MODELS[model])


# --------------------------------------------------------------------------------------------
# 5. Hybrid: out-of-fold GNN probability + CRV features
# --------------------------------------------------------------------------------------------

def hybrid_baseline(data_list, gnn, with_logit_stats: bool = True, n_layers: int = 26,
                    n_inner: int = 5) -> FitPredict:
    """GBM on CRV features plus an out-of-fold GNN probability column.

    ``gnn`` is either
      * a ``FitPredict`` for the GNN (e.g. ``train.gnn_fit_predict``): nested cross-fitting. Per
        outer fold, the training rows get OOF probabilities from a problem-grouped ``n_inner``-fold
        CV *within* the training rows, and the test rows from a GNN fit on all training rows.
        Leak-free, but costs ``n_inner + 1`` GNN fits per outer fold; or
      * precomputed OOF probabilities from ``grouped_cv`` (an array, or its ``{seed: array}`` dict;
        run the GNN and the hybrid with the same seeds). Cheap, but mildly optimistic: the training
        rows' probabilities came from GNNs that were trained on the hybrid fold's test rows.
    """
    X, y = crv_matrix(data_list, with_logit_stats, n_layers), _labels(data_list)
    groups = np.array([d.problem_id for d in data_list])

    def gnn_column(train_idx, test_idx, seed) -> np.ndarray:
        if not callable(gnn):
            return np.asarray(gnn[seed] if isinstance(gnn, dict) else gnn, dtype=float)
        p = np.full(len(y), np.nan)
        inner = StratifiedGroupKFold(n_inner, shuffle=True, random_state=seed)
        for tr, va in inner.split(train_idx, y[train_idx], groups[train_idx]):
            p[train_idx[va]] = gnn(train_idx[tr], train_idx[va], seed)
        p[test_idx] = gnn(train_idx, test_idx, seed)
        return p

    def fit_predict(train_idx, test_idx, seed):
        Xh = np.column_stack([X, gnn_column(train_idx, test_idx, seed)])
        return _matrix_fit_predict(Xh, y, _gbm)(train_idx, test_idx, seed)

    return fit_predict


def all_baselines(data_list, gnn=None) -> dict[str, FitPredict]:
    """Every baseline by name (the hybrid only if ``gnn`` is given; see ``hybrid_baseline``)."""
    out = {
        "graph_size": graph_size_baseline(data_list),
        "top_prob": top_prob_baseline(data_list),
        "operands": operand_baseline(data_list),  # difficulty control (no answer)
        "operands_answer": operand_answer_baseline(data_list),  # lookup-table ceiling, not a comparison
        "crv": crv_baseline(data_list),
        "crv_no_logit": crv_baseline(data_list, with_logit_stats=False),
        "oracle_check": oracle_baseline(data_list),  # ceiling / label sanity check, not a comparison
    }
    if gnn is not None:
        out["hybrid"] = hybrid_baseline(data_list, gnn)
    return out
