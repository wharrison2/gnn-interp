"""Predicting arithmetic-step correctness from attribution graphs with GNNs.

Modules:
    problems    -- arithmetic problem generation, few-shot prompts, step grading (labels).
    gemma       -- load Gemma 2 2B + GemmaScope transcoders, generate, locate steps, error norms.
    attribution -- build + prune attribution graphs per labeled step, write the dataset manifest.
    graph_data  -- convert circuit-tracer Graphs to PyG Data (features, PCA, normalization, ablations).
    models      -- GNN / DeepSets classifiers.
    baselines   -- non-GNN baselines (size, probability, operand+answer, CRV-style GBM, hybrid).
    train       -- training loop and grouped cross-validation harness.
    metrics     -- AUROC, AUPR, FPR@95TPR, accuracy, precision, recall.
"""
import os

# transformers probes for TensorFlow on import; a broken system TF install (e.g. anaconda)
# segfaults. We never use TF, so disable the probe before anything imports transformers.
os.environ.setdefault("USE_TF", "0")
