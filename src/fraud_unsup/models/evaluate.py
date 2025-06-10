"""Common evaluation helpers for unsupervised fraud detectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


# ────────────────────────────────────────────────────────────────────────────
# basic metrics
# ────────────────────────────────────────────────────────────────────────────
def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve."""
    return float(average_precision_score(y_true, scores))


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k_frac: float = 0.01) -> float:
    """Precision among the top-k_frac fraction of highest scores."""
    k = max(1, int(len(scores) * k_frac))
    idx = np.argsort(scores)[-k:]
    return float(y_true[idx].mean())


# ────────────────────────────────────────────────────────────────────────────
# grid search for ensemble weights
# ────────────────────────────────────────────────────────────────────────────
def best_weights_for_precision(
    y_true: np.ndarray,
    score_matrix: np.ndarray,
    step: float = 0.05,
    k_frac: float = 0.01,
) -> tuple[list[float], float]:
    """
    Brute-force search over simplex of three weights that sum to 1.
    Returns (best_weight_list, best_precision)
    """
    best_p, best_w = -1.0, [0.33, 0.33, 0.34]
    w_vals = np.arange(0.0, 1.0 + 1e-9, step)

    for w0 in w_vals:
        for w1 in w_vals:
            if w0 + w1 > 1.0 + 1e-9:
                continue
            w2 = 1.0 - w0 - w1
            if w2 < -1e-9:
                continue
            ens = (score_matrix * np.array([w0, w1, w2])).sum(axis=1)
            prec = precision_at_k(y_true, ens, k_frac)
            if prec > best_p:
                best_p, best_w = prec, [w0, w1, w2]
    return best_w, best_p


def save_weights_json(weights: Sequence[float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        json.dump({"iso": weights[0], "ae": weights[1], "gmm": weights[2]}, fp)
