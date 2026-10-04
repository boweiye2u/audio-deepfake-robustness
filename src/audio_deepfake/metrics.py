from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def equal_error_rate(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("EER requires both binary classes")
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    false_negative_rate = 1.0 - true_positive_rate
    index = int(np.nanargmin(np.abs(false_positive_rate - false_negative_rate)))
    eer = float((false_positive_rate[index] + false_negative_rate[index]) / 2.0)
    return eer, float(thresholds[index])


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")

    predictions = (scores >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    eer, eer_threshold = equal_error_rate(labels, scores)
    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "decision_threshold": float(threshold),
    }


def relative_degradation(clean: float, shifted: float, higher_is_better: bool = True) -> float:
    if clean == 0:
        raise ValueError("clean reference must be nonzero")
    delta = clean - shifted if higher_is_better else shifted - clean
    return float(delta / abs(clean))
