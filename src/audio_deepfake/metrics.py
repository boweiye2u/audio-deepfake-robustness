from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def _validated(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be aligned one-dimensional arrays")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("metrics require both binary classes")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    return labels, scores


def equal_error_rate(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Linearly interpolate the ROC point where false-positive and false-negative rates meet."""
    labels, scores = _validated(labels, scores)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    false_negative_rate = 1.0 - true_positive_rate
    difference = false_positive_rate - false_negative_rate
    upper = int(np.flatnonzero(difference >= 0)[0])
    if difference[upper] == 0 or upper == 0:
        return float(false_positive_rate[upper]), float(thresholds[upper])
    lower = upper - 1
    fraction = -difference[lower] / (difference[upper] - difference[lower])
    eer = false_positive_rate[lower] + fraction * (
        false_positive_rate[upper] - false_positive_rate[lower]
    )
    threshold = thresholds[lower] + fraction * (thresholds[upper] - thresholds[lower])
    return float(eer), float(threshold)


def select_balanced_accuracy_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    labels, scores = _validated(labels, scores)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    values = (true_positive_rate + 1.0 - false_positive_rate) / 2.0
    valid = np.isfinite(thresholds)
    best = np.max(values[valid])
    candidates = np.flatnonzero(valid & np.isclose(values, best, atol=1e-12))
    index = min(candidates, key=lambda item: (abs(thresholds[item] - 0.5), thresholds[item]))
    return float(thresholds[index])


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | list[list[int]]]:
    labels, scores = _validated(labels, scores)
    predictions = (scores >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    eer, eer_threshold = equal_error_rate(labels, scores)
    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "decision_threshold": float(threshold),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def relative_degradation(clean: float, shifted: float, higher_is_better: bool = True) -> float:
    if clean == 0:
        raise ValueError("clean reference must be nonzero")
    delta = clean - shifted if higher_is_better else shifted - clean
    return float(delta / abs(clean))
