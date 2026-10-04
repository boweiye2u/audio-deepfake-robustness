import numpy as np
import pytest

from audio_deepfake.metrics import binary_metrics, equal_error_rate, relative_degradation


def test_perfect_binary_metrics():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    result = binary_metrics(labels, scores)
    assert result["eer"] == pytest.approx(0.0)
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["balanced_accuracy"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_eer_requires_both_classes():
    with pytest.raises(ValueError, match="both binary classes"):
        equal_error_rate(np.array([1, 1]), np.array([0.7, 0.8]))


def test_relative_degradation_direction():
    assert relative_degradation(0.9, 0.72) == pytest.approx(0.2)
    assert relative_degradation(0.1, 0.15, higher_is_better=False) == pytest.approx(0.5)
