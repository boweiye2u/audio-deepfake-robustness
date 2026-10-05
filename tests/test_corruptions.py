import numpy as np
import pytest

from audio_deepfake.corruptions import (
    CORE_CORRUPTIONS,
    CorruptionSpec,
    add_noise,
    apply_corruption,
    paired_bootstrap_conditions,
    round_trip_resample,
    signal_to_noise_db,
    validate_alignment,
)
from audio_deepfake.metrics import binary_metrics


def _signal() -> np.ndarray:
    time = np.arange(16000, dtype=np.float32) / 16000
    return np.sin(2 * np.pi * 440 * time).astype(np.float32)


def test_noise_is_deterministic_and_meets_snr():
    signal = _signal()
    first = add_noise(signal, 10, 7, "U1", "noise_10db")
    second = add_noise(signal, 10, 7, "U1", "noise_10db")
    assert np.array_equal(first, second)
    assert signal_to_noise_db(signal, first) == pytest.approx(10, abs=1e-5)


def test_noise_changes_with_identity():
    signal = _signal()
    assert not np.array_equal(
        add_noise(signal, 20, 7, "U1", "noise_20db"),
        add_noise(signal, 20, 7, "U2", "noise_20db"),
    )


def test_resampling_is_aligned_and_nontrivial():
    signal = _signal()
    restored = round_trip_resample(signal)
    assert restored.shape == signal.shape
    assert np.all(np.isfinite(restored))
    assert not np.array_equal(restored, signal)


def test_corruption_metadata_is_complete():
    names = [spec.name for spec in CORE_CORRUPTIONS]
    assert len(names) == len(set(names)) == 8
    for spec in CORE_CORRUPTIONS:
        assert isinstance(spec, CorruptionSpec)
        assert spec.kind in {"clean", "mp3", "resample", "noise"}


def test_paired_alignment_rejects_reordering():
    validate_alignment(["A", "B"], ["A", "B"])
    with pytest.raises(ValueError, match="not aligned"):
        validate_alignment(["A", "B"], ["B", "A"])


def test_fixed_threshold_metrics_are_not_reoptimized():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.7, 0.6, 0.9])
    result = binary_metrics(labels, scores, threshold=0.8)
    assert result["decision_threshold"] == 0.8
    assert result["confusion_matrix"] == [[2, 0], [1, 1]]


def test_paired_bootstrap_is_reproducible():
    labels = np.array([0, 0, 0, 1, 1, 1])
    clean = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    shifted = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
    kwargs = {
        "labels": labels,
        "scores_by_condition": {"clean": clean, "shifted": shifted},
        "clean_condition": "clean",
        "threshold": 0.5,
        "replicates": 50,
        "seed": 9,
    }
    assert paired_bootstrap_conditions(**kwargs) == paired_bootstrap_conditions(**kwargs)


def test_unknown_corruption_is_rejected():
    with pytest.raises(ValueError, match="unknown corruption"):
        apply_corruption(_signal(), CorruptionSpec("bad", "bad"), 1, "U1")


def test_corrupted_dataset_requires_explicit_final_evaluation_purpose(tmp_path):
    from audio_deepfake.baseline import AudioRecord
    from audio_deepfake.corruptions import CORE_CORRUPTIONS, CorruptedAudioDataset

    record = AudioRecord("E1", tmp_path / "unused.flac", "eval", 1, "S1", "A07", 1.0, 16000)
    with pytest.raises(PermissionError, match="final_evaluation"):
        CorruptedAudioDataset([record], CORE_CORRUPTIONS[0], 1, 16000, 4.0, "center")
    dataset = CorruptedAudioDataset(
        [record], CORE_CORRUPTIONS[0], 1, 16000, 4.0, "center", purpose="final_evaluation"
    )
    assert len(dataset) == 1
