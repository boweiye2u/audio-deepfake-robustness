import csv

import numpy as np
import pytest
import soundfile as sf

from audio_deepfake.asvspoof import MANIFEST_FIELDS
from audio_deepfake.augmentation import (
    NoiseAugmentedTrainingDataset,
    noise_augmentation_decision,
    select_robust_checkpoint,
    validate_paired_ids,
)
from audio_deepfake.baseline import AudioRecord, load_development_manifest
from audio_deepfake.config import load_config


def record(path, split="train"):
    return AudioRecord("T1", path, split, 1, "S1", "A01", 0.1, 16000)


def test_noise_decision_is_deterministic_and_probability_extremes():
    choices = [20.0, 10.0, 5.0]
    first = noise_augmentation_decision("T1", 4, 0.5, choices)
    assert first == noise_augmentation_decision("T1", 4, 0.5, choices)
    assert not noise_augmentation_decision("T1", 4, 0.0, choices).applied
    always = noise_augmentation_decision("T1", 4, 1.0, choices)
    assert always.applied and always.snr_db in choices
    with pytest.raises(ValueError, match="probability"):
        noise_augmentation_decision("T1", 4, 1.1, choices)


def test_training_dataset_target_snr_and_disabled_identity(tmp_path):
    audio = tmp_path / "x.wav"
    clean = np.sin(np.linspace(0, 30, 1600)).astype(np.float32) * 0.1
    sf.write(audio, clean, 16000, subtype="FLOAT")
    augmented = NoiseAugmentedTrainingDataset([record(audio)], 7, 1.0, [10], 16000, 0.1, "center")[0]
    assert augmented["augmented"]
    assert augmented["measured_snr_db"] == pytest.approx(10, abs=1e-5)
    disabled = NoiseAugmentedTrainingDataset([record(audio)], 7, 0.0, [10], 16000, 0.1, "center")[0]
    assert not disabled["augmented"]
    # Both paths apply the common normalization; disabled must match normalized clean audio.
    expected = (clean - clean.mean()) / np.sqrt(clean.var() + 1e-7)
    assert np.allclose(disabled["waveform"].numpy(), expected, atol=2e-6)


def test_augmentation_refuses_dev(tmp_path):
    audio = tmp_path / "x.flac"
    sf.write(audio, np.ones(100, dtype=np.float32), 16000)
    with pytest.raises(PermissionError, match="TRAIN"):
        NoiseAugmentedTrainingDataset([record(audio, "dev")], 1, 0.5, [10], 16000, 1, "center")


def test_robust_selection_rule():
    history = [
        {"epoch": 1, "mean_selection_eer": 0.2, "clean_eer": 0.04},
        {"epoch": 2, "mean_selection_eer": 0.1, "clean_eer": 0.06},
        {"epoch": 3, "mean_selection_eer": 0.1, "clean_eer": 0.05},
        {"epoch": 4, "mean_selection_eer": 0.1, "clean_eer": 0.05},
    ]
    assert select_robust_checkpoint(history)["epoch"] == 3


def test_paired_alignment():
    validate_paired_ids(["a", "b"], ["a", "b"])
    with pytest.raises(ValueError, match="aligned"):
        validate_paired_ids(["a", "b"], ["b", "a"])


def test_phase5_config_is_train_dev_only_and_parses(monkeypatch, tmp_path):
    for name in ["AUDIO_DATA_ROOT", "AUDIO_MODEL_ROOT", "AUDIO_CACHE_ROOT", "AUDIO_OUTPUT_ROOT"]:
        monkeypatch.setenv(name, str(tmp_path))
    config = load_config("configs/noise_augmentation_5.yaml")
    assert config["augmentation"]["apply_to"] == "train_only"
    assert config["augmentation"]["probability"] == 0.5
    assert config["augmentation"]["snr_db_choices"] == [20.0, 10.0, 5.0]
    assert "eval" not in config["paths"]


def test_existing_manifest_guard_refuses_eval(tmp_path):
    audio = tmp_path / "audio.flac"
    sf.write(audio, np.ones(100, dtype=np.float32), 16000)
    manifest = tmp_path / "eval.csv"
    row = {
        "utterance_id": "E1", "audio_path": audio.name, "split": "eval",
        "label": "spoof", "speaker_id": "S1", "attack_id": "A07",
        "duration_seconds": "0.00625", "sample_rate": "16000",
    }
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(PermissionError):
        load_development_manifest(manifest, tmp_path)

@pytest.mark.parametrize(
    ("path", "probability"),
    [
        ("configs/noise_augmentation_5b_p20.yaml", 0.2),
        ("configs/noise_augmentation_5b_p35.yaml", 0.35),
    ],
)
def test_phase5b_probability_configs_change_only_probability_paths(
    path, probability, monkeypatch, tmp_path
):
    for name in ["AUDIO_DATA_ROOT", "AUDIO_MODEL_ROOT", "AUDIO_CACHE_ROOT", "AUDIO_OUTPUT_ROOT"]:
        monkeypatch.setenv(name, str(tmp_path))
    reference = load_config("configs/noise_augmentation_5.yaml")
    candidate = load_config(path)
    assert candidate["augmentation"]["probability"] == probability
    assert candidate["augmentation"]["snr_db_choices"] == reference["augmentation"]["snr_db_choices"]
    assert candidate["model"] == reference["model"]
    assert candidate["training"] == reference["training"]
    assert candidate["selection_conditions"] == reference["selection_conditions"]
    assert candidate["evaluation_conditions"] == reference["evaluation_conditions"]


def test_phase5b_tradeoff_helpers():
    from scripts.analyze_augmentation_probability_5b import gain_retained, penalty_recovered

    assert gain_retained(0.30, 0.10, 0.15) == pytest.approx(0.75)
    assert penalty_recovered(0.05, 0.09, 0.06) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        gain_retained(0.10, 0.20, 0.15)
    with pytest.raises(ValueError):
        penalty_recovered(0.10, 0.05, 0.08)
