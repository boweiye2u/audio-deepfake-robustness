import csv

import numpy as np
import pytest
import soundfile as sf
import torch

from audio_deepfake.asvspoof import MANIFEST_FIELDS
from audio_deepfake.baseline import (
    BinaryClassificationHead,
    label_to_index,
    load_development_manifest,
    load_head_checkpoint,
    load_waveform,
    pad_or_crop,
    save_head_checkpoint,
)
from audio_deepfake.metrics import (
    binary_metrics,
    equal_error_rate,
    select_balanced_accuracy_threshold,
)


def test_padding_and_center_crop():
    assert np.array_equal(pad_or_crop(np.arange(6), 4), np.array([1, 2, 3, 4]))
    assert np.array_equal(pad_or_crop(np.array([1, 2]), 4), np.array([1, 2, 0, 0]))


def test_audio_loading(tmp_path):
    path = tmp_path / "sample.flac"
    sf.write(path, np.ones(800, dtype=np.float32), 16000)
    waveform = load_waveform(path, sample_rate=16000, max_samples=1600)
    assert waveform.shape == (1600,)
    assert waveform.dtype == np.float32
    with pytest.raises(ValueError, match="expected 8000 Hz"):
        load_waveform(path, sample_rate=8000, max_samples=800)


def test_label_mapping():
    assert label_to_index("bonafide") == 0
    assert label_to_index("spoof") == 1
    with pytest.raises(ValueError, match="unknown label"):
        label_to_index("other")


def test_interpolated_eer_and_pr_auc():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    eer, _ = equal_error_rate(labels, scores)
    assert eer == pytest.approx(0.5)
    threshold = select_balanced_accuracy_threshold(labels, scores)
    metrics = binary_metrics(labels, scores, threshold)
    assert metrics["pr_auc"] == pytest.approx(5 / 6)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_head_checkpoint_round_trip(tmp_path):
    torch.manual_seed(1)
    model = BinaryClassificationHead(4, 0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    features = torch.randn(3, 4)
    model(features).sum().backward()
    optimizer.step()
    expected = model(features).detach()
    path = tmp_path / "head.pt"
    save_head_checkpoint(path, model, optimizer, 2, {"test": True})
    loaded, payload = load_head_checkpoint(path, 4, 0.0, torch.device("cpu"))
    assert torch.equal(expected, loaded(features).detach())
    assert payload["epoch"] == 2


def test_training_loader_refuses_eval_manifest(tmp_path):
    audio = tmp_path / "audio.flac"
    sf.write(audio, np.zeros(100, dtype=np.float32), 16000)
    manifest = tmp_path / "eval.csv"
    row = {
        "utterance_id": "E1",
        "audio_path": audio.name,
        "split": "eval",
        "label": "spoof",
        "speaker_id": "S1",
        "attack_id": "A07",
        "duration_seconds": "0.00625",
        "sample_rate": "16000",
    }
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(PermissionError, match="final_evaluation"):
        load_development_manifest(manifest, tmp_path)
