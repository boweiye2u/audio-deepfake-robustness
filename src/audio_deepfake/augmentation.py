from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import soundfile as sf
import torch

from audio_deepfake.baseline import AudioRecord, pad_or_crop
from audio_deepfake.corruptions import add_noise, signal_to_noise_db


@dataclass(frozen=True)
class NoiseAugmentationDecision:
    applied: bool
    snr_db: float | None


def noise_augmentation_decision(
    utterance_id: str, seed: int, probability: float, snr_choices: Sequence[float]
) -> NoiseAugmentationDecision:
    if not 0 <= probability <= 1:
        raise ValueError("augmentation probability must be in [0, 1]")
    if not snr_choices:
        raise ValueError("at least one SNR choice is required")
    digest = hashlib.sha256(f"{seed}:{utterance_id}:augment-decision".encode()).digest()
    uniform = int.from_bytes(digest[:8], "little") / 2**64
    if uniform >= probability:
        return NoiseAugmentationDecision(False, None)
    index = int.from_bytes(digest[8:16], "little") % len(snr_choices)
    return NoiseAugmentationDecision(True, float(snr_choices[index]))


def select_robust_checkpoint(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        raise ValueError("checkpoint history is empty")
    required = {"epoch", "mean_selection_eer", "clean_eer"}
    if any(not required.issubset(record) for record in history):
        raise ValueError("checkpoint history lacks selection fields")
    return min(
        history,
        key=lambda record: (
            float(record["mean_selection_eer"]),
            float(record["clean_eer"]),
            int(record["epoch"]),
        ),
    )


def validate_paired_ids(reference: list[str], candidate: list[str]) -> None:
    if reference != candidate:
        raise ValueError("baseline and robust predictions are not aligned")


class NoiseAugmentedTrainingDataset(torch.utils.data.Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[AudioRecord],
        seed: int,
        probability: float,
        snr_choices: Sequence[float],
        sample_rate: int,
        max_duration_seconds: float,
        crop_strategy: str,
    ) -> None:
        if any(record.split != "train" for record in records):
            raise PermissionError("noise augmentation accepts TRAIN records only")
        self.records = records
        self.seed = seed
        self.probability = probability
        self.snr_choices = tuple(float(value) for value in snr_choices)
        self.sample_rate = sample_rate
        self.max_samples = round(sample_rate * max_duration_seconds)
        self.crop_strategy = crop_strategy

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        waveform, rate = sf.read(record.audio_path, dtype="float32", always_2d=False)
        if rate != self.sample_rate or waveform.ndim != 1:
            raise ValueError(f"invalid source audio: {record.audio_path}")
        decision = noise_augmentation_decision(
            record.utterance_id, self.seed, self.probability, self.snr_choices
        )
        measured_snr = float("nan")
        if decision.applied:
            clean = waveform.copy()
            condition = f"train_noise_{decision.snr_db:g}db"
            waveform = add_noise(
                waveform, float(decision.snr_db), self.seed, record.utterance_id, condition
            )
            measured_snr = signal_to_noise_db(clean, waveform)
        waveform = pad_or_crop(waveform, self.max_samples, self.crop_strategy)
        valid_samples = min(round(record.duration_seconds * self.sample_rate), self.max_samples)
        attention_mask = torch.zeros(self.max_samples, dtype=torch.long)
        attention_mask[:valid_samples] = 1
        valid = waveform[:valid_samples]
        mean = float(valid.mean()) if valid_samples else 0.0
        variance = float(valid.var()) if valid_samples else 0.0
        waveform[:valid_samples] = (valid - mean) / np.sqrt(variance + 1e-7)
        return {
            "waveform": torch.from_numpy(waveform),
            "attention_mask": attention_mask,
            "label": record.label,
            "index": index,
            "augmented": decision.applied,
            "snr_db": float(decision.snr_db) if decision.applied else float("nan"),
            "measured_snr_db": measured_snr,
        }
