from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from audio_deepfake.baseline import AudioRecord, pad_or_crop
from audio_deepfake.metrics import binary_metrics


@dataclass(frozen=True)
class CorruptionSpec:
    name: str
    kind: str
    parameter: float | int | None = None
    unit: str | None = None


CORE_CORRUPTIONS = (
    CorruptionSpec("clean", "clean"),
    CorruptionSpec("mp3_128k", "mp3", 128, "kbps"),
    CorruptionSpec("mp3_64k", "mp3", 64, "kbps"),
    CorruptionSpec("mp3_32k", "mp3", 32, "kbps"),
    CorruptionSpec("resample_8k", "resample", 8000, "Hz"),
    CorruptionSpec("noise_20db", "noise", 20, "dB"),
    CorruptionSpec("noise_10db", "noise", 10, "dB"),
    CorruptionSpec("noise_5db", "noise", 5, "dB"),
)


def deterministic_seed(base_seed: int, utterance_id: str, condition: str) -> int:
    value = f"{base_seed}:{utterance_id}:{condition}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def signal_to_noise_db(clean: np.ndarray, corrupted: np.ndarray) -> float:
    clean = np.asarray(clean, dtype=np.float64)
    corrupted = np.asarray(corrupted, dtype=np.float64)
    if clean.shape != corrupted.shape:
        raise ValueError("SNR inputs must be aligned")
    signal_power = float(np.mean(clean**2))
    noise_power = float(np.mean((corrupted - clean) ** 2))
    if signal_power <= 0 or noise_power <= 0:
        raise ValueError("SNR requires nonzero signal and noise")
    return float(10 * np.log10(signal_power / noise_power))


def add_noise(
    waveform: np.ndarray, snr_db: float, base_seed: int, utterance_id: str, condition: str
) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    signal_power = float(np.mean(waveform.astype(np.float64) ** 2))
    if signal_power <= 0:
        raise ValueError("cannot add target-SNR noise to a silent signal")
    generator = np.random.default_rng(deterministic_seed(base_seed, utterance_id, condition))
    noise = generator.standard_normal(len(waveform)).astype(np.float64)
    noise -= noise.mean()
    noise_power = float(np.mean(noise**2))
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise *= np.sqrt(target_noise_power / noise_power)
    return (waveform.astype(np.float64) + noise).astype(np.float32)


def round_trip_resample(waveform: np.ndarray, source_rate: int = 16000, middle_rate: int = 8000) -> np.ndarray:
    if source_rate % middle_rate != 0:
        raise ValueError("resampling implementation requires an integer rate ratio")
    down = resample_poly(np.asarray(waveform, dtype=np.float32), middle_rate, source_rate)
    restored = resample_poly(down, source_rate, middle_rate).astype(np.float32)
    return align_length(restored, len(waveform))


def align_length(waveform: np.ndarray, target_length: int) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if len(waveform) >= target_length:
        return waveform[:target_length].copy()
    return np.pad(waveform, (0, target_length - len(waveform))).astype(np.float32, copy=False)


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    candidate = Path(sys.prefix) / "bin" / "ffmpeg"
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError("ffmpeg is unavailable")


def mp3_round_trip(waveform: np.ndarray, bitrate_kbps: int, sample_rate: int = 16000) -> np.ndarray:
    raw = np.asarray(waveform, dtype="<f4").tobytes()
    encode = subprocess.run(
        [
            _ffmpeg(), "-hide_banner", "-loglevel", "error", "-threads", "1",
            "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
            "-codec:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k", "-f", "mp3", "pipe:1",
        ],
        input=raw,
        capture_output=True,
        check=True,
    )
    decode = subprocess.run(
        [
            _ffmpeg(), "-hide_banner", "-loglevel", "error", "-threads", "1",
            "-f", "mp3", "-i", "pipe:0", "-f", "f32le", "-ar", str(sample_rate),
            "-ac", "1", "pipe:1",
        ],
        input=encode.stdout,
        capture_output=True,
        check=True,
    )
    output = np.frombuffer(decode.stdout, dtype="<f4").copy()
    if len(output) < max(1, int(0.9 * len(waveform))):
        raise ValueError("MP3 round trip returned implausibly short audio")
    return align_length(output, len(waveform))


def apply_corruption(
    waveform: np.ndarray,
    spec: CorruptionSpec,
    seed: int,
    utterance_id: str,
    sample_rate: int = 16000,
) -> np.ndarray:
    if spec.kind == "clean":
        output = np.asarray(waveform, dtype=np.float32).copy()
    elif spec.kind == "noise":
        output = add_noise(waveform, float(spec.parameter), seed, utterance_id, spec.name)
    elif spec.kind == "resample":
        output = round_trip_resample(waveform, sample_rate, int(spec.parameter))
    elif spec.kind == "mp3":
        output = mp3_round_trip(waveform, int(spec.parameter), sample_rate)
    else:
        raise ValueError(f"unknown corruption kind: {spec.kind}")
    if output.shape != np.asarray(waveform).shape:
        raise ValueError(f"corruption changed aligned length: {spec.name}")
    if not np.all(np.isfinite(output)):
        raise ValueError(f"corruption produced nonfinite samples: {spec.name}")
    if np.max(np.abs(output)) == 0:
        raise ValueError(f"corruption produced silence: {spec.name}")
    return output


class CorruptedAudioDataset(torch.utils.data.Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[AudioRecord],
        spec: CorruptionSpec,
        seed: int,
        sample_rate: int,
        max_duration_seconds: float,
        crop_strategy: str,
        purpose: str = "development",
    ) -> None:
        splits = {record.split for record in records}
        permitted = splits == {"dev"} and purpose == "development"
        permitted = permitted or (splits == {"eval"} and purpose == "final_evaluation")
        if not permitted:
            raise PermissionError(
                "corruption evaluation accepts DEV for development or EVAL with "
                "purpose='final_evaluation'"
            )
        self.records = records
        self.spec = spec
        self.seed = seed
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
        waveform = apply_corruption(
            waveform, self.spec, self.seed, record.utterance_id, self.sample_rate
        )
        waveform = pad_or_crop(waveform, self.max_samples, self.crop_strategy)
        valid_samples = min(round(record.duration_seconds * self.sample_rate), self.max_samples)
        attention_mask = torch.zeros(self.max_samples, dtype=torch.long)
        attention_mask[:valid_samples] = 1
        valid = waveform[:valid_samples]
        mean = float(valid.mean())
        variance = float(valid.var())
        waveform[:valid_samples] = (valid - mean) / np.sqrt(variance + 1e-7)
        return {
            "waveform": torch.from_numpy(waveform),
            "attention_mask": attention_mask,
            "index": index,
        }


def validate_alignment(reference_ids: list[str], condition_ids: list[str]) -> None:
    if reference_ids != condition_ids:
        raise ValueError("corruption predictions are not aligned with clean sample identities")


def paired_bootstrap_conditions(
    labels: np.ndarray,
    scores_by_condition: dict[str, np.ndarray],
    clean_condition: str,
    threshold: float,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int64)
    clean_scores = np.asarray(scores_by_condition[clean_condition], dtype=np.float64)
    if any(len(scores) != len(labels) for scores in scores_by_condition.values()):
        raise ValueError("bootstrap inputs are not aligned")
    metrics = ("eer", "roc_auc", "recall", "f1")
    samples = {
        condition: {metric: [] for metric in metrics} for condition in scores_by_condition
    }
    deltas = {
        condition: {metric: [] for metric in metrics}
        for condition in scores_by_condition
        if condition != clean_condition
    }
    generator = np.random.default_rng(seed)
    negative_indices = np.flatnonzero(labels == 0)
    positive_indices = np.flatnonzero(labels == 1)
    if not len(negative_indices) or not len(positive_indices):
        raise ValueError("bootstrap requires both classes")
    for _ in range(replicates):
        indices = np.concatenate(
            [
                generator.choice(negative_indices, len(negative_indices), replace=True),
                generator.choice(positive_indices, len(positive_indices), replace=True),
            ]
        )
        selected_labels = labels[indices]
        clean = binary_metrics(selected_labels, clean_scores[indices], threshold)
        for condition, scores in scores_by_condition.items():
            result = clean if condition == clean_condition else binary_metrics(
                selected_labels, np.asarray(scores)[indices], threshold
            )
            for metric in metrics:
                samples[condition][metric].append(float(result[metric]))
                if condition != clean_condition:
                    deltas[condition][metric].append(float(result[metric]) - float(clean[metric]))
    output: dict[str, dict[str, Any]] = {}
    for condition in scores_by_condition:
        output[condition] = {"absolute": {}, "delta_from_clean": {}}
        for metric in metrics:
            output[condition]["absolute"][metric] = [
                float(value) for value in np.quantile(samples[condition][metric], [0.025, 0.975])
            ]
            if condition != clean_condition:
                output[condition]["delta_from_clean"][metric] = [
                    float(value) for value in np.quantile(deltas[condition][metric], [0.025, 0.975])
                ]
    return output
