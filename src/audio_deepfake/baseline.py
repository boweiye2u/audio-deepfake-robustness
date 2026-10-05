from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch import nn

from audio_deepfake.asvspoof import read_manifest

LABEL_TO_INDEX = {"bonafide": 0, "spoof": 1}
DEVELOPMENT_SPLITS = {"train", "dev"}


@dataclass(frozen=True)
class AudioRecord:
    utterance_id: str
    audio_path: Path
    split: str
    label: int
    speaker_id: str
    attack_id: str
    duration_seconds: float
    sample_rate: int


def label_to_index(label: str) -> int:
    try:
        return LABEL_TO_INDEX[label]
    except KeyError as exc:
        raise ValueError(f"unknown label: {label}") from exc


def load_development_manifest(path: Path, audio_data_root: Path) -> list[AudioRecord]:
    rows = read_manifest(path, purpose="development")
    records: list[AudioRecord] = []
    for row in rows:
        if row["split"] not in DEVELOPMENT_SPLITS:
            raise PermissionError(f"training pipeline refuses split {row['split']!r}")
        audio_path = audio_data_root / row["audio_path"]
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        records.append(
            AudioRecord(
                utterance_id=row["utterance_id"],
                audio_path=audio_path,
                split=row["split"],
                label=label_to_index(row["label"]),
                speaker_id=row["speaker_id"],
                attack_id=row["attack_id"],
                duration_seconds=float(row["duration_seconds"]),
                sample_rate=int(row["sample_rate"]),
            )
        )
    return records


def pad_or_crop(waveform: np.ndarray, length: int, strategy: str = "center") -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError("waveform must be mono")
    if length <= 0:
        raise ValueError("target length must be positive")
    if len(waveform) > length:
        if strategy != "center":
            raise ValueError(f"unsupported crop strategy: {strategy}")
        start = (len(waveform) - length) // 2
        return waveform[start : start + length].copy()
    if len(waveform) < length:
        return np.pad(waveform, (0, length - len(waveform))).astype(np.float32, copy=False)
    return waveform.copy()


def load_waveform(
    path: Path, sample_rate: int, max_samples: int, crop_strategy: str = "center"
) -> np.ndarray:
    waveform, actual_rate = sf.read(path, dtype="float32", always_2d=False)
    if actual_rate != sample_rate:
        raise ValueError(f"expected {sample_rate} Hz, got {actual_rate} Hz: {path}")
    if waveform.ndim != 1:
        raise ValueError(f"expected mono audio: {path}")
    return pad_or_crop(waveform, max_samples, crop_strategy)


class AudioBatchDataset(torch.utils.data.Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[AudioRecord],
        sample_rate: int,
        max_duration_seconds: float,
        crop_strategy: str = "center",
    ) -> None:
        self.records = records
        self.sample_rate = sample_rate
        self.max_samples = round(sample_rate * max_duration_seconds)
        self.crop_strategy = crop_strategy

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        waveform = load_waveform(
            record.audio_path, self.sample_rate, self.max_samples, self.crop_strategy
        )
        valid_samples = min(round(record.duration_seconds * self.sample_rate), self.max_samples)
        attention_mask = torch.zeros(self.max_samples, dtype=torch.long)
        attention_mask[:valid_samples] = 1
        valid_waveform = waveform[:valid_samples]
        mean = float(valid_waveform.mean()) if valid_samples else 0.0
        variance = float(valid_waveform.var()) if valid_samples else 0.0
        waveform[:valid_samples] = (valid_waveform - mean) / np.sqrt(variance + 1e-7)
        return {
            "waveform": torch.from_numpy(waveform),
            "attention_mask": attention_mask,
            "label": record.label,
            "index": index,
        }


class BinaryClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(embeddings).squeeze(-1)


def save_head_checkpoint(
    path: Path,
    model: BinaryClassificationHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metadata": metadata,
        },
        temporary,
    )
    temporary.replace(path)


def load_head_checkpoint(
    path: Path, hidden_size: int, dropout: float, device: torch.device
) -> tuple[BinaryClassificationHead, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = BinaryClassificationHead(hidden_size, dropout).to(device)
    model.load_state_dict(payload["model_state"])
    return model, payload


def write_records_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty record table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
