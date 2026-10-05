from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import soundfile as sf

OFFICIAL_RECORD_URL = "https://zenodo.org/records/6906306"
OFFICIAL_DOWNLOAD_URL = "https://zenodo.org/api/records/6906306/files/LA.zip/content"
ARCHIVE_SIZE_BYTES = 7_640_952_520
ARCHIVE_MD5 = "30c98f11d8b2bc21f2c257bfd78bb5c5"
SPLITS = ("train", "dev", "eval")
PROTOCOL_FILES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}
MANIFEST_FIELDS = (
    "utterance_id", "audio_path", "split", "label", "speaker_id", "attack_id",
    "duration_seconds", "sample_rate",
)


@dataclass(frozen=True)
class ProtocolRecord:
    speaker_id: str
    utterance_id: str
    attack_id: str
    label: str


@dataclass(frozen=True)
class ManifestRecord:
    utterance_id: str
    audio_path: str
    split: str
    label: str
    speaker_id: str
    attack_id: str
    duration_seconds: float
    sample_rate: int


def md5_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, expected_md5: str = ARCHIVE_MD5) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = md5_file(path)
    if actual.lower() != expected_md5.lower():
        raise ValueError(f"archive checksum mismatch: expected {expected_md5}, got {actual}")
    return actual


def download_archive(url: str, destination: Path, expected_size: int = ARCHIVE_SIZE_BYTES) -> None:
    """Download to a .part file, resuming when the server honors HTTP Range."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        if destination.stat().st_size != expected_size:
            raise ValueError(f"existing archive has unexpected size: {destination}")
        return
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise ValueError(f"partial download exceeds expected size: {partial}")
    request = urllib.request.Request(url, headers={"User-Agent": "audio-deepfake-robustness/0.1"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request) as response:
        status = getattr(response, "status", response.getcode())
        append = bool(offset and status == 206)
        if offset and not append:
            offset = 0
        with partial.open("ab" if append else "wb") as handle:
            while chunk := response.read(8 * 1024 * 1024):
                handle.write(chunk)
    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"incomplete download retained at {partial}: expected {expected_size}, got {actual_size}"
        )
    os.replace(partial, destination)


def archive_uncompressed_size(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return sum(item.file_size for item in archive.infolist() if not item.is_dir())


def _safe_members(archive: zipfile.ZipFile, destination: Path) -> list[zipfile.ZipInfo]:
    root = destination.resolve()
    members = archive.infolist()
    for member in members:
        name = PurePosixPath(member.filename)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"unsafe archive member: {member.filename}")
        target = (destination / Path(*name.parts)).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"archive member escapes destination: {member.filename}")
    return members


def extract_verified_archive(archive_path: Path, dataset_root: Path) -> Path:
    """Verify MD5 first, then extract without exposing a partial final LA tree."""
    verify_archive(archive_path)
    final = dataset_root / "LA"
    if final.exists():
        return final
    dataset_root.mkdir(parents=True, exist_ok=True)
    staging = dataset_root / ".extracting-LA"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(staging, members=_safe_members(archive, staging))
        extracted = staging / "LA"
        if not extracted.is_dir():
            raise ValueError("archive does not contain the expected LA directory")
        extracted.rename(final)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return final


def _validate_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")


def protocol_path(la_root: Path, split: str) -> Path:
    _validate_split(split)
    return la_root / "ASVspoof2019_LA_cm_protocols" / PROTOCOL_FILES[split]


def audio_directory(la_root: Path, split: str) -> Path:
    _validate_split(split)
    return la_root / f"ASVspoof2019_LA_{split}" / "flac"


def parse_protocol(path: Path, split: str) -> list[ProtocolRecord]:
    _validate_split(split)
    records: list[ProtocolRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields, got {len(fields)}")
        speaker_id, utterance_id, _unused, attack_id, label = fields
        if label not in {"bonafide", "spoof"}:
            raise ValueError(f"{path}:{line_number}: invalid label {label!r}")
        if label == "bonafide" and attack_id != "-":
            raise ValueError(f"{path}:{line_number}: bonafide row has attack ID {attack_id!r}")
        if label == "spoof" and attack_id == "-":
            raise ValueError(f"{path}:{line_number}: spoof row lacks an attack ID")
        if utterance_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate utterance {utterance_id}")
        seen.add(utterance_id)
        records.append(ProtocolRecord(speaker_id, utterance_id, attack_id, label))
    if not records:
        raise ValueError(f"empty protocol: {path}")
    return records


def build_manifest_records(
    records: Iterable[ProtocolRecord], split: str, la_root: Path, audio_data_root: Path
) -> list[ManifestRecord]:
    _validate_split(split)
    output: list[ManifestRecord] = []
    for record in records:
        path = audio_directory(la_root, split) / f"{record.utterance_id}.flac"
        if not path.is_file():
            raise FileNotFoundError(f"protocol audio is missing: {path}")
        try:
            relative = path.resolve().relative_to(audio_data_root.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError(f"audio path is outside AUDIO_DATA_ROOT: {path}") from exc
        info = sf.info(path)
        output.append(ManifestRecord(
            utterance_id=record.utterance_id,
            audio_path=relative,
            split=split,
            label=record.label,
            speaker_id=record.speaker_id,
            attack_id=record.attack_id,
            duration_seconds=info.frames / info.samplerate,
            sample_rate=info.samplerate,
        ))
    return output


def write_manifest(records: Iterable[ManifestRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["duration_seconds"] = f"{record.duration_seconds:.8f}"
            writer.writerow(row)
    os.replace(temporary, path)


def read_manifest(path: Path, purpose: str = "development") -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError(f"invalid manifest schema: {path}")
        rows = list(reader)
    if "eval" in {row["split"] for row in rows} and purpose != "final_evaluation":
        raise PermissionError("evaluation manifest requires purpose='final_evaluation'")
    return rows


def manifest_paths(dataset_root: Path) -> dict[str, Path]:
    root = dataset_root / "manifests"
    return {
        "train": root / "development" / "train.csv",
        "dev": root / "development" / "dev.csv",
        "eval": root / "final-evaluation" / "eval.csv",
    }


def build_all_manifests(dataset_root: Path, audio_data_root: Path) -> dict[str, Path]:
    la_root = dataset_root / "LA"
    paths = manifest_paths(dataset_root)
    for split in SPLITS:
        records = parse_protocol(protocol_path(la_root, split), split)
        write_manifest(build_manifest_records(records, split, la_root, audio_data_root), paths[split])
    return paths


def validate_manifests(dataset_root: Path, audio_data_root: Path) -> dict[str, object]:
    la_root = dataset_root / "LA"
    paths = manifest_paths(dataset_root)
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    for split in SPLITS:
        directory = audio_directory(la_root, split)
        protocol = protocol_path(la_root, split)
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        if not protocol.is_file():
            raise FileNotFoundError(protocol)
        purpose = "final_evaluation" if split == "eval" else "development"
        rows = read_manifest(paths[split], purpose=purpose)
        if any(row["split"] != split for row in rows):
            raise ValueError(f"manifest contains wrong split rows: {paths[split]}")
        for row in rows:
            if row["label"] not in {"bonafide", "spoof"}:
                raise ValueError(f"invalid label in {paths[split]}")
            if float(row["duration_seconds"]) <= 0 or int(row["sample_rate"]) <= 0:
                raise ValueError(f"invalid audio metadata in {paths[split]}")
            path = audio_data_root / row["audio_path"]
            if not path.is_file():
                raise FileNotFoundError(path)
        manifest_ids = {row["utterance_id"] for row in rows}
        audio_ids = {path.stem for path in directory.glob("*.flac")}
        missing_audio = manifest_ids - audio_ids
        if missing_audio:
            raise ValueError(f"{len(missing_audio)} protocol entries lack audio in {split}")
        rows_by_split[split] = rows

    utterances = {s: {row["utterance_id"] for row in rows} for s, rows in rows_by_split.items()}
    speakers = {s: {row["speaker_id"] for row in rows} for s, rows in rows_by_split.items()}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            overlap = utterances[left] & utterances[right]
            if overlap:
                raise ValueError(f"utterance overlap between {left} and {right}: {len(overlap)}")
            overlap = speakers[left] & speakers[right]
            if overlap:
                raise ValueError(f"speaker overlap between {left} and {right}: {len(overlap)}")

    split_summary: dict[str, object] = {}
    for split, rows in rows_by_split.items():
        labels = Counter(row["label"] for row in rows)
        attacks = Counter(row["attack_id"] for row in rows)
        rates = Counter(row["sample_rate"] for row in rows)
        duration = sum(float(row["duration_seconds"]) for row in rows)
        split_summary[split] = {
            "utterances": len(rows),
            "audio_files": len({path.stem for path in audio_directory(la_root, split).glob("*.flac")}),
            "unreferenced_audio_files": len(
                {path.stem for path in audio_directory(la_root, split).glob("*.flac")}
                - {row["utterance_id"] for row in rows}
            ),
            "speakers": len(speakers[split]),
            "labels": dict(sorted(labels.items())),
            "attack_ids": dict(sorted(attacks.items())),
            "sample_rates_hz": dict(sorted(rates.items())),
            "duration_hours": duration / 3600,
        }
    return {
        "dataset": "ASVspoof 2019 Logical Access",
        "source": OFFICIAL_RECORD_URL,
        "official_partitions_preserved": True,
        "utterance_overlap": False,
        "speaker_overlap": False,
        "splits": split_summary,
    }


def write_summary(summary: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
