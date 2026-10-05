import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_deepfake.asvspoof import (
    MANIFEST_FIELDS,
    build_all_manifests,
    extract_verified_archive,
    md5_file,
    parse_protocol,
    read_manifest,
    validate_manifests,
    verify_archive,
)


def _fixture_dataset(tmp_path: Path, overlap: bool = False) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    dataset_root = data_root / "asvspoof2019"
    la_root = dataset_root / "LA"
    protocol_root = la_root / "ASVspoof2019_LA_cm_protocols"
    protocol_root.mkdir(parents=True)
    split_rows = {
        "train": [("TR1", "LA_T_1", "-", "bonafide"), ("TR2", "LA_T_2", "A01", "spoof")],
        "dev": [("DV1", "LA_D_1", "-", "bonafide"), ("DV2", "LA_D_2", "A02", "spoof")],
        "eval": [("EV1", "LA_E_1", "-", "bonafide"), ("EV2", "LA_E_2", "A07", "spoof")],
    }
    if overlap:
        split_rows["dev"][0] = ("TR1", "LA_T_1", "-", "bonafide")
    protocol_names = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
    }
    audio = np.zeros(1600, dtype=np.float32)
    for split, rows in split_rows.items():
        audio_dir = la_root / f"ASVspoof2019_LA_{split}" / "flac"
        audio_dir.mkdir(parents=True)
        lines = []
        for speaker, utterance, attack, label in rows:
            sf.write(audio_dir / f"{utterance}.flac", audio, 16000)
            lines.append(f"{speaker} {utterance} - {attack} {label}")
        (protocol_root / protocol_names[split]).write_text("\n".join(lines) + "\n")
    return data_root, dataset_root


def test_protocol_parsing_rejects_bad_label(tmp_path):
    protocol = tmp_path / "protocol.txt"
    protocol.write_text("SPK UTT - A01 unknown\n")
    with pytest.raises(ValueError, match="invalid label"):
        parse_protocol(protocol, "train")


def test_checksum_utility(tmp_path):
    archive = tmp_path / "sample.bin"
    archive.write_bytes(b"abc")
    expected = hashlib.md5(b"abc").hexdigest()
    assert md5_file(archive) == expected
    assert verify_archive(archive, expected) == expected
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_archive(archive, "0" * 32)


def test_checksum_failure_refuses_extraction(tmp_path):
    archive = tmp_path / "invalid.zip"
    archive.write_bytes(b"not the official archive")
    dataset_root = tmp_path / "dataset"
    with pytest.raises(ValueError, match="checksum mismatch"):
        extract_verified_archive(archive, dataset_root)
    assert not dataset_root.exists()


def test_manifest_schema_paths_and_eval_guard(tmp_path):
    data_root, dataset_root = _fixture_dataset(tmp_path)
    paths = build_all_manifests(dataset_root, data_root)
    with paths["train"].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        assert tuple(rows[0]) == MANIFEST_FIELDS
        assert rows[0]["audio_path"].startswith("asvspoof2019/")
        assert not Path(rows[0]["audio_path"]).is_absolute()
    with pytest.raises(PermissionError, match="final_evaluation"):
        read_manifest(paths["eval"])
    assert len(read_manifest(paths["eval"], purpose="final_evaluation")) == 2


def test_split_and_speaker_isolation(tmp_path):
    data_root, dataset_root = _fixture_dataset(tmp_path)
    build_all_manifests(dataset_root, data_root)
    summary = validate_manifests(dataset_root, data_root)
    assert summary["utterance_overlap"] is False
    assert summary["speaker_overlap"] is False
    assert summary["splits"]["train"]["labels"] == {"bonafide": 1, "spoof": 1}


def test_split_overlap_is_rejected(tmp_path):
    data_root, dataset_root = _fixture_dataset(tmp_path, overlap=True)
    build_all_manifests(dataset_root, data_root)
    with pytest.raises(ValueError, match="utterance overlap"):
        validate_manifests(dataset_root, data_root)
