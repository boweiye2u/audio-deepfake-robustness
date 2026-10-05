from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from audio_deepfake.asvspoof import (
    ARCHIVE_MD5,
    ARCHIVE_SIZE_BYTES,
    OFFICIAL_DOWNLOAD_URL,
    OFFICIAL_RECORD_URL,
    archive_uncompressed_size,
    build_all_manifests,
    download_archive,
    extract_verified_archive,
    validate_manifests,
    verify_archive,
    write_summary,
)
from audio_deepfake.config import ArtifactRoots


def locations() -> tuple[Path, Path, Path]:
    roots = ArtifactRoots.from_environment()
    dataset_root = roots.data / "asvspoof2019"
    archive = roots.cache / "downloads" / "asvspoof2019" / "LA.zip"
    summary = dataset_root / "manifests" / "dataset-summary.json"
    return dataset_root, archive, summary


def print_plan() -> None:
    dataset_root, archive, _ = locations()
    print(json.dumps({
        "official_record": OFFICIAL_RECORD_URL,
        "download_url": OFFICIAL_DOWNLOAD_URL,
        "archive_size_bytes": ARCHIVE_SIZE_BYTES,
        "archive_size_gib": ARCHIVE_SIZE_BYTES / 1024**3,
        "expected_md5": ARCHIVE_MD5,
        "archive_destination": str(archive),
        "extraction_destination": str(dataset_root / "LA"),
        "expected_extracted_size": "computed from the ZIP directory after download",
        "evaluation_guard": "eval manifest is isolated and requires final_evaluation purpose",
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the official ASVspoof 2019 LA corpus")
    parser.add_argument(
        "action", choices=("plan", "download", "verify", "extract", "manifest", "validate", "all")
    )
    args = parser.parse_args()
    dataset_root, archive, summary_path = locations()
    data_root = ArtifactRoots.from_environment().data
    if args.action == "plan":
        print_plan()
        return
    if args.action in {"download", "all"}:
        download_archive(OFFICIAL_DOWNLOAD_URL, archive)
    if args.action in {"verify", "extract", "all"}:
        print(f"verified MD5: {verify_archive(archive)}")
        print(f"archive uncompressed bytes: {archive_uncompressed_size(archive)}")
    if args.action in {"extract", "all"}:
        print(f"extracted to: {extract_verified_archive(archive, dataset_root)}")
    if args.action in {"manifest", "all"}:
        paths = build_all_manifests(dataset_root, data_root)
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    if args.action in {"validate", "all"}:
        result = validate_manifests(dataset_root, data_root)
        write_summary(result, summary_path)
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    os.umask(0o027)
    main()
