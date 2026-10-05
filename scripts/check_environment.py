#!/usr/bin/env python
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from audio_deepfake.config import ArtifactRoots


def command_version(command: str, flag: str = "--version") -> str | None:
    executable = shutil.which(command)
    if executable is None:
        environment_executable = Path(sys.prefix) / "bin" / command
        executable = str(environment_executable) if environment_executable.is_file() else None
    if executable is None:
        return None
    result = subprocess.run([executable, flag], check=False, capture_output=True, text=True)
    return (result.stdout or result.stderr).splitlines()[0]


def main() -> None:
    roots = ArtifactRoots.from_environment()
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ffmpeg": command_version("ffmpeg", "-version"),
        "ffprobe": command_version("ffprobe", "-version"),
        "roots": {
            name: {"path": str(path), "exists": path.is_dir()}
            for name, path in {
                "data": roots.data,
                "models": roots.models,
                "cache": roots.cache,
                "outputs": roots.outputs,
            }.items()
        },
    }
    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
    except ImportError:
        report["torch"] = None

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
