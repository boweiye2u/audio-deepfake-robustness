from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise KeyError(f"required environment variable is not set: {name}")
        return os.environ[name]

    return str(Path(_ENV_PATTERN.sub(replace, value)).expanduser())


def _walk(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _walk(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_walk(item) for item in value]
    if isinstance(value, str):
        return _expand(value) if "~" in value or "${" in value else value
    return value


@dataclass(frozen=True)
class ArtifactRoots:
    data: Path
    models: Path
    cache: Path
    outputs: Path

    @classmethod
    def from_environment(cls) -> "ArtifactRoots":
        defaults = {
            "AUDIO_DATA_ROOT": "~/audio/data",
            "AUDIO_MODEL_ROOT": "~/audio/models",
            "AUDIO_CACHE_ROOT": "~/audio/cache",
            "AUDIO_OUTPUT_ROOT": "~/audio/outputs",
        }
        values = {
            name: Path(os.path.expandvars(os.path.expanduser(os.environ.get(name, default))))
            for name, default in defaults.items()
        }
        return cls(
            data=values["AUDIO_DATA_ROOT"],
            models=values["AUDIO_MODEL_ROOT"],
            cache=values["AUDIO_CACHE_ROOT"],
            outputs=values["AUDIO_OUTPUT_ROOT"],
        )

    def validate(self, create: bool = False) -> None:
        for path in (self.data, self.models, self.cache, self.outputs):
            if create:
                path.mkdir(parents=True, exist_ok=True)
            if not path.is_dir():
                raise FileNotFoundError(f"artifact root does not exist: {path}")


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return _walk(value)
