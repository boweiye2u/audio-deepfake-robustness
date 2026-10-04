import pytest

from audio_deepfake.config import ArtifactRoots, load_config


def test_artifact_roots_expand_without_username(monkeypatch, tmp_path):
    for name, leaf in {
        "AUDIO_DATA_ROOT": "data",
        "AUDIO_MODEL_ROOT": "models",
        "AUDIO_CACHE_ROOT": "cache",
        "AUDIO_OUTPUT_ROOT": "outputs",
    }.items():
        monkeypatch.setenv(name, str(tmp_path / leaf))
    roots = ArtifactRoots.from_environment()
    roots.validate(create=True)
    assert all(path.is_dir() for path in (roots.data, roots.models, roots.cache, roots.outputs))


def test_config_requires_referenced_environment_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIO_DATA_ROOT", raising=False)
    config = tmp_path / "config.yaml"
    config.write_text("manifest: ${AUDIO_DATA_ROOT}/manifest.csv\n")
    with pytest.raises(KeyError, match="AUDIO_DATA_ROOT"):
        load_config(config)


def test_config_expands_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_DATA_ROOT", str(tmp_path))
    config = tmp_path / "config.yaml"
    config.write_text("manifest: ${AUDIO_DATA_ROOT}/manifest.csv\n")
    assert load_config(config)["manifest"] == str(tmp_path / "manifest.csv")
