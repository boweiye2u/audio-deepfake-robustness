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


def test_baseline_config_expands_with_documented_roots(tmp_path, monkeypatch):
    roots = {
        "AUDIO_DATA_ROOT": tmp_path / "data",
        "AUDIO_MODEL_ROOT": tmp_path / "models",
        "AUDIO_CACHE_ROOT": tmp_path / "cache",
        "AUDIO_OUTPUT_ROOT": tmp_path / "outputs",
    }
    for name, value in roots.items():
        monkeypatch.setenv(name, str(value))
    repository = __import__("pathlib").Path(__file__).resolve().parents[1]
    config = load_config(repository / "configs" / "baseline.yaml")
    assert config["paths"]["train_manifest"].startswith(str(roots["AUDIO_DATA_ROOT"]))
    assert config["paths"]["checkpoint_dir"].startswith(str(roots["AUDIO_MODEL_ROOT"]))
