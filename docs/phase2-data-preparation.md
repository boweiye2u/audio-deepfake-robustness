# Phase 2 — environment and ASVspoof 2019 LA preparation

Phase 2 prepared data and validated infrastructure only. It did not download a
pretrained model, train a detector, generate corruptions, or score the final
evaluation partition.

## Source and integrity

The archive came from the official Zenodo record at
`https://zenodo.org/records/6906306`, using the record's content URL. The downloaded
size is 7,640,952,520 bytes and its MD5 is
`30c98f11d8b2bc21f2c257bfd78bb5c5`, exactly matching the published metadata. The
ZIP contains 7,655,650,274 uncompressed bytes. The original archive remains in the
external cache.

## Protocol validation

| Split | Protocol trials | Bonafide | Spoof | Speakers | Duration (h) | Attacks |
|---|---:|---:|---:|---:|---:|---|
| Train | 25,380 | 2,580 | 22,800 | 20 | 24.15 | A01–A06 |
| Dev | 24,844 | 2,548 | 22,296 | 20 | 24.00 | A01–A06 |
| Eval | 71,237 | 7,355 | 63,882 | 67 | 61.50 | A07–A19 |

All protocol-referenced files exist and are 16 kHz FLAC. Utterance IDs and speaker
IDs are disjoint across train, development, and evaluation. Full manifests use paths
relative to `AUDIO_DATA_ROOT` and remain outside Git. Train and development manifests
are under `manifests/development`; evaluation is isolated under
`manifests/final-evaluation` and the reader requires an explicit final-evaluation
purpose.

The release contains 142 DEV and 696 EVAL FLAC files that do not occur in the
corresponding official protocol. They are reported but excluded: the official
protocol, rather than directory contents, defines each trial partition. Train has no
unreferenced FLAC files.

## Validation

The environment provides Python 3.11, PyTorch 2.9.1+cu126, torchaudio 2.9.1+cu126,
CUDA 12.6, and ffmpeg. A real generated signal passed MP3 encode/decode at 128, 64,
and 32 kbps and a 16 kHz → 8 kHz → 16 kHz resampling round trip. Native AAC and
libopus encoders are also available but are optional.

Commands used:

```bash
conda env create --prefix "$HOME/audio/envs/audio-deepfake" -f environments/environment.yml
python -m pip install torch==2.9.1+cu126 torchaudio==2.9.1+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[dev]'
python scripts/check_environment.py
python scripts/prepare_asvspoof2019_la.py plan
python scripts/prepare_asvspoof2019_la.py all
python -m pytest -q
ruff check src scripts tests
```
