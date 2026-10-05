# Robust Synthetic Speech Detection under Audio Distribution Shift

A reproducible study of how synthetic-speech detectors respond to realistic audio
transformations and whether targeted augmentation can improve robustness.

> **Status:** environment and ASVspoof data preparation pipeline established. No
> model, benchmark result, or robustness claim has been produced yet.

## Research question

How much does a clean-trained synthetic-speech detector degrade under compression,
resampling, and additive noise, and can mixed-condition training recover performance
without sacrificing clean accuracy?

## Planned experimental design

- **Dataset:** ASVspoof 2019 Logical Access, official protocols validated
- **Baseline:** compact CNN over log-Mel spectrograms
- **Model selection:** official development split only
- **Final evaluation:** untouched official evaluation split
- **Metrics:** EER, ROC-AUC, balanced accuracy, precision, recall, and F1
- **Shifts:** MP3 128/64/32 kbps, 8 kHz round-trip resampling, noise at 20/10/5 dB SNR
- **Mitigation:** mixed-condition corruption augmentation
- **Final comparison:** three seeds when practical, with bootstrap uncertainty

No result values will be added until the complete protocol has been executed and
validated.

## Filesystem separation

Only code, lightweight configuration, documentation, tests, small tables, and plots
belong in this repository. Audio, datasets, checkpoints, caches, and generated
corruptions remain outside Dropbox:

~~~text
Code:        ~/Dropbox/Non-coding-variant/audio-deepfake-robustness
Data:        ~/audio/data
Models:      ~/audio/models
Cache:       ~/audio/cache
Outputs:     ~/audio/outputs
Environment: ~/audio/envs/audio-deepfake
~~~

Set the artifact roots before running project commands:

~~~bash
export AUDIO_DATA_ROOT="${AUDIO_DATA_ROOT:-$HOME/audio/data}"
export AUDIO_MODEL_ROOT="${AUDIO_MODEL_ROOT:-$HOME/audio/models}"
export AUDIO_CACHE_ROOT="${AUDIO_CACHE_ROOT:-$HOME/audio/cache}"
export AUDIO_OUTPUT_ROOT="${AUDIO_OUTPUT_ROOT:-$HOME/audio/outputs}"
~~~

The repository contains no raw or transformed audio.

## Repository layout

~~~text
configs/       Versioned experiment definitions
src/           Installable Python package
scripts/       Data, training, evaluation, and reporting entry points
tests/         Fast unit and correctness tests
results/       Small metrics tables and figures only
docs/          Methodology and experiment documentation
environments/  Portable environment specification
~~~

## Environment

Python 3.11 is the target runtime. The isolated environment provides ffmpeg with
MP3 support for later compression experiments.

~~~bash
conda env create --prefix "$HOME/audio/envs/audio-deepfake" \
  -f environments/environment.yml
conda activate "$HOME/audio/envs/audio-deepfake"
python -m pip install -e '.[dev]'
python scripts/check_environment.py
python -m pytest
~~~

CUDA-enabled PyTorch 2.9.1 (CUDA 12.6) is installed in the isolated environment.
Dataset preparation is separate from model training, which remains deferred.

## Results

Results are intentionally blank during setup.

| System | Condition | EER | ROC-AUC | F1 | Relative degradation |
|---|---|---:|---:|---:|---:|
| Pending | Pending | — | — | — | — |

## Limitations

- Final claims remain pending corpus validation and model experiments.
- No clean or shifted benchmark has been run.
- Results will apply to the selected corpus, model, and corruption definitions rather
  than all synthetic speech in deployment.

See [the methodology](docs/methodology.md) for the frozen evaluation principles.
