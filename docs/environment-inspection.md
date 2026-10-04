# Environment inspection — 2026-10-04

## Host

- Python base environment: 3.12.7
- Target project environment: Python 3.11
- Conda: 24.11.3
- GPUs: 8 × NVIDIA L40S, 46,068 MiB each
- NVIDIA driver: 565.57.01
- Host CUDA compatibility reported by nvidia-smi: 12.7
- Validated PyTorch stack available elsewhere on host: 2.9.1+cu126
- CPU cores: 128
- RAM: approximately 1 TiB
- Free disk under ~/audio: approximately 1.5 TiB

## Codec tooling

Neither ffmpeg nor ffprobe is installed on the host PATH. The isolated environment
specification includes ffmpeg from conda-forge; MP3 corruption work must not begin
until its encoders are verified after environment creation.

## Validated separation

Code is isolated under:

    ~/Dropbox/Non-coding-variant/audio-deepfake-robustness/

Large artifacts are isolated under:

    ~/audio/data/
    ~/audio/models/
    ~/audio/cache/
    ~/audio/outputs/

The code repository does not hard-code the local username. Environment variables
control all artifact roots.
