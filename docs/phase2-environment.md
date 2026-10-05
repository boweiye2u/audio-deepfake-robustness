# Phase 2 environment validation

The isolated prefix is `~/audio/envs/audio-deepfake`, created from
`environments/environment.yml`. Python 3.11 and ffmpeg come from Conda. PyTorch and
torchaudio are pinned to 2.9.1 CUDA 12.6 builds in the captured dependency files.

Validation on 2026-10-05 found:

- PyTorch 2.9.1+cu126 and torchaudio 2.9.1+cu126
- CUDA available on an NVIDIA L40S
- ffmpeg can encode and decode MP3 at 128, 64, and 32 kbps
- ffmpeg can perform a 16 kHz to 8 kHz to 16 kHz round trip
- optional native AAC and libopus encoders are available

AAC and Opus are reported capabilities, not core project requirements. The generated
codec-test signals lived only under `/tmp` and were removed after validation.
