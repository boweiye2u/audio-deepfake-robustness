# Methodology

## Research question

How much does a synthetic-speech detector degrade under realistic compression,
resampling, and additive-noise shifts, and can mixed-condition augmentation recover
robustness without materially reducing clean performance?

## Planned protocol

- Primary corpus: ASVspoof 2019 Logical Access, subject to access validation.
- Preserve official train, development, and evaluation protocols.
- Select checkpoints and operating thresholds using development data only.
- Evaluate the frozen selected systems once on the official evaluation split.
- Report EER, ROC-AUC, balanced accuracy, precision, recall, and F1.
- Compare the clean-trained and mixed-condition systems under identical conditions.
- Use three final seeds and bootstrap confidence intervals when practical.

## Initial model

The initial baseline is a compact CNN over log-Mel spectrograms. This avoids making
pretrained model scale the main result and keeps the robustness experiment auditable.
A pretrained encoder may be added later only as a separately versioned comparison.

## Distribution shifts

The predefined core shifts are MP3 at 128/64/32 kbps, 16 kHz to 8 kHz to 16 kHz
round-trip resampling, and additive noise at 20/10/5 dB SNR. Corruptions are generated
from frozen source splits with deterministic parameters and stored outside Git.

## Leakage controls

The official evaluation split is not used for checkpoint selection, threshold selection,
augmentation design, or corruption-policy tuning. Manifests record speaker and attack
metadata so overlap and subgroup checks can be performed before training.
