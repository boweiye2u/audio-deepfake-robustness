# Methodology

## Research question

How much does a synthetic-speech detector degrade under realistic compression,
resampling, and additive-noise shifts, and can mixed-condition augmentation recover
robustness without materially reducing clean performance?

## Dataset protocol

The official ASVspoof 2019 LA train split fits parameters. The development split is
used for checkpoint selection and threshold analysis. The official evaluation split
is isolated and unavailable to the Phase 3 training runner. It will be used only after
the baseline and robustness protocol are frozen.

## Phase 3 baseline

The baseline is pinned `microsoft/wavlm-base-plus` with a frozen encoder and a
LayerNorm/dropout/linear binary head. Waveforms are loaded from root-relative
manifests, required to be mono 16 kHz, deterministically center-cropped when longer
than four seconds, and right-padded when shorter. Masked mean pooling excludes padded
frames. Frozen pooled features are cached outside Git.

Spoof is the positive class. Class-weighted binary cross-entropy uses
`bonafide_count / spoof_count` as `pos_weight`, equalizing aggregate class
contributions without changing the DEV distribution. Only the head is optimized.

The selected checkpoint is the epoch with the lowest DEV EER, then highest DEV
PR-AUC, then earliest epoch. Early stopping uses this same ordering with patience
five. After checkpoint selection, the operating threshold maximizes DEV balanced
accuracy; ties choose the threshold closest to 0.5 and then the lower threshold.
No different checkpoint is selected for another metric.

## Metrics

Threshold-free metrics are EER, ROC-AUC, and PR-AUC. EER is linearly interpolated at
the ROC crossing where false-positive and false-negative rates meet. Balanced
accuracy, precision, recall, F1, and the confusion matrix use the single selected DEV
threshold. All scores treat spoof as positive.

## Distribution shifts

The predefined later shifts are MP3 at 128/64/32 kbps, 16 kHz to 8 kHz to 16 kHz
round-trip resampling, and additive noise at 20/10/5 dB SNR. Phase 3 does not create
or evaluate these shifts.
