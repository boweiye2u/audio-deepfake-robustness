# Phase 3 — clean single-GPU baseline on DEV

> **Scope:** All model selection, threshold selection, metrics, and error analysis use
the official ASVspoof 2019 LA development split. The evaluation split was not loaded.
No robustness corruption or augmentation was run.

## Model and training

The selected encoder is `microsoft/wavlm-base-plus`, revision
`4c66d4806a428f2e922ccfa1a962776e232d487b`. It was chosen before experiments from
three similar 16 kHz candidates because its 94,000-hour pretraining and explicit
content/speaker modeling fit the project better than Wav2Vec2 Base or HuBERT Base.

The 94,381,936-parameter encoder is frozen. Four-second mono 16 kHz waveforms are
normalized, deterministically center-cropped or right-padded, encoded in BF16, and
masked-mean pooled to 768 dimensions. The trainable head is LayerNorm, dropout 0.1,
and one linear output: **2,305 trainable parameters**.

The head uses AdamW, batch 512, learning rate 1e-3, weight decay 1e-4, BF16 training,
and class-weighted BCE with `pos_weight = 2580 / 22800 = 0.113158`. Spoof is positive.
The official train distribution is unchanged. The frozen rule selects the lowest DEV
EER, then highest PR-AUC, then earliest epoch. Epoch 30 was selected. The operating
threshold maximizes DEV balanced accuracy with a deterministic tie-break.

Smoke and balanced-subset gates passed before the full run. On the 1,024-train /
512-DEV subset, EER decreased from 0.3247 to 0.1078 to 0.0430 across three epochs.

## DEV results

The full run uses 25,380 train trials and 24,844 DEV trials.

| Metric | Value |
|---|---:|
| EER | 0.05220 |
| ROC-AUC | 0.98978 |
| PR-AUC / average precision | 0.99880 |
| Balanced accuracy | 0.95004 |
| Precision | 0.99616 |
| Recall | 0.93147 |
| F1 | 0.96273 |
| Selected threshold | 0.693743 |

The confusion matrix is `[[2468, 80], [1528, 20768]]`, with rows as true bonafide /
spoof and columns as predicted bonafide / spoof.

Feature extraction took 68.73 seconds and the corrected head run took 10.25 seconds,
for 78.98 seconds total measured pipeline time. Peak PyTorch GPU allocation was
2,550,470,656 bytes (about 2.38 GiB) on one NVIDIA L40S. A preliminary run evaluated
the head in BF16; it is preserved externally but superseded by the reported FP32 DEV
inference. Training settings and model selection were unchanged.

## DEV error analysis

Attack A06 accounts for 1,504 of the 1,528 spoof false negatives. Its one-vs-bonafide
EER is 0.1244 and recall at the global threshold is 0.5953. A01–A05 each have EER at
or below 0.0142 and recall at or above 0.9970. The baseline therefore learns most
known training attack families well but generalizes poorly to A06 even though A06 is
present in training.

Longer-than-four-second trials have the weakest duration-bin EER (0.0644), compared
with 0.0454–0.0488 for the shorter bins. This is descriptive and may reflect attack
composition as well as four-second cropping; it does not isolate duration as a cause.
Bonafide and spoof mean scores are 0.1346 and 0.9443. Eighty bonafide trials are false
positives and 1,528 spoof trials are false negatives.

## Reproducibility

The final run uses source commit `56d5969f2d6d69d309e1c4a302ebbc7d74e462e5`.
The unchanged feature cache was created by commit
`45b8679a61021c7a732f71bcbd6957bbcf60f6bc`. The selected checkpoint SHA-256 is
`0d16ce385c2171c4a161dd6ff4c66f8570b9b0b4f1b990c2e7bfb93117fa9800`.
All aggregates reconstruct exactly from 24,844 saved DEV records. CPU checkpoint
replay differs from saved GPU probabilities by at most 5.96e-7.

Artifacts:

- Checkpoint: `~/audio/models/audio-deepfake/wavlm-base-plus-frozen-v1/full/selected.pt`
- Feature cache: `~/audio/cache/features/wavlm-base-plus-frozen-v1/full/`
- Logs and DEV predictions: `~/audio/outputs/experiments/wavlm-base-plus-frozen-v1/full/`
- Repository metrics: `results/metrics/baseline-dev-summary.json`
- Per-attack and duration tables: `results/metrics/baseline-dev-by-{attack,duration}.csv`
- ROC, precision-recall, and score-distribution plots: `results/figures/baseline-dev-*.png`

Phase 3 stops here. The official evaluation split, robustness corruptions, and
robustness training remain untouched.
