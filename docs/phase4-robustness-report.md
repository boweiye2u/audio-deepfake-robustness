# Phase 4 — frozen-baseline robustness on DEV

> **Scope:** This evaluation uses the same 24,844 official ASVspoof 2019 LA DEV trials used in Phase 3. The frozen detector, checkpoint, preprocessing, and operating threshold are unchanged. No training occurred and the official EVAL split was not loaded.

## Experimental question

How does the clean-trained detector respond to lossy compression, an 8 kHz transmission-like round trip, and additive Gaussian noise, and which transformations cause the largest degradation?

## Frozen baseline and protocol

The system is the Phase 3 `microsoft/wavlm-base-plus` encoder at revision `4c66d4806a428f2e922ccfa1a962776e232d487b`, frozen with masked-mean pooling and the selected linear head from epoch 30. The checkpoint is:

```text
~/audio/models/audio-deepfake/wavlm-base-plus-frozen-v1/full/selected.pt
SHA-256: 0d16ce385c2171c4a161dd6ff4c66f8570b9b0b4f1b990c2e7bfb93117fa9800
```

All threshold-dependent metrics use the exact stored clean DEV threshold **0.693742573261261**. The threshold was never re-optimized by condition. Condition-specific EER thresholds are retained only as diagnostics in `results/metrics/robustness-main.csv`.

The Phase 4 clean path reproduced all Phase 3 aggregate metrics and the confusion matrix exactly. Its maximum absolute score difference from the saved Phase 3 predictions was **3.28e-7**, below the predefined 1e-6 tolerance.

## Corruptions

Every transformation is applied in memory to the complete waveform before the unchanged four-second center crop, normalization, WavLM encoder, and classifier.

- MP3 at 128, 64, and 32 kbps uses ffmpeg/libmp3lame encode/decode through pipes.
- Resampling uses a polyphase 16 kHz → 8 kHz → 16 kHz round trip.
- Additive zero-mean Gaussian noise targets 20, 10, or 5 dB SNR. A SHA-256-derived seed fixes the noise separately for every utterance and condition.

Decoded signals are aligned back to the original length. Preflight validation covered four DEV utterances under all eight conditions and checked sample rate, length, finite values, non-silent RMS, loadability, and measured SNR. AAC and Opus were omitted because they were secondary and would add two further full-corpus codec passes without changing the required experiment.

## Main robustness results

Metrics use spoof as the positive class. Deltas are condition minus clean; higher EER and lower ROC-AUC, recall, or F1 indicate degradation.

| Condition | EER | Δ EER | ROC-AUC | Δ ROC | PR-AUC | Bal. acc. | Precision | Recall | Δ recall | F1 | Δ F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean | 0.0522 | — | 0.9898 | — | 0.9988 | 0.9500 | 0.9962 | 0.9315 | — | 0.9627 | — |
| MP3 128k | 0.0496 | -0.0026 | 0.9906 | +0.0009 | 0.9989 | 0.9526 | 0.9964 | 0.9347 | +0.0032 | 0.9645 | +0.0018 |
| MP3 64k | 0.0492 | -0.0030 | 0.9908 | +0.0010 | 0.9989 | 0.9536 | 0.9967 | 0.9339 | +0.0024 | 0.9643 | +0.0016 |
| MP3 32k | 0.0440 | -0.0082 | 0.9921 | +0.0023 | 0.9991 | 0.9509 | 0.9923 | 0.9678 | +0.0363 | 0.9799 | +0.0171 |
| 8 kHz round trip | 0.0432 | -0.0090 | 0.9924 | +0.0026 | 0.9991 | 0.9502 | 0.9915 | 0.9729 | +0.0414 | 0.9821 | +0.0194 |
| Noise 20 dB | 0.1035 | +0.0513 | 0.9591 | -0.0306 | 0.9948 | 0.5130 | 0.8998 | 0.9997 | +0.0682 | 0.9471 | -0.0156 |
| Noise 10 dB | 0.2260 | +0.1738 | 0.8206 | -0.1692 | 0.9779 | 0.4964 | 0.8968 | 0.9841 | +0.0527 | 0.9384 | -0.0243 |
| Noise 5 dB | 0.3902 | +0.3380 | 0.6603 | -0.3295 | 0.9515 | 0.5154 | 0.9007 | 0.8848 | -0.0466 | 0.8927 | -0.0701 |

The largest degradation is 5 dB noise by EER, ROC-AUC, recall, and F1. The smallest change is MP3 128k if absolute EER change is used; its paired interval indicates a small improvement rather than harm. The strongest apparent improvement is the 8 kHz round trip. These improvements are development-set observations, not evidence that destructive preprocessing is generally beneficial.

## Paired uncertainty

Intervals use 2,000 class-stratified paired utterance bootstrap replicates with seed `20261009`. Each sampled utterance index is shared between clean and corrupted scores. Delta intervals therefore measure the paired condition-minus-clean change.

| Condition | EER 95% CI | Δ EER 95% CI | ROC-AUC 95% CI | Δ ROC 95% CI | Recall 95% CI | Δ recall 95% CI | F1 95% CI | Δ F1 95% CI |
|---|---|---|---|---|---|---|---|---|
| Clean | [0.0483, 0.0557] | — | [0.9886, 0.9910] | — | [0.9281, 0.9346] | — | [0.9609, 0.9645] | — |
| MP3 128k | [0.0455, 0.0530] | [-0.0055, -0.0001] | [0.9896, 0.9917] | [+0.0004, +0.0014] | [0.9314, 0.9379] | [+0.0014, +0.0050] | [0.9627, 0.9663] | [+0.0008, +0.0028] |
| MP3 64k | [0.0453, 0.0530] | [-0.0055, -0.0002] | [0.9898, 0.9918] | [+0.0005, +0.0016] | [0.9305, 0.9372] | [+0.0006, +0.0043] | [0.9625, 0.9661] | [+0.0005, +0.0026] |
| MP3 32k | [0.0408, 0.0480] | [-0.0110, -0.0044] | [0.9911, 0.9930] | [+0.0017, +0.0029] | [0.9654, 0.9701] | [+0.0337, +0.0388] | [0.9786, 0.9812] | [+0.0156, +0.0185] |
| 8 kHz round trip | [0.0389, 0.0467] | [-0.0124, -0.0063] | [0.9914, 0.9933] | [+0.0021, +0.0032] | [0.9708, 0.9750] | [+0.0386, +0.0441] | [0.9809, 0.9833] | [+0.0179, +0.0209] |
| Noise 20 dB | [0.0979, 0.1091] | [+0.0457, +0.0573] | [0.9560, 0.9623] | [-0.0337, -0.0276] | [0.9994, 0.9999] | [+0.0651, +0.0716] | [0.9468, 0.9475] | [-0.0174, -0.0137] |
| Noise 10 dB | [0.2198, 0.2323] | [+0.1667, +0.1808] | [0.8153, 0.8263] | [-0.1745, -0.1636] | [0.9825, 0.9858] | [+0.0489, +0.0565] | [0.9376, 0.9393] | [-0.0262, -0.0223] |
| Noise 5 dB | [0.3836, 0.3968] | [+0.3307, +0.3457] | [0.6523, 0.6686] | [-0.3377, -0.3214] | [0.8809, 0.8891] | [-0.0518, -0.0409] | [0.8904, 0.8952] | [-0.0730, -0.0668] |

## Attack-specific behavior

Each attack row is evaluated against all 2,548 bonafide trials, as in Phase 3.

**EER by attack**

| Condition | A01 | A02 | A03 | A04 | A05 | A06 |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 0.0063 | 0.0059 | 0.0141 | 0.0110 | 0.0039 | 0.1244 |
| MP3 128k | 0.0039 | 0.0040 | 0.0140 | 0.0100 | 0.0043 | 0.1233 |
| MP3 64k | 0.0032 | 0.0031 | 0.0124 | 0.0090 | 0.0032 | 0.1228 |
| MP3 32k | 0.0024 | 0.0024 | 0.0108 | 0.0078 | 0.0031 | 0.1114 |
| 8 kHz | 0.0057 | 0.0039 | 0.0164 | 0.0116 | 0.0038 | 0.1063 |
| Noise 20 dB | 0.0579 | 0.0793 | 0.1025 | 0.0587 | 0.0339 | 0.2142 |
| Noise 10 dB | 0.2077 | 0.2021 | 0.1819 | 0.1846 | 0.3385 | 0.2464 |
| Noise 5 dB | 0.3202 | 0.3956 | 0.3120 | 0.3345 | 0.5871 | 0.4058 |

**Recall / false negatives by attack**

| Condition | A01 | A02 | A03 | A04 | A05 | A06 |
|---|---|---|---|---|---|---|
| Clean | .9995 / 2 | 1.0000 / 0 | .9978 / 8 | .9970 / 11 | .9992 / 3 | .5953 / 1504 |
| MP3 128k | .9995 / 2 | 1.0000 / 0 | .9957 / 16 | .9968 / 12 | .9992 / 3 | .6168 / 1424 |
| MP3 64k | .9995 / 2 | 1.0000 / 0 | .9973 / 10 | .9960 / 15 | .9992 / 3 | .6114 / 1444 |
| MP3 32k | 1.0000 / 0 | 1.0000 / 0 | .9992 / 3 | .9995 / 2 | .9992 / 3 | .8087 / 711 |
| 8 kHz | .9997 / 1 | 1.0000 / 0 | .9995 / 2 | .9981 / 7 | .9992 / 3 | .8410 / 591 |
| Noise 20 dB | .9992 / 3 | 1.0000 / 0 | 1.0000 / 0 | .9997 / 1 | 1.0000 / 0 | .9992 / 3 |
| Noise 10 dB | .9930 / 26 | .9812 / 70 | .9852 / 55 | .9943 / 21 | .9618 / 142 | .9892 / 40 |
| Noise 5 dB | .9766 / 87 | .8283 / 638 | .9279 / 268 | .9639 / 134 | .7158 / 1056 | .8964 / 385 |

A06 remains the hardest clean and codec/resampling attack family by EER. MP3 32k and 8 kHz raise A06 recall substantially, but the frozen threshold also produces more bonafide false positives. At 20 dB noise, nearly every spoof crosses the fixed threshold while 2,481/2,548 bonafide trials also cross it; recall alone is therefore misleading. At 10 and 5 dB, previously easy A01–A05 lose ranking separation, with A05 becoming the hardest at 5 dB. Corruption changes the attack-difficulty ordering rather than simply amplifying A06.

## A06 paired transitions

| Condition | Clean FN still FN | Clean TP → FN | Clean FN → TP |
|---|---:|---:|---:|
| MP3 128k | 1261 | 163 | 243 |
| MP3 64k | 1263 | 181 | 241 |
| MP3 32k | 687 | 24 | 817 |
| 8 kHz | 569 | 22 | 935 |
| Noise 20 dB | 1 | 2 | 1503 |
| Noise 10 dB | 10 | 30 | 1494 |
| Noise 5 dB | 132 | 253 | 1372 |

The noise results do not show that the A06 representation weakness was solved. Noise drives scores toward the spoof side for both classes, converting many A06 false negatives while simultaneously destroying bonafide specificity and reducing ranking quality.

## Score shifts and paired errors

Clean mean scores are 0.1346 for bonafide and 0.9443 for spoof. MP3 128/64k barely changes either mean. MP3 32k and 8 kHz move bonafide means upward by 0.0837 and 0.0890 and spoof means upward by 0.0256 and 0.0279. Noise causes the dominant calibration shift: at 20 dB, bonafide mean rises by 0.8249 to 0.9595; at 10 dB it rises by 0.8576 to 0.9923. At 5 dB, mean scores are 0.8765 bonafide and 0.9029 spoof, so separation is greatly reduced as well as shifted.

Overall paired correctness transitions are:

| Condition | correct→correct | correct→incorrect | incorrect→correct | incorrect→incorrect |
|---|---:|---:|---:|---:|
| MP3 128k | 23040 | 196 | 272 | 1336 |
| MP3 64k | 23029 | 207 | 273 | 1335 |
| MP3 32k | 23110 | 126 | 847 | 761 |
| 8 kHz | 23095 | 141 | 960 | 648 |
| Noise 20 dB | 20829 | 2407 | 1527 | 81 |
| Noise 10 dB | 20445 | 2791 | 1519 | 89 |
| Noise 5 dB | 18688 | 4548 | 1412 | 196 |

The full class- and attack-specific transition table is saved in `results/metrics/robustness-transitions.csv`.

## Duration analysis

| Condition | EER (0,2] | EER (2,3] | EER (3,4] | EER (4,∞] |
|---|---:|---:|---:|---:|
| Clean | 0.0472 | 0.0454 | 0.0488 | 0.0644 |
| MP3 128k | 0.0315 | 0.0440 | 0.0448 | 0.0664 |
| MP3 64k | 0.0315 | 0.0429 | 0.0451 | 0.0677 |
| MP3 32k | 0.0315 | 0.0384 | 0.0400 | 0.0638 |
| 8 kHz | 0.0472 | 0.0416 | 0.0395 | 0.0514 |
| Noise 20 dB | 0.1260 | 0.1079 | 0.1047 | 0.0951 |
| Noise 10 dB | 0.2892 | 0.2701 | 0.1790 | 0.1127 |
| Noise 5 dB | 0.3006 | 0.3932 | 0.3574 | 0.2293 |

Duration and attack composition are confounded, and cropping affects only longer trials. These descriptive bins do not establish a causal duration effect.

## Runtime, artifacts, and validation

The seven corrupted conditions required **582.1 seconds** of measured feature extraction on one NVIDIA L40S; the separate clean reproduction required 35.1 seconds. The full cached-condition analysis, including bootstrap and plotting, completed in about 16.5 minutes wall time. Peak PyTorch GPU allocation was **2,532,764,160 bytes** (about 2.36 GiB). Robustness caches use **588 MiB** and external run outputs use **13 MiB**. No transformed audio corpus was written.

Commands:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=$HOME/audio/cache/huggingface \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  $HOME/audio/envs/audio-deepfake/bin/python scripts/run_robustness_4.py --phase preflight

# Same environment variables for these phases:
$HOME/audio/envs/audio-deepfake/bin/python scripts/run_robustness_4.py --phase clean
$HOME/audio/envs/audio-deepfake/bin/python scripts/run_robustness_4.py --phase full
python -m pytest -q
ruff check .
```

Artifacts:

- Frozen configuration: `configs/robustness_eval.yaml`
- External predictions and summaries: `~/audio/outputs/experiments/wavlm-base-plus-robustness-v1/`
- Feature caches: `~/audio/cache/features/wavlm-base-plus-robustness-v1/`
- Main, attack, duration, transition, score-shift, and bootstrap results: `results/metrics/robustness-*`
- Five requested figures: `results/figures/robustness-*`

All sample IDs and labels align across conditions. The runner refuses a checkpoint hash mismatch, records that EVAL was not accessed, and uses a DEV-only config. The final test and Ruff results are recorded with the phase commit.

## Key findings and limitations

The frozen model is stable under the tested MP3 and resampling transformations on this DEV set. Gaussian noise is the clear weakness: it causes a large class-wide score shift at 20 dB and progressively destroys ranking separation at 10 and 5 dB. Fixed-threshold recall can increase during this failure because bonafide and spoof scores both move toward the spoof class; EER, ROC-AUC, balanced accuracy, score distributions, and transitions expose that behavior.

These are development findings. A01–A06 were present during training, DEV selected the Phase 3 checkpoint and threshold, and robustness to unseen attacks has not been measured. The official EVAL split remains untouched. Post-processing corruption is not an adversarial attack, so these results must not be described as adversarial robustness. Gaussian noise is synthetic and stationary; the experiment does not cover real acoustic backgrounds, reverberation, device response, packet loss, or all codec implementations.

## Proposed Phase 5 mitigation

The next controlled experiment should keep the architecture and evaluation protocol fixed and compare clean-only training with mixed-condition augmentation focused on deterministic additive-noise exposure, while retaining clean examples in every training mixture. Selection should jointly track clean and corrupted DEV metrics to expose calibration and clean-performance tradeoffs. The final configuration should be frozen before the first EVAL access. No mitigation training was started in Phase 4.

Phase 4 stops here.
