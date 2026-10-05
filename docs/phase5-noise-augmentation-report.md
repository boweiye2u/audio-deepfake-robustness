# Phase 5 — controlled additive-noise robustness training

> **Scope:** Training uses the official ASVspoof 2019 LA TRAIN split, while checkpoint and threshold selection use DEV only. The same 24,844 DEV trials and Phase 4 corruptions are used for comparison. The official EVAL split was not loaded.

## Experimental question

Can controlled waveform-level additive-noise augmentation reduce the frozen detector's noise sensitivity without sacrificing too much clean performance?

## Frozen design

Model A is the Phase 3 clean-trained baseline. Model B retains exactly the same architecture and optimizer:

```text
WavLM Base Plus, revision 4c66d4806a428f2e922ccfa1a962776e232d487b (frozen)
→ masked mean pooling
→ LayerNorm
→ dropout 0.1
→ linear binary head
```

The only training-data change is deterministic additive Gaussian noise. For each TRAIN utterance and seed, a SHA-256 decision leaves it clean with probability 0.5 or chooses 20, 10, or 5 dB uniformly. Noise is added to the waveform before the unchanged crop, normalization, and WavLM encoder. No DEV waveform is augmented during clean evaluation, and no MP3, resampling, external-noise data, encoder unfreezing, or architecture change is introduced.

Three seeds were declared before training: `20261010`, `20261011`, and `20261012`. Their observed augmentation rates were 0.5036, 0.5001, and 0.5010; every SNR stratum contained approximately 4,200 trials. Maximum measured SNR error was `5.08e-8` dB.

## Selection and threshold rules

Every epoch is evaluated on clean, 20 dB, 10 dB, and 5 dB DEV using the fixed corruption caches. Checkpoints are ordered by:

1. lowest mean EER across those four conditions;
2. lowest clean EER;
3. earliest epoch.

This rule was frozen before training. Selected epochs were 29, 29, and 30. Each selected model then receives one operating threshold selected on clean DEV by maximum balanced accuracy, closest to 0.5 on ties. The thresholds are 0.653612, 0.617781, and 0.643408. No corruption-specific operating threshold is used.

## Main comparison

Robust values are mean ± sample standard deviation across the three seeds. Deltas are robust mean minus the frozen baseline. Threshold-dependent metrics use each model's clean-selected threshold.

| Condition | Baseline EER | Robust EER | Δ EER | Baseline ROC | Robust ROC | Baseline BA | Robust BA | Baseline recall | Robust recall | Baseline F1 | Robust F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean | .0522 | .0688 ± .0013 | +.0166 | .9898 | .9816 | .9500 | .9338 ± .0014 | .9315 | .9023 ± .0099 | .9627 | .9466 ± .0049 |
| MP3 128k | .0496 | .0696 ± .0006 | +.0200 | .9906 | .9812 | .9526 | .9357 ± .0006 | .9347 | .9064 ± .0108 | .9645 | .9489 ± .0054 |
| MP3 64k | .0492 | .0704 ± .0006 | +.0211 | .9908 | .9810 | .9536 | .9358 ± .0014 | .9339 | .9096 ± .0099 | .9643 | .9505 ± .0047 |
| MP3 32k | .0440 | .0681 ± .0007 | +.0240 | .9921 | .9820 | .9509 | .9268 ± .0048 | .9678 | .9424 ± .0078 | .9799 | .9653 ± .0032 |
| 8 kHz | .0432 | .0653 ± .0003 | +.0221 | .9924 | .9834 | .9502 | .9373 ± .0018 | .9729 | .9080 ± .0098 | .9821 | .9499 ± .0050 |
| Noise 20 dB | .1035 | .0619 ± .0021 | -.0416 | .9591 | .9825 | .5130 | .9336 ± .0054 | .9997 | .8990 ± .0177 | .9471 | .9450 ± .0094 |
| Noise 10 dB | .2260 | .0808 ± .0018 | -.1452 | .8206 | .9742 | .4964 | .9065 ± .0075 | .9841 | .8558 ± .0254 | .9384 | .9197 ± .0141 |
| Noise 5 dB | .3902 | .1267 ± .0023 | -.2635 | .6603 | .9480 | .5154 | .8690 ± .0049 | .8848 | .8252 ± .0307 | .8927 | .8991 ± .0171 |

PR-AUC changes are -0.00091 clean, +0.00299 at 20 dB, +0.01881 at 10 dB, and +0.04173 at 5 dB. Full precision, PR-AUC, and standard-deviation fields are in `results/metrics/noise-augmentation-main.csv`.

The summary mean EER across clean and the three noise conditions falls from **0.1930 to 0.0846**. Across all eight conditions it falls from **0.1197 to 0.0765**. These averages supplement rather than replace the condition results.

## Clean-performance tradeoff

Noise augmentation increases clean EER by **0.01665 absolute**, or **31.9% relative**. Clean ROC-AUC decreases by 0.00818, PR-AUC by 0.00091, balanced accuracy by 0.01624, recall by 0.02921, and F1 by 0.01610. The model therefore gains substantial noise robustness at a measurable clean and codec/resampling cost.

## Paired bootstrap

The predefined comparison uses 2,000 class-stratified paired utterance resamples. Within every resample, each robust seed is scored at its own clean threshold, the three metrics are averaged, and the baseline is scored at its frozen threshold. Intervals are robust mean minus baseline.

| Condition | Δ EER 95% CI | Δ ROC-AUC 95% CI | Δ balanced accuracy 95% CI | Δ F1 95% CI |
|---|---|---|---|---|
| Clean | [.0133, .0201] | [-.0092, -.0072] | [-.0195, -.0130] | [-.0175, -.0148] |
| MP3 128k | [.0165, .0241] | [-.0104, -.0084] | [-.0205, -.0132] | [-.0171, -.0142] |
| MP3 64k | [.0174, .0246] | [-.0107, -.0087] | [-.0213, -.0143] | [-.0152, -.0125] |
| MP3 32k | [.0198, .0271] | [-.0111, -.0091] | [-.0289, -.0190] | [-.0158, -.0132] |
| 8 kHz | [.0186, .0264] | [-.0100, -.0080] | [-.0176, -.0080] | [-.0341, -.0305] |
| Noise 20 dB | [-.0460, -.0366] | [.0211, .0256] | [.4159, .4253] | [-.0043, -.0002] |
| Noise 10 dB | [-.1518, -.1381] | [.1482, .1590] | [.4053, .4147] | [-.0212, -.0161] |
| Noise 5 dB | [-.2722, -.2546] | [.2795, .2966] | [.3444, .3630] | [.0032, .0097] |

The baseline's high noisy-condition recall and F1 partly result from predicting nearly everything as spoof. Balanced accuracy, EER, ROC-AUC, class score shifts, and paired transitions show that the robust models recover discrimination and bonafide specificity even when recall or F1 decreases.

## Attack-specific results

On clean DEV, mean A06 EER worsens from **0.1244 to 0.1875**, recall falls from 0.5953 to 0.4244, and false negatives rise from 1,504 to an average 2,139. A01–A05 remain comparatively easy, with robust clean EER between 0.0026 and 0.0154.

Under noise, threshold-free EER improves for every attack family. At 5 dB:

| Attack | Baseline EER | Robust EER mean | Baseline recall | Robust recall mean | Baseline FN | Robust FN mean |
|---|---:|---:|---:|---:|---:|---:|
| A01 | .3202 | .0211 | .9766 | .9961 | 87 | 14.3 |
| A02 | .3956 | .1743 | .8283 | .6809 | 638 | 1185.7 |
| A03 | .3120 | .0777 | .9279 | .9289 | 268 | 264.3 |
| A04 | .3345 | .0445 | .9639 | .9743 | 134 | 95.3 |
| A05 | .5871 | .1213 | .7158 | .8267 | 1056 | 644.0 |
| A06 | .4058 | .2182 | .8964 | .5445 | 385 | 1692.7 |

A05 is no longer the hardest attack by EER at 5 dB; A06 becomes hardest. This is another tradeoff: augmentation improves ranking separation for A06 under noise but its clean-selected threshold yields substantially lower A06 recall. The detailed per-seed attack table is retained rather than treating mean fractional false-negative counts as literal observations.

## Score shift and calibration behavior

Phase 4 noise drove bonafide mean spoof scores from 0.1346 clean to 0.9595, 0.9923, and 0.8765 at 20, 10, and 5 dB. Noise augmentation changes those means to **0.1576, 0.1666, and 0.2545**, close to its clean bonafide mean of 0.1584. Spoof means remain separated at 0.8918, 0.8601, and 0.8437. Thus augmentation largely removes the catastrophic class-wide shift toward spoof while preserving useful separation. No extra calibration metric was added because the fixed-threshold score and class diagnostics already answer the declared question.

## Paired correctness transitions

Mean transition counts across seeds are descriptive; exact integer counts per seed and attack are in `noise-augmentation-transitions.csv`.

| Condition | Both correct | Baseline correct, robust wrong | Baseline wrong, robust correct | Both wrong |
|---|---:|---:|---:|---:|
| Clean | 22387.7 | 848.3 | 188.7 | 1419.3 |
| Noise 20 dB | 20103.3 | 2252.7 | 2408.0 | 80.0 |
| Noise 10 dB | 18946.0 | 3018.0 | 2574.0 | 306.0 |
| Noise 5 dB | 17497.3 | 2602.7 | 3228.0 | 1516.0 |

At 10 dB, augmentation fixes many baseline errors but creates slightly more new errors at the operating threshold; balanced accuracy improves because the class distribution and error types differ. At 5 dB it fixes more errors than it creates.

## Runtime, storage, and reproducibility

Feature extraction took 35.73, 36.05, and 36.44 seconds per seed. Head training took 23.99, 22.55, and 22.51 seconds. Total measured feature extraction plus training was **177.3 seconds** on one NVIDIA L40S; paired analysis and bootstrap took about 17 minutes on CPU. Peak PyTorch GPU allocation was 2,532,723,200 bytes (about 2.36 GiB).

External storage:

- augmented TRAIN feature caches: 227 MiB;
- checkpoints: 3.3 MiB;
- predictions, histories, and analysis: 39 MiB.

Artifacts:

- Frozen config: `configs/noise_augmentation_5.yaml`
- External caches: `~/audio/cache/features/wavlm-noise-augmented-v1/`
- Checkpoints: `~/audio/models/audio-deepfake/wavlm-noise-augmented-v1/`
- External run: `~/audio/outputs/experiments/wavlm-noise-augmented-v1/`
- Repository tables: `results/metrics/noise-augmentation-*`
- Figures: `results/figures/noise-augmentation-*`

Commands:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=$HOME/audio/cache/huggingface \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  $HOME/audio/envs/audio-deepfake/bin/python scripts/run_noise_augmentation_5.py --phase preflight

# Repeated for seeds 20261010, 20261011, and 20261012:
$HOME/audio/envs/audio-deepfake/bin/python scripts/run_noise_augmentation_5.py --phase train --seed <seed>

$HOME/audio/envs/audio-deepfake/bin/python scripts/run_noise_augmentation_5.py --phase analyze
$HOME/audio/envs/audio-deepfake/bin/python scripts/finalize_noise_augmentation_5.py
python -m pytest -q
ruff check .
```

## Interpretation and limitations

Controlled Gaussian-noise augmentation substantially improves noisy-speech ranking and prevents the baseline's collapse toward spoof predictions. It also degrades clean, codec, and resampling performance, and it worsens the existing clean A06 weakness. The result supports augmentation as a useful but incomplete mitigation under this exact configuration; it does not establish a universally superior detector.

The noise is synthetic rather than real telephony, channel, environmental, or reverberant noise. A01–A06 occur in TRAIN. DEV selected checkpoints and thresholds, so these are development findings. The official EVAL split remains untouched. Phase 4 MP3/resampling improvements may not generalize. Additive-noise robustness is not adversarial robustness.

## Recommended next step

Before EVAL, use one bounded DEV-only follow-up to test whether a less aggressive augmentation mixture preserves more clean and A06 performance—for example reducing augmentation probability while keeping the same SNR set and all other settings frozen. This would be a new versioned experiment, not a reinterpretation of Phase 5. No such experiment was launched here.

Phase 5 stops here.
