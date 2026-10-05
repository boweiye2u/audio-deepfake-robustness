from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

main = pd.read_csv(METRICS / "noise-augmentation-main.csv")
conditions = main["condition"].tolist()
x = np.arange(len(conditions))
width = 0.38
for metric, label, filename in [
    ("eer", "EER", "noise-augmentation-eer.png"),
    ("balanced_accuracy", "Balanced accuracy", "noise-augmentation-balanced-accuracy.png"),
]:
    fig, axis = plt.subplots(figsize=(9, 4))
    axis.bar(x - width / 2, main[f"baseline_{metric}"], width, label="Clean-trained baseline")
    axis.bar(x + width / 2, main[f"robust_mean_{metric}"], width, label="Noise augmented")
    axis.errorbar(
        x + width / 2, main[f"robust_mean_{metric}"],
        yerr=main[f"robust_std_{metric}"], fmt="none", color="black", capsize=3,
    )
    axis.set_xticks(x, conditions, rotation=35, ha="right")
    axis.set_ylabel(label)
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=160)
    plt.close(fig)

shifts = pd.read_csv(METRICS / "noise-augmentation-score-shifts.csv")
noise = shifts[shifts["condition"].isin(["clean", "noise_20db", "noise_10db", "noise_5db"])]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for axis, label in zip(axes, ["bonafide", "spoof"], strict=True):
    selected = noise[noise["label"] == label]
    pos = np.arange(len(selected))
    axis.plot(pos, selected["baseline_mean"], marker="o", label="Baseline")
    axis.plot(pos, selected["robust_mean"], marker="o", label="Noise augmented")
    axis.set_xticks(pos, selected["condition"], rotation=30, ha="right")
    axis.set_title(label)
    axis.set_ylabel("Mean spoof probability")
    axis.legend()
fig.tight_layout()
fig.savefig(FIGURES / "noise-augmentation-score-shifts.png", dpi=160)
plt.close(fig)

attacks = pd.read_csv(METRICS / "noise-augmentation-by-attack.csv")
selected = attacks[attacks["condition"].isin(["clean", "noise_20db", "noise_10db", "noise_5db"])]
robust = selected.groupby(["condition", "attack_id"], sort=False)["robust_eer"].mean().unstack()
fig, axis = plt.subplots(figsize=(7, 4))
image = axis.imshow(robust.to_numpy(), vmin=0, vmax=max(0.25, robust.to_numpy().max()), aspect="auto", cmap="magma")
axis.set_xticks(range(len(robust.columns)), robust.columns)
axis.set_yticks(range(len(robust.index)), robust.index)
axis.set_title("Noise-augmented mean EER by attack")
fig.colorbar(image, ax=axis)
fig.tight_layout()
fig.savefig(FIGURES / "noise-augmentation-attack-eer.png", dpi=160)
plt.close(fig)
