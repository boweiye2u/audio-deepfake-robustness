from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audio_deepfake.metrics import binary_metrics, equal_error_rate

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"
OUTPUTS = Path.home() / "audio" / "outputs" / "experiments"
SEEDS = [20261010, 20261011, 20261012]
CONDITIONS = ["clean", "noise_20db", "noise_10db", "noise_5db"]
ATTACK_CONDITIONS = ["clean", "noise_10db", "noise_5db"]
BASELINE_THRESHOLD = 0.693742573261261
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261013


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def by_condition(rows: list[dict[str, str]], condition: str) -> list[dict[str, str]]:
    return [row for row in rows if row["condition"] == condition]


def arrays(rows: list[dict[str, str]]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    return (
        [row["utterance_id"] for row in rows],
        np.array([int(row["label"]) for row in rows]),
        np.array([float(row["score"]) for row in rows]),
        np.array([row["attack_id"] for row in rows]),
    )


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def gain_retained(baseline: float, reference: float, candidate: float) -> float:
    denominator = baseline - reference
    if denominator <= 0:
        raise ValueError("reference must improve on baseline")
    return float((baseline - candidate) / denominator)


def penalty_recovered(baseline: float, reference: float, candidate: float) -> float:
    denominator = reference - baseline
    if denominator <= 0:
        raise ValueError("reference must have a penalty relative to baseline")
    return float((reference - candidate) / denominator)


def bootstrap_eer_differences(
    labels: np.ndarray,
    left_scores: list[np.ndarray],
    right_scores: list[np.ndarray],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    differences = np.empty(replicates)
    for replicate in range(replicates):
        indices = np.concatenate(
            [rng.choice(negative, len(negative), replace=True), rng.choice(positive, len(positive), replace=True)]
        )
        sampled_labels = labels[indices]
        left = np.mean([equal_error_rate(sampled_labels, scores[indices])[0] for scores in left_scores])
        right = np.mean([equal_error_rate(sampled_labels, scores[indices])[0] for scores in right_scores])
        differences[replicate] = left - right
    observed = float(
        np.mean([equal_error_rate(labels, scores)[0] for scores in left_scores])
        - np.mean([equal_error_rate(labels, scores)[0] for scores in right_scores])
    )
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "difference": observed,
        "ci_95": [float(low), float(high)],
        "replicates": replicates,
        "seed": seed,
        "unit": "paired class-stratified DEV utterance",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    baseline_rows = load_rows(OUTPUTS / "wavlm-base-plus-robustness-v1" / "predictions.csv")
    systems = {
        "p20": {
            "probability": 0.20,
            "rows": [load_rows(OUTPUTS / "wavlm-noise-augmented-p20-v1" / f"seed-{seed}" / "predictions.csv") for seed in SEEDS],
            "summaries": [json.loads((OUTPUTS / "wavlm-noise-augmented-p20-v1" / f"seed-{seed}" / "summary.json").read_text()) for seed in SEEDS],
        },
        "p35": {
            "probability": 0.35,
            "rows": [load_rows(OUTPUTS / "wavlm-noise-augmented-p35-v1" / "seed-20261010" / "predictions.csv")],
            "summaries": [json.loads((OUTPUTS / "wavlm-noise-augmented-p35-v1" / "seed-20261010" / "summary.json").read_text())],
        },
        "p50": {
            "probability": 0.50,
            "rows": [load_rows(OUTPUTS / "wavlm-noise-augmented-v1" / f"seed-{seed}" / "predictions.csv") for seed in SEEDS],
            "summaries": [json.loads((OUTPUTS / "wavlm-noise-augmented-v1" / f"seed-{seed}" / "summary.json").read_text()) for seed in SEEDS],
        },
    }
    baseline_condition: dict[str, dict[str, object]] = {}
    system_condition: dict[str, dict[str, dict[str, object]]] = {key: {} for key in systems}
    per_seed: list[dict[str, object]] = []
    for condition in CONDITIONS:
        base = by_condition(baseline_rows, condition)
        ids, labels, scores, _ = arrays(base)
        baseline_condition[condition] = binary_metrics(labels, scores, BASELINE_THRESHOLD)
        for key, system in systems.items():
            metrics = []
            for index, (rows, summary) in enumerate(zip(system["rows"], system["summaries"], strict=True)):
                selected = by_condition(rows, condition)
                other_ids, other_labels, other_scores, _ = arrays(selected)
                if ids != other_ids or not np.array_equal(labels, other_labels):
                    raise ValueError(f"unaligned predictions for {key} {condition}")
                result = binary_metrics(labels, other_scores, summary["threshold"])
                metrics.append(result)
                per_seed.append({
                    "system": key,
                    "augmentation_probability": system["probability"],
                    "seed": summary["seed"],
                    "condition": condition,
                    "selected_epoch": summary["selected_epoch"],
                    **{name: result[name] for name in ["eer", "balanced_accuracy", "recall", "f1"]},
                })
            summary_metrics = {}
            for metric in ["eer", "balanced_accuracy", "recall", "f1"]:
                summary_metrics[f"mean_{metric}"], summary_metrics[f"std_{metric}"] = mean_std(
                    [float(result[metric]) for result in metrics]
                )
            system_condition[key][condition] = summary_metrics

    sweep: list[dict[str, object]] = []
    base_noise = float(np.mean([baseline_condition[x]["eer"] for x in CONDITIONS[1:]]))
    base_four = float(np.mean([baseline_condition[x]["eer"] for x in CONDITIONS]))
    base_a06 = attack_metric(baseline_rows, "clean", BASELINE_THRESHOLD, "A06")
    sweep.append({
        "system": "p0", "augmentation_probability": 0.0, "seed_count": 1,
        "clean_eer": baseline_condition["clean"]["eer"], "mean_noise_eer": base_noise,
        "mean_four_condition_eer": base_four,
        "clean_balanced_accuracy": baseline_condition["clean"]["balanced_accuracy"],
        "clean_f1": baseline_condition["clean"]["f1"],
        "clean_a06_eer": base_a06["eer"], "clean_a06_recall": base_a06["recall"],
    })
    for key, system in systems.items():
        clean = system_condition[key]["clean"]
        noise = float(np.mean([system_condition[key][x]["mean_eer"] for x in CONDITIONS[1:]]))
        four = float(np.mean([system_condition[key][x]["mean_eer"] for x in CONDITIONS]))
        a06_values = [
            attack_metric(rows, "clean", summary["threshold"], "A06")
            for rows, summary in zip(system["rows"], system["summaries"], strict=True)
        ]
        sweep.append({
            "system": key, "augmentation_probability": system["probability"],
            "seed_count": len(system["rows"]), "clean_eer": clean["mean_eer"],
            "mean_noise_eer": noise, "mean_four_condition_eer": four,
            "clean_balanced_accuracy": clean["mean_balanced_accuracy"], "clean_f1": clean["mean_f1"],
            "clean_a06_eer": float(np.mean([x["eer"] for x in a06_values])),
            "clean_a06_recall": float(np.mean([x["recall"] for x in a06_values])),
        })
    by_key = {row["system"]: row for row in sweep}
    for row in sweep:
        for metric in ["clean_eer", "mean_noise_eer", "mean_four_condition_eer", "clean_a06_eer"]:
            row[f"delta_vs_p0_{metric}"] = float(row[metric] - by_key["p0"][metric])
            row[f"delta_vs_p50_{metric}"] = float(row[metric] - by_key["p50"][metric])
    decision = {
        "development_seed": 20261010,
        "selected_probability": 0.20,
        "reason": "Both lower probabilities exceeded the approximate 75% target; p=0.20 recovered more clean and A06 performance.",
        "confirmed_p20_noise_gain_retained": gain_retained(
            by_key["p0"]["mean_noise_eer"], by_key["p50"]["mean_noise_eer"], by_key["p20"]["mean_noise_eer"]
        ),
        "confirmed_p20_clean_penalty_recovered": penalty_recovered(
            by_key["p0"]["clean_eer"], by_key["p50"]["clean_eer"], by_key["p20"]["clean_eer"]
        ),
        "confirmed_p20_a06_penalty_recovered": penalty_recovered(
            by_key["p0"]["clean_a06_eer"], by_key["p50"]["clean_a06_eer"], by_key["p20"]["clean_a06_eer"]
        ),
    }

    attack_rows: list[dict[str, object]] = []
    for condition in ATTACK_CONDITIONS:
        for attack in [f"A{i:02d}" for i in range(1, 7)]:
            base_metric = attack_metric(baseline_rows, condition, BASELINE_THRESHOLD, attack)
            for key in ["p20", "p35", "p50"]:
                values = [
                    attack_metric(rows, condition, summary["threshold"], attack)
                    for rows, summary in zip(systems[key]["rows"], systems[key]["summaries"], strict=True)
                ]
                attack_rows.append({
                    "condition": condition, "attack_id": attack, "system": key,
                    "seed_count": len(values), "baseline_eer": base_metric["eer"],
                    "mean_eer": float(np.mean([x["eer"] for x in values])),
                    "std_eer": float(np.std([x["eer"] for x in values], ddof=1)) if len(values) > 1 else float("nan"),
                    "baseline_recall": base_metric["recall"],
                    "mean_recall": float(np.mean([x["recall"] for x in values])),
                    "baseline_false_negatives": base_metric["false_negatives"],
                    "mean_false_negatives": float(np.mean([x["false_negatives"] for x in values])),
                })

    score_rows: list[dict[str, object]] = []
    for condition in ["clean", "noise_10db", "noise_5db"]:
        base = by_condition(baseline_rows, condition)
        ids, labels, base_scores, _ = arrays(base)
        for label_name, label_value in [("bonafide", 0), ("spoof", 1)]:
            mask = labels == label_value
            for key, score_sets in [("p0", [base_scores]), *[(key, score_arrays(systems[key]["rows"], condition, ids, labels)) for key in ["p20", "p50"]]]:
                score_rows.append({
                    "condition": condition, "label": label_name, "system": key,
                    "mean_score": float(np.mean([scores[mask].mean() for scores in score_sets])),
                    "mean_seed_median_score": float(np.mean([np.median(scores[mask]) for scores in score_sets])),
                })

    bootstrap: dict[str, object] = {"method": "paired class-stratified DEV-utterance bootstrap", "comparisons": {}}
    for offset, condition in enumerate(["clean", "noise_10db", "noise_5db"]):
        base = by_condition(baseline_rows, condition)
        ids, labels, base_scores, _ = arrays(base)
        p20_scores = score_arrays(systems["p20"]["rows"], condition, ids, labels)
        p50_scores = score_arrays(systems["p50"]["rows"], condition, ids, labels)
        bootstrap["comparisons"][condition] = {
            "p20_minus_p0": bootstrap_eer_differences(labels, p20_scores, [base_scores], BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED + offset * 2),
            "p20_minus_p50": bootstrap_eer_differences(labels, p20_scores, p50_scores, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED + offset * 2 + 1),
        }
    # A06 uses the shared bonafide set plus A06 spoof trials, making paired class-stratified resampling well-defined.
    base = by_condition(baseline_rows, "clean")
    ids, labels, base_scores, attacks = arrays(base)
    mask = np.isin(attacks, ["-", "A06"])
    p20_scores = score_arrays(systems["p20"]["rows"], "clean", ids, labels)
    p50_scores = score_arrays(systems["p50"]["rows"], "clean", ids, labels)
    bootstrap["comparisons"]["clean_a06"] = {
        "p20_minus_p0": bootstrap_eer_differences(labels[mask], [x[mask] for x in p20_scores], [base_scores[mask]], BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED + 6),
        "p20_minus_p50": bootstrap_eer_differences(labels[mask], [x[mask] for x in p20_scores], [x[mask] for x in p50_scores], BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED + 7),
    }

    write_csv(METRICS / "augmentation-probability-sweep.csv", sweep)
    write_csv(METRICS / "augmentation-probability-per-seed.csv", per_seed)
    write_csv(METRICS / "augmentation-probability-by-attack.csv", attack_rows)
    write_csv(METRICS / "augmentation-probability-score-shifts.csv", score_rows)
    (METRICS / "augmentation-probability-bootstrap.json").write_text(json.dumps(bootstrap, indent=2) + "\n")
    summary = {
        "scope": "DEV-only", "evaluation_split_accessed": False,
        "probability_sweep": sweep, "decision": decision,
        "p20_selected_epochs": {str(x["seed"]): x["selected_epoch"] for x in systems["p20"]["summaries"]},
        "p20_thresholds": [x["threshold"] for x in systems["p20"]["summaries"]],
        "p20_checkpoint_sha256": {str(x["seed"]): x["checkpoint_sha256"] for x in systems["p20"]["summaries"]},
        "runtime_seconds": {
            key: {str(x["seed"]): x["feature_extraction_seconds"] + x["training_seconds"] for x in system["summaries"]}
            for key, system in systems.items() if key in {"p20", "p35"}
        },
        "bootstrap": bootstrap,
    }
    (METRICS / "augmentation-probability-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_plots(sweep, score_rows)
    print(json.dumps(summary, indent=2))


def score_arrays(all_rows, condition, expected_ids, expected_labels):
    values = []
    for rows in all_rows:
        ids, labels, scores, _ = arrays(by_condition(rows, condition))
        if ids != expected_ids or not np.array_equal(labels, expected_labels):
            raise ValueError("predictions are not aligned")
        values.append(scores)
    return values


def attack_metric(rows, condition, threshold, attack):
    selected = by_condition(rows, condition)
    _, labels, scores, attacks = arrays(selected)
    mask = np.isin(attacks, ["-", attack])
    result = binary_metrics(labels[mask], scores[mask], threshold)
    result["false_negatives"] = int(np.sum((attacks == attack) & (scores < threshold)))
    return result


def make_plots(sweep, score_rows):
    data = pd.DataFrame(sweep).sort_values("augmentation_probability")
    fig, axis = plt.subplots(figsize=(6.4, 4.6))
    axis.plot(data["clean_eer"], data["mean_noise_eer"], color="#9aa0a6", linewidth=1, zorder=1)
    for row in data.to_dict("records"):
        label = f"p={row['augmentation_probability']:.2f}" + (" (1 seed)" if row["seed_count"] == 1 and row["system"] == "p35" else "")
        axis.scatter(row["clean_eer"], row["mean_noise_eer"], s=65, zorder=2)
        axis.annotate(label, (row["clean_eer"], row["mean_noise_eer"]), xytext=(5, 5), textcoords="offset points")
    axis.set_xlabel("Clean DEV EER (lower is better)")
    axis.set_ylabel("Mean noise DEV EER (lower is better)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "augmentation-probability-pareto.png", dpi=180)
    plt.close(fig)

    frame = pd.DataFrame(score_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, label in zip(axes, ["bonafide", "spoof"], strict=True):
        subset = frame[frame["label"] == label]
        for system, name in [("p0", "p=0"), ("p20", "p=0.20"), ("p50", "p=0.50")]:
            values = subset[subset["system"] == system]
            axis.plot(values["condition"], values["mean_score"], marker="o", label=name)
        axis.set_title(label)
        axis.set_ylabel("Mean spoof probability")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "augmentation-probability-score-shifts.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
