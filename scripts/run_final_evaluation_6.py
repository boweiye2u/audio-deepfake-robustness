from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from audio_deepfake.asvspoof import read_manifest
from audio_deepfake.baseline import (
    AudioRecord,
    label_to_index,
    load_head_checkpoint,
    write_records_csv,
)
from audio_deepfake.config import ArtifactRoots, load_config
from audio_deepfake.corruptions import CORE_CORRUPTIONS, CorruptedAudioDataset
from audio_deepfake.metrics import binary_metrics, equal_error_rate
from audio_deepfake.utils import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "final_evaluation.yaml"
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"
SPEC_BY_NAME = {spec.name: spec for spec in CORE_CORRUPTIONS}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def git_clean() -> bool:
    return not subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_eval_records(config: dict[str, Any], roots: ArtifactRoots) -> list[AudioRecord]:
    rows = read_manifest(Path(config["dataset"]["manifest"]), purpose="final_evaluation")
    records = []
    for row in rows:
        if row["split"] != "eval":
            raise PermissionError("final evaluation accepts only official EVAL records")
        audio_path = roots.data / row["audio_path"]
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        records.append(
            AudioRecord(
                utterance_id=row["utterance_id"], audio_path=audio_path, split="eval",
                label=label_to_index(row["label"]), speaker_id=row["speaker_id"],
                attack_id=row["attack_id"], duration_seconds=float(row["duration_seconds"]),
                sample_rate=int(row["sample_rate"]),
            )
        )
    return records


def validate_protocol(config: dict[str, Any], roots: ArtifactRoots, records: list[AudioRecord]) -> dict[str, Any]:
    if config["protocol"]["purpose"] != "final_evaluation":
        raise PermissionError("explicit final_evaluation purpose is required")
    if config["protocol"]["model_selection_allowed"] or config["protocol"]["retraining_allowed"]:
        raise PermissionError("model selection and retraining must remain disabled")
    manifest = Path(config["dataset"]["manifest"])
    if sha256(manifest) != config["dataset"]["manifest_sha256"]:
        raise ValueError("frozen EVAL manifest hash mismatch")
    labels = np.array([record.label for record in records])
    attacks = {record.attack_id for record in records if record.attack_id != "-"}
    if len(records) != config["dataset"]["expected_trials"]:
        raise ValueError("unexpected EVAL trial count")
    if int(np.sum(labels == 0)) != config["dataset"]["expected_bonafide"]:
        raise ValueError("unexpected EVAL bonafide count")
    if int(np.sum(labels == 1)) != config["dataset"]["expected_spoof"]:
        raise ValueError("unexpected EVAL spoof count")
    if attacks != set(config["dataset"]["expected_attack_ids"]):
        raise ValueError("unexpected EVAL attacks")
    if len({record.utterance_id for record in records}) != len(records):
        raise ValueError("duplicate EVAL utterance IDs")
    for name in ["train", "dev"]:
        path = roots.data / f"asvspoof2019/manifests/development/{name}.csv"
        other = {row["utterance_id"] for row in read_manifest(path, purpose="development")}
        if other & {record.utterance_id for record in records}:
            raise ValueError(f"EVAL overlaps {name}")
    baseline = config["models"]["baseline"]
    checkpoints = [(baseline["checkpoint"], baseline["checkpoint_sha256"], baseline["selected_epoch"])]
    checkpoints += [
        (item["path"], item["sha256"], item["selected_epoch"])
        for item in config["models"]["robust"]["checkpoints"].values()
    ]
    verified = []
    for path_value, expected_hash, expected_epoch in checkpoints:
        path = Path(path_value)
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"checkpoint hash mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["epoch"] != expected_epoch:
            raise ValueError(f"checkpoint epoch mismatch: {path}")
        verified.append({"path": str(path), "sha256": actual_hash, "epoch": payload["epoch"]})
    configured = config["corruptions"]["conditions"]
    expected = [
        {key: value for key, value in spec.__dict__.items() if value is not None}
        for spec in CORE_CORRUPTIONS
    ]
    if configured != expected:
        raise ValueError("corruption definitions differ from frozen implementation")
    if sha256(ROOT / "src/audio_deepfake/corruptions.py") != config["corruptions"]["implementation_sha256"]:
        raise ValueError("corruption implementation hash mismatch")
    if sha256(ROOT / "src/audio_deepfake/metrics.py") != config["metrics"]["implementation_sha256"]:
        raise ValueError("metric implementation hash mismatch")
    return {
        "trials": len(records), "bonafide": int(np.sum(labels == 0)),
        "spoof": int(np.sum(labels == 1)), "attack_ids": sorted(attacks),
        "manifest_sha256": sha256(manifest), "checkpoints": verified,
        "git_commit": git_commit(), "git_clean": git_clean(),
    }


def preflight(config: dict[str, Any], roots: ArtifactRoots) -> None:
    if not git_clean():
        raise RuntimeError("preflight requires a clean committed working tree")
    records = load_eval_records(config, roots)
    validation = validate_protocol(config, roots, records)
    output_dir = Path(config["inference"]["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite final run: {output_dir}")
    output_dir.mkdir(parents=True)
    validation["config_sha256"] = sha256(CONFIG_PATH)
    validation["conditions"] = [item["name"] for item in config["corruptions"]["conditions"]]
    validation["thresholds"] = {
        "baseline": config["models"]["baseline"]["threshold"],
        **{
            f"robust_{seed}": config["models"]["robust"]["checkpoints"][str(seed)]["threshold"]
            for seed in config["models"]["robust"]["seeds"]
        },
    }
    validation["evaluation_results_observed"] = False
    write_json_atomic(output_dir / "pre-run-manifest.json", validation)
    print(json.dumps(validation, indent=2))


def load_encoder(config: dict[str, Any], roots: ArtifactRoots, device: torch.device):
    from transformers import WavLMModel

    encoder = WavLMModel.from_pretrained(
        config["encoder"]["model"], revision=config["encoder"]["revision"],
        cache_dir=roots.cache / "huggingface", local_files_only=True,
    )
    return encoder.eval().requires_grad_(False).to(device)


def load_heads(config: dict[str, Any], device: torch.device):
    models = {}
    baseline = config["models"]["baseline"]
    models["baseline"], _ = load_head_checkpoint(Path(baseline["checkpoint"]), 768, 0.1, device)
    for seed in config["models"]["robust"]["seeds"]:
        item = config["models"]["robust"]["checkpoints"][str(seed)]
        models[f"robust_{seed}"], _ = load_head_checkpoint(Path(item["path"]), 768, 0.1, device)
    for model in models.values():
        model.eval().requires_grad_(False)
    return models


def predict(head, embeddings: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    values = []
    loader = DataLoader(TensorDataset(embeddings), batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for (batch,) in loader:
            values.append(torch.sigmoid(head(batch.to(device, non_blocking=True))).cpu())
    return torch.cat(values).numpy()


def signature(config: dict[str, Any], records: list[AudioRecord], condition: str) -> dict[str, Any]:
    return {
        "manifest_sha256": config["dataset"]["manifest_sha256"],
        "utterance_ids_sha256": hashlib.sha256(
            "\n".join(record.utterance_id for record in records).encode()
        ).hexdigest(),
        "encoder": config["encoder"]["model"], "revision": config["encoder"]["revision"],
        "condition": next(x for x in config["corruptions"]["conditions"] if x["name"] == condition),
        "corruption_seed": config["corruptions"]["deterministic_seed"],
    }


def run_condition(config: dict[str, Any], roots: ArtifactRoots, condition: str, device: torch.device) -> None:
    if condition not in SPEC_BY_NAME:
        raise ValueError(f"unknown frozen condition: {condition}")
    records = load_eval_records(config, roots)
    validate_protocol(config, roots, records)
    output_dir = Path(config["inference"]["output_dir"])
    if not (output_dir / "pre-run-manifest.json").is_file():
        raise FileNotFoundError("preflight manifest is required")
    condition_dir = output_dir / "conditions"
    condition_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = condition_dir / f"{condition}.npz"
    metadata_path = condition_dir / f"{condition}.json"
    if prediction_path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite condition: {condition}")
    expected_signature = signature(config, records, condition)
    cache_path = Path(config["inference"]["feature_cache"]) / f"{condition}.pt"
    encoder = load_encoder(config, roots, device)
    started = time.perf_counter()
    peak = 0
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached["signature"] != expected_signature:
            raise ValueError(f"feature cache signature mismatch: {condition}")
        embeddings = cached["embeddings"]
        extraction_seconds = 0.0
    else:
        dataset = CorruptedAudioDataset(
            records, SPEC_BY_NAME[condition], config["corruptions"]["deterministic_seed"],
            config["encoder"]["sample_rate"], config["encoder"]["max_duration_seconds"],
            config["encoder"]["crop_strategy"], purpose="final_evaluation",
        )
        loader = DataLoader(
            dataset, batch_size=config["inference"]["batch_size"], shuffle=False,
            num_workers=config["inference"]["num_workers"], pin_memory=True,
            persistent_workers=config["inference"]["num_workers"] > 0,
        )
        batches, indices = [], []
        torch.cuda.reset_peak_memory_stats(device)
        extraction_started = time.perf_counter()
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"EVAL {condition}"):
                waveform = batch["waveform"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    hidden = encoder(waveform, attention_mask=attention_mask).last_hidden_state
                    mask = encoder._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
                    weights = mask.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
                batches.append(pooled.float().cpu())
                indices.append(batch["index"])
        extraction_seconds = time.perf_counter() - extraction_started
        peak = torch.cuda.max_memory_allocated(device)
        if not np.array_equal(torch.cat(indices).numpy(), np.arange(len(records))):
            raise RuntimeError("EVAL ordering changed")
        embeddings = torch.cat(batches)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".pt.tmp")
        torch.save({"signature": expected_signature, "embeddings": embeddings}, temporary)
        temporary.replace(cache_path)
    heads = load_heads(config, device)
    scores = {
        name: predict(head, embeddings, config["inference"]["batch_size"], device)
        for name, head in heads.items()
    }
    if any(len(values) != len(records) or not np.all(np.isfinite(values)) for values in scores.values()):
        raise RuntimeError("invalid final prediction array")
    np.savez_compressed(prediction_path, **scores)
    metadata = {
        "condition": condition, "signature": expected_signature, "trials": len(records),
        "models": list(scores), "extraction_seconds": extraction_seconds,
        "total_seconds": time.perf_counter() - started, "peak_gpu_bytes": peak,
        "prediction_sha256": sha256(prediction_path), "git_commit": git_commit(),
    }
    write_json_atomic(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))


def comparison_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = scores >= threshold
    return {
        "eer": equal_error_rate(labels, scores)[0],
        "roc_auc": float(roc_auc_score(labels, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def paired_bootstrap(
    labels: np.ndarray, baseline: np.ndarray, robust: list[np.ndarray],
    baseline_threshold: float, robust_thresholds: list[float], replicates: int, seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    negative, positive = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    values = {metric: [] for metric in ["eer", "roc_auc", "balanced_accuracy", "f1"]}
    for _ in range(replicates):
        indices = np.concatenate([
            rng.choice(negative, len(negative), replace=True),
            rng.choice(positive, len(positive), replace=True),
        ])
        sampled_labels = labels[indices]
        base = comparison_metrics(sampled_labels, baseline[indices], baseline_threshold)
        robust_metrics = [
            comparison_metrics(sampled_labels, scores[indices], threshold)
            for scores, threshold in zip(robust, robust_thresholds, strict=True)
        ]
        for metric, samples in values.items():
            samples.append(float(np.mean([x[metric] for x in robust_metrics])) - base[metric])
    return {
        metric: {
            "difference": float(
                np.mean([comparison_metrics(labels, scores, threshold)[metric]
                         for scores, threshold in zip(robust, robust_thresholds, strict=True)])
                - comparison_metrics(labels, baseline, baseline_threshold)[metric]
            ),
            "ci_95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        }
        for metric, samples in values.items()
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_records_csv(path, rows)


def analyze(config: dict[str, Any], roots: ArtifactRoots) -> None:
    records = load_eval_records(config, roots)
    validate_protocol(config, roots, records)
    labels = np.array([record.label for record in records])
    attacks = np.array([record.attack_id for record in records])
    condition_names = [item["name"] for item in config["corruptions"]["conditions"]]
    output_dir = Path(config["inference"]["output_dir"])
    arrays = {}
    for condition in condition_names:
        path = output_dir / "conditions" / f"{condition}.npz"
        metadata = json.loads((output_dir / "conditions" / f"{condition}.json").read_text())
        if sha256(path) != metadata["prediction_sha256"] or metadata["trials"] != len(records):
            raise ValueError(f"prediction validation failed: {condition}")
        with np.load(path) as payload:
            arrays[condition] = {name: payload[name].copy() for name in payload.files}
        if set(arrays[condition]) != {"baseline", *[f"robust_{s}" for s in config["models"]["robust"]["seeds"]]}:
            raise ValueError("unexpected prediction systems")
        if any(len(x) != len(records) or not np.all(np.isfinite(x)) for x in arrays[condition].values()):
            raise ValueError("invalid predictions")
    baseline_threshold = config["models"]["baseline"]["threshold"]
    seeds = config["models"]["robust"]["seeds"]
    robust_thresholds = [config["models"]["robust"]["checkpoints"][str(seed)]["threshold"] for seed in seeds]
    main_rows, per_seed_rows, attack_rows, shift_rows, transition_rows, generalization_rows = [], [], [], [], [], []
    for condition in condition_names:
        baseline = binary_metrics(labels, arrays[condition]["baseline"], baseline_threshold)
        robust_metrics = []
        for seed, threshold in zip(seeds, robust_thresholds, strict=True):
            result = binary_metrics(labels, arrays[condition][f"robust_{seed}"], threshold)
            robust_metrics.append(result)
            per_seed_rows.append({"condition": condition, "seed": seed, **result})
        row = {"condition": condition}
        for metric in ["eer", "roc_auc", "pr_auc", "balanced_accuracy", "precision", "recall", "f1"]:
            values = [float(item[metric]) for item in robust_metrics]
            mean, std = mean_std(values)
            row[f"baseline_{metric}"] = baseline[metric]
            row[f"robust_mean_{metric}"] = mean
            row[f"robust_std_{metric}"] = std
            row[f"delta_{metric}"] = mean - float(baseline[metric])
        row["baseline_confusion_matrix"] = json.dumps(baseline["confusion_matrix"])
        row["robust_confusion_matrices"] = json.dumps([x["confusion_matrix"] for x in robust_metrics])
        main_rows.append(row)
        if condition in {"clean", "noise_20db", "noise_10db", "noise_5db"}:
            for attack in config["dataset"]["expected_attack_ids"]:
                mask = np.isin(attacks, ["-", attack])
                attack_only = attacks == attack
                base_result = binary_metrics(labels[mask], arrays[condition]["baseline"][mask], baseline_threshold)
                robust_results = [
                    binary_metrics(labels[mask], arrays[condition][f"robust_{seed}"][mask], threshold)
                    for seed, threshold in zip(seeds, robust_thresholds, strict=True)
                ]
                attack_rows.append({
                    "condition": condition, "attack_id": attack,
                    "spoof_trials": int(attack_only.sum()), "baseline_eer": base_result["eer"],
                    "robust_mean_eer": float(np.mean([x["eer"] for x in robust_results])),
                    "robust_std_eer": float(np.std([x["eer"] for x in robust_results], ddof=1)),
                    "baseline_recall": base_result["recall"],
                    "robust_mean_recall": float(np.mean([x["recall"] for x in robust_results])),
                    "baseline_false_negatives": int(np.sum(attack_only & (arrays[condition]["baseline"] < baseline_threshold))),
                    "robust_mean_false_negatives": float(np.mean([
                        np.sum(attack_only & (arrays[condition][f"robust_{seed}"] < threshold))
                        for seed, threshold in zip(seeds, robust_thresholds, strict=True)
                    ])),
                    "baseline_spoof_score_mean": float(arrays[condition]["baseline"][attack_only].mean()),
                    "robust_spoof_score_mean": float(np.mean([
                        arrays[condition][f"robust_{seed}"][attack_only].mean() for seed in seeds
                    ])),
                })
        if condition in {"clean", "noise_10db", "noise_5db"}:
            for label_name, value in [("bonafide", 0), ("spoof", 1)]:
                mask = labels == value
                shift_rows.append({
                    "condition": condition, "label": label_name,
                    "baseline_mean": float(arrays[condition]["baseline"][mask].mean()),
                    "baseline_median": float(np.median(arrays[condition]["baseline"][mask])),
                    "robust_mean": float(np.mean([
                        arrays[condition][f"robust_{seed}"][mask].mean() for seed in seeds
                    ])),
                    "robust_mean_seed_median": float(np.mean([
                        np.median(arrays[condition][f"robust_{seed}"][mask]) for seed in seeds
                    ])),
                })
            for seed, threshold in zip(seeds, robust_thresholds, strict=True):
                base_correct = (arrays[condition]["baseline"] >= baseline_threshold) == labels
                robust_correct = (arrays[condition][f"robust_{seed}"] >= threshold) == labels
                groups = [("all", np.ones(len(labels), dtype=bool)), ("bonafide", labels == 0), ("spoof", labels == 1)]
                groups += [(attack, attacks == attack) for attack in config["dataset"]["expected_attack_ids"]]
                for group, mask in groups:
                    transition_rows.append({
                        "condition": condition, "seed": seed, "group": group,
                        "both_correct": int(np.sum(mask & base_correct & robust_correct)),
                        "baseline_correct_robust_wrong": int(np.sum(mask & base_correct & ~robust_correct)),
                        "baseline_wrong_robust_correct": int(np.sum(mask & ~base_correct & robust_correct)),
                        "both_wrong": int(np.sum(mask & ~base_correct & ~robust_correct)),
                    })
    # DEV metrics are frozen prior artifacts; no selection is performed here.
    dev_baseline = {row["condition"]: row for row in json.loads((METRICS_DIR / "robustness-summary.json").read_text())["main"]}
    p20_summaries = [
        json.loads((roots.outputs / f"experiments/wavlm-noise-augmented-p20-v1/seed-{seed}/summary.json").read_text())
        for seed in seeds
    ]
    eval_lookup = {row["condition"]: row for row in main_rows}
    for condition in condition_names:
        baseline_dev = float(dev_baseline[condition]["eer"])
        robust_dev = float(np.mean([summary["metrics"][condition]["eer"] for summary in p20_summaries]))
        generalization_rows += [
            {"model": "baseline", "condition": condition, "dev_eer": baseline_dev,
             "eval_eer": eval_lookup[condition]["baseline_eer"],
             "eval_minus_dev_eer": eval_lookup[condition]["baseline_eer"] - baseline_dev,
             "dev_delta_from_clean": baseline_dev - float(dev_baseline["clean"]["eer"]),
             "eval_delta_from_clean": eval_lookup[condition]["baseline_eer"] - eval_lookup["clean"]["baseline_eer"]},
            {"model": "robust_p20", "condition": condition, "dev_eer": robust_dev,
             "eval_eer": eval_lookup[condition]["robust_mean_eer"],
             "eval_minus_dev_eer": eval_lookup[condition]["robust_mean_eer"] - robust_dev,
             "dev_delta_from_clean": robust_dev - float(np.mean([x["metrics"]["clean"]["eer"] for x in p20_summaries])),
             "eval_delta_from_clean": eval_lookup[condition]["robust_mean_eer"] - eval_lookup["clean"]["robust_mean_eer"]},
        ]
    uncertainty = {"method": config["uncertainty"]["method"], "conditions": {}}
    for offset, condition in enumerate(config["uncertainty"]["conditions"]):
        uncertainty["conditions"][condition] = paired_bootstrap(
            labels, arrays[condition]["baseline"],
            [arrays[condition][f"robust_{seed}"] for seed in seeds],
            baseline_threshold, robust_thresholds, config["uncertainty"]["replicates"],
            config["uncertainty"]["seed"] + offset,
        )
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(METRICS_DIR / "final-evaluation-main.csv", main_rows)
    write_csv(METRICS_DIR / "final-evaluation-per-seed.csv", per_seed_rows)
    write_csv(METRICS_DIR / "final-evaluation-by-attack.csv", attack_rows)
    write_csv(METRICS_DIR / "final-evaluation-score-shifts.csv", shift_rows)
    write_csv(METRICS_DIR / "final-evaluation-transitions.csv", transition_rows)
    write_csv(METRICS_DIR / "final-evaluation-dev-vs-eval.csv", generalization_rows)
    write_json_atomic(METRICS_DIR / "final-evaluation-bootstrap.json", uncertainty)
    make_figures(main_rows, attack_rows)
    summary = {
        "protocol_id": config["protocol"]["id"], "frozen_protocol_commit": git_commit(),
        "trials": len(records), "main": main_rows, "bootstrap": uncertainty,
        "runtime": {
            condition: json.loads((output_dir / "conditions" / f"{condition}.json").read_text())
            for condition in condition_names
        },
        "evaluation_split_accessed": True, "post_eval_model_selection": False,
        "post_eval_retraining": False,
    }
    write_json_atomic(METRICS_DIR / "final-evaluation-summary.json", summary)
    write_json_atomic(output_dir / "final-summary.json", summary)
    print(json.dumps({"protocol_id": summary["protocol_id"], "trials": len(records), "main": main_rows}, indent=2))


def make_figures(main_rows, attack_rows):
    conditions = [row["condition"] for row in main_rows]
    x = np.arange(len(conditions))
    width = 0.38
    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(x - width / 2, [row["baseline_eer"] for row in main_rows], width, label="Clean-trained")
    axis.bar(x + width / 2, [row["robust_mean_eer"] for row in main_rows], width, label="p=0.20")
    axis.errorbar(x + width / 2, [row["robust_mean_eer"] for row in main_rows],
                  yerr=[row["robust_std_eer"] for row in main_rows], fmt="none", color="black", capsize=3)
    axis.set_xticks(x, conditions, rotation=30, ha="right")
    axis.set_ylabel("EER")
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final-evaluation-eer.png", dpi=180)
    plt.close(fig)
    attacks = [f"A{i:02d}" for i in range(7, 20)]
    heat_conditions = ["clean", "noise_20db", "noise_10db", "noise_5db"]
    lookup = {(r["condition"], r["attack_id"]): r for r in attack_rows}
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharex=True, sharey=True)
    for axis, field, title in [
        (axes[0], "baseline_eer", "Clean-trained"), (axes[1], "robust_mean_eer", "p=0.20 mean")
    ]:
        matrix = np.array([[lookup[(condition, attack)][field] for condition in heat_conditions] for attack in attacks])
        image = axis.imshow(matrix, vmin=0, vmax=max(0.5, float(matrix.max())), aspect="auto", cmap="magma")
        axis.set_xticks(range(4), ["Clean", "20 dB", "10 dB", "5 dB"])
        axis.set_yticks(range(len(attacks)), attacks)
        axis.set_title(title)
    fig.colorbar(image, ax=axes, label="EER", shrink=0.8)
    fig.subplots_adjust(left=0.08, right=0.9, bottom=0.08, top=0.92, wspace=0.12)
    fig.savefig(FIGURES_DIR / "final-evaluation-attack-heatmap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["preflight", "condition", "analyze"])
    parser.add_argument("--condition", choices=list(SPEC_BY_NAME))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    roots = ArtifactRoots.from_environment()
    for name, value in {
        "AUDIO_DATA_ROOT": roots.data, "AUDIO_MODEL_ROOT": roots.models,
        "AUDIO_CACHE_ROOT": roots.cache, "AUDIO_OUTPUT_ROOT": roots.outputs,
    }.items():
        os.environ.setdefault(name, str(value))
    config = load_config(CONFIG_PATH)
    if args.phase == "preflight":
        preflight(config, roots)
    elif args.phase == "condition":
        if args.condition is None:
            raise ValueError("--condition is required")
        run_condition(config, roots, args.condition, torch.device(args.device))
    else:
        analyze(config, roots)


if __name__ == "__main__":
    main()
