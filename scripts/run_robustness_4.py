from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from audio_deepfake.baseline import (
    BinaryClassificationHead,
    load_development_manifest,
    load_head_checkpoint,
    write_records_csv,
)
from audio_deepfake.config import ArtifactRoots, load_config
from audio_deepfake.corruptions import (
    CORE_CORRUPTIONS,
    CorruptedAudioDataset,
    CorruptionSpec,
    apply_corruption,
    paired_bootstrap_conditions,
    signal_to_noise_db,
    validate_alignment,
)
from audio_deepfake.metrics import binary_metrics
from audio_deepfake.utils import set_seed, write_json_atomic

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "robustness_eval.yaml"
SPECS = {spec.name: spec for spec in CORE_CORRUPTIONS}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_models(config: dict[str, Any], roots: ArtifactRoots, device: torch.device):
    from transformers import WavLMModel

    encoder = WavLMModel.from_pretrained(
        config["frozen"]["model"],
        revision=config["frozen"]["revision"],
        cache_dir=roots.cache / "huggingface",
        local_files_only=True,
    )
    encoder.eval().requires_grad_(False).to(device)
    head, payload = load_head_checkpoint(
        Path(config["paths"]["checkpoint"]), encoder.config.hidden_size, 0.1, device
    )
    head.eval().requires_grad_(False)
    return encoder, head, payload


def condition_signature(
    spec: CorruptionSpec, config: dict[str, Any], manifest: Path, utterance_ids: list[str]
) -> dict[str, Any]:
    return {
        "condition": spec.__dict__,
        "config_sha256": sha256(CONFIG_PATH),
        "manifest_sha256": sha256(manifest),
        "utterance_ids_sha256": hashlib.sha256("\n".join(utterance_ids).encode()).hexdigest(),
        "checkpoint_sha256": config["frozen"]["checkpoint_sha256"],
        "model_revision": config["frozen"]["revision"],
        "seed": config["experiment"]["seed"],
    }


def predict_head(
    head: BinaryClassificationHead,
    embeddings: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(TensorDataset(embeddings), batch_size=batch_size, shuffle=False)
    values: list[torch.Tensor] = []
    with torch.inference_mode():
        for (features,) in loader:
            values.append(torch.sigmoid(head(features.to(device, non_blocking=True))).cpu())
    return torch.cat(values).numpy()


def extract_condition(
    spec: CorruptionSpec,
    records,
    manifest: Path,
    config: dict[str, Any],
    cache_path: Path,
    encoder,
    head,
    device: torch.device,
) -> tuple[dict[str, Any], float, int]:
    signature = condition_signature(
        spec, config, manifest, [record.utterance_id for record in records]
    )
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload["signature"] != signature:
            raise ValueError(f"cache signature mismatch: {cache_path}")
        return payload, 0.0, 0
    dataset = CorruptedAudioDataset(
        records,
        spec,
        config["experiment"]["seed"],
        config["frozen"]["sample_rate"],
        config["frozen"]["max_duration_seconds"],
        config["frozen"]["crop_strategy"],
    )
    loader = DataLoader(
        dataset,
        batch_size=config["inference"]["batch_size"],
        shuffle=False,
        num_workers=config["inference"]["num_workers"],
        pin_memory=True,
        persistent_workers=config["inference"]["num_workers"] > 0,
    )
    embeddings: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=spec.name):
            waveform = batch["waveform"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = encoder(waveform, attention_mask=attention_mask).last_hidden_state
                feature_mask = encoder._get_feature_vector_attention_mask(
                    hidden.shape[1], attention_mask
                )
                weights = feature_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
            embeddings.append(pooled.float().cpu())
            indices.append(batch["index"])
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device)
    order = torch.cat(indices).numpy()
    if not np.array_equal(order, np.arange(len(records))):
        raise RuntimeError(f"sample order changed for {spec.name}")
    features = torch.cat(embeddings)
    scores = predict_head(head, features, config["inference"]["batch_size"], device)
    payload = {
        "signature": signature,
        "utterance_ids": [record.utterance_id for record in records],
        "embeddings": features,
        "scores": torch.from_numpy(scores),
        "runtime_seconds": elapsed,
        "peak_gpu_bytes": peak,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(cache_path)
    return payload, elapsed, peak


def preflight(config: dict[str, Any], records, output_dir: Path) -> None:
    selected = records[:4]
    rows: list[dict[str, Any]] = []
    import soundfile as sf

    for record in selected:
        waveform, rate = sf.read(record.audio_path, dtype="float32")
        for spec in CORE_CORRUPTIONS:
            transformed = apply_corruption(
                waveform, spec, config["experiment"]["seed"], record.utterance_id, rate
            )
            row = {
                "utterance_id": record.utterance_id,
                "condition": spec.name,
                "samples": len(transformed),
                "sample_rate": rate,
                "finite": bool(np.all(np.isfinite(transformed))),
                "rms": float(np.sqrt(np.mean(transformed.astype(np.float64) ** 2))),
            }
            if spec.kind == "noise":
                row["measured_snr_db"] = signal_to_noise_db(waveform, transformed)
            rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "preflight.json", {"checks": rows, "passed": True})
    print(json.dumps({"preflight_checks": len(rows), "passed": True}, indent=2))


def load_phase3_scores() -> tuple[list[str], np.ndarray]:
    path = Path.home() / "audio/outputs/experiments/wavlm-base-plus-frozen-v1/full/dev-predictions.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row["utterance_id"] for row in rows], np.array([float(row["score"]) for row in rows])


def per_attack_metrics(labels, scores, attacks, threshold, condition):
    rows = []
    for attack in [f"A{number:02d}" for number in range(1, 7)]:
        mask = np.array([value in {"-", attack} for value in attacks])
        metrics = binary_metrics(labels[mask], scores[mask], threshold)
        predictions = scores[mask] >= threshold
        selected_attacks = np.asarray(attacks)[mask]
        false_negatives = int(np.sum((selected_attacks == attack) & ~predictions))
        rows.append(
            {
                "condition": condition,
                "attack_id": attack,
                "spoof_trials": int(np.sum(selected_attacks == attack)),
                "false_negatives": false_negatives,
                **metrics,
            }
        )
    return rows


def per_duration_metrics(labels, scores, durations, threshold, condition):
    rows = []
    for lower, upper in [(0, 2), (2, 3), (3, 4), (4, float("inf"))]:
        mask = (durations > lower) & (durations <= upper)
        metrics = binary_metrics(labels[mask], scores[mask], threshold)
        rows.append(
            {
                "condition": condition,
                "duration_bin": f"({lower},{upper}]",
                "trials": int(mask.sum()),
                **metrics,
            }
        )
    return rows


def transition_counts(labels, clean_scores, scores, attacks, threshold, condition):
    clean_correct = (clean_scores >= threshold).astype(int) == labels
    shifted_correct = (scores >= threshold).astype(int) == labels
    rows = []
    groups = [("all", np.ones(len(labels), dtype=bool)), ("bonafide", labels == 0), ("spoof", labels == 1)]
    groups.extend((attack, np.asarray(attacks) == attack) for attack in [f"A{i:02d}" for i in range(1, 7)])
    for group, mask in groups:
        rows.append(
            {
                "condition": condition,
                "group": group,
                "clean_correct_corrupt_correct": int(np.sum(mask & clean_correct & shifted_correct)),
                "clean_correct_corrupt_incorrect": int(np.sum(mask & clean_correct & ~shifted_correct)),
                "clean_incorrect_corrupt_correct": int(np.sum(mask & ~clean_correct & shifted_correct)),
                "clean_incorrect_corrupt_incorrect": int(np.sum(mask & ~clean_correct & ~shifted_correct)),
            }
        )
    return rows


def analyze(config, records, payloads, output_dir):
    import matplotlib.pyplot as plt

    threshold = config["frozen"]["threshold"]
    labels = np.array([record.label for record in records])
    attacks = [record.attack_id for record in records]
    durations = np.array([record.duration_seconds for record in records])
    scores_by_condition = {name: payload["scores"].numpy() for name, payload in payloads.items()}
    clean_scores = scores_by_condition["clean"]
    clean_metrics = binary_metrics(labels, clean_scores, threshold)
    main_rows = []
    attack_rows = []
    duration_rows = []
    transition_rows = []
    score_shift_rows = []
    prediction_rows = []
    for condition in config["conditions"]:
        scores = scores_by_condition[condition]
        metrics = binary_metrics(labels, scores, threshold)
        main_rows.append(
            {
                "condition": condition,
                **metrics,
                **{f"delta_{key}": float(metrics[key]) - float(clean_metrics[key]) for key in ["eer", "roc_auc", "pr_auc", "balanced_accuracy", "precision", "recall", "f1"]},
                "runtime_seconds": payloads[condition]["runtime_seconds"],
            }
        )
        attack_rows.extend(per_attack_metrics(labels, scores, attacks, threshold, condition))
        duration_rows.extend(per_duration_metrics(labels, scores, durations, threshold, condition))
        transition_rows.extend(
            transition_counts(labels, clean_scores, scores, attacks, threshold, condition)
        )
        for label_name, label_value in [("bonafide", 0), ("spoof", 1)]:
            mask = labels == label_value
            score_shift_rows.append(
                {
                    "condition": condition,
                    "label": label_name,
                    "mean_score": float(scores[mask].mean()),
                    "median_score": float(np.median(scores[mask])),
                    "mean_shift_from_clean": float(scores[mask].mean() - clean_scores[mask].mean()),
                    "median_shift_from_clean": float(np.median(scores[mask]) - np.median(clean_scores[mask])),
                }
            )
        prediction_rows.extend(
            {
                "utterance_id": record.utterance_id,
                "condition": condition,
                "label": record.label,
                "speaker_id": record.speaker_id,
                "attack_id": record.attack_id,
                "duration_seconds": record.duration_seconds,
                "score": float(scores[index]),
                "prediction": int(scores[index] >= threshold),
            }
            for index, record in enumerate(records)
        )

    uncertainty = paired_bootstrap_conditions(
        labels,
        scores_by_condition,
        "clean",
        threshold,
        config["uncertainty"]["replicates"],
        config["uncertainty"]["seed"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_records_csv(output_dir / "predictions.csv", prediction_rows)
    write_json_atomic(output_dir / "bootstrap.json", uncertainty)

    metrics_dir = REPOSITORY_ROOT / "results/metrics"
    figures_dir = REPOSITORY_ROOT / "results/figures"
    write_records_csv(metrics_dir / "robustness-main.csv", main_rows)
    write_records_csv(metrics_dir / "robustness-by-attack.csv", attack_rows)
    write_records_csv(metrics_dir / "robustness-by-duration.csv", duration_rows)
    write_records_csv(metrics_dir / "robustness-transitions.csv", transition_rows)
    write_records_csv(metrics_dir / "robustness-score-shifts.csv", score_shift_rows)
    write_json_atomic(metrics_dir / "robustness-bootstrap.json", uncertainty)

    conditions = config["conditions"]
    values = {row["condition"]: row for row in main_rows}
    for metric, filename in [("eer", "robustness-eer.png"), ("recall", "robustness-recall.png")]:
        plt.figure(figsize=(9, 4))
        plt.bar(conditions, [values[name][metric] for name in conditions])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel(metric.upper())
        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=160)
        plt.close()

    plt.figure(figsize=(9, 4))
    x = np.arange(len(conditions))
    width = 0.25
    relative_eer = [(values[name]["eer"] - clean_metrics["eer"]) / clean_metrics["eer"] for name in conditions]
    relative_recall = [(clean_metrics["recall"] - values[name]["recall"]) / clean_metrics["recall"] for name in conditions]
    relative_f1 = [(clean_metrics["f1"] - values[name]["f1"]) / clean_metrics["f1"] for name in conditions]
    plt.bar(x - width, relative_eer, width, label="EER increase")
    plt.bar(x, relative_recall, width, label="Recall loss")
    plt.bar(x + width, relative_f1, width, label="F1 loss")
    plt.xticks(x, conditions, rotation=35, ha="right")
    plt.ylabel("Relative change from clean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "robustness-relative-degradation.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, label_value, label_name in zip(axes, [0, 1], ["bonafide", "spoof"], strict=True):
        data = [scores_by_condition[name][labels == label_value] for name in conditions]
        axis.boxplot(data, tick_labels=conditions, showfliers=False)
        axis.tick_params(axis="x", rotation=35)
        axis.set_title(label_name)
        axis.set_ylabel("Spoof probability")
    fig.tight_layout()
    fig.savefig(figures_dir / "robustness-score-shifts.png", dpi=160)
    plt.close(fig)

    attack_lookup = {(row["condition"], row["attack_id"]): row for row in attack_rows}
    attacks_order = [f"A{i:02d}" for i in range(1, 7)]
    matrix = np.array(
        [[attack_lookup[(condition, attack)]["recall"] for attack in attacks_order] for condition in conditions]
    )
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(attacks_order)), attacks_order)
    axis.set_yticks(range(len(conditions)), conditions)
    axis.set_xlabel("Attack ID")
    axis.set_title("Recall at frozen clean threshold")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(figures_dir / "robustness-attack-recall.png", dpi=160)
    plt.close(fig)

    a06_mask = np.asarray(attacks) == "A06"
    clean_positive = clean_scores >= threshold
    a06 = {}
    for condition in conditions:
        shifted_positive = scores_by_condition[condition] >= threshold
        a06[condition] = {
            "clean_fn_remains_fn": int(np.sum(a06_mask & ~clean_positive & ~shifted_positive)),
            "clean_tp_becomes_fn": int(np.sum(a06_mask & clean_positive & ~shifted_positive)),
            "clean_fn_becomes_tp": int(np.sum(a06_mask & ~clean_positive & shifted_positive)),
        }
    summary = {
        "experiment_id": config["experiment"]["id"],
        "git_commit": git_commit(),
        "checkpoint_sha256": config["frozen"]["checkpoint_sha256"],
        "threshold": threshold,
        "conditions": conditions,
        "main": main_rows,
        "a06_transitions": a06,
        "bootstrap_replicates": config["uncertainty"]["replicates"],
        "bootstrap_seed": config["uncertainty"]["seed"],
        "evaluation_split_accessed": False,
    }
    write_json_atomic(metrics_dir / "robustness-summary.json", summary)
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("preflight", "clean", "full"))
    args = parser.parse_args()
    roots = ArtifactRoots.from_environment()
    for name, value in {
        "AUDIO_DATA_ROOT": roots.data,
        "AUDIO_MODEL_ROOT": roots.models,
        "AUDIO_CACHE_ROOT": roots.cache,
        "AUDIO_OUTPUT_ROOT": roots.outputs,
    }.items():
        os.environ.setdefault(name, str(value))
    config = load_config(CONFIG_PATH)
    checkpoint = Path(config["paths"]["checkpoint"])
    if sha256(checkpoint) != config["frozen"]["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint hash mismatch")
    if set(config["conditions"]) != set(SPECS):
        raise ValueError("configured conditions do not match frozen core conditions")
    set_seed(config["experiment"]["seed"])
    device = torch.device(config["inference"]["device"])
    manifest = Path(config["paths"]["dev_manifest"])
    records = load_development_manifest(manifest, roots.data)
    if any(record.split != "dev" for record in records):
        raise PermissionError("Phase 4 accepts DEV only")
    output_dir = Path(config["paths"]["output_dir"])
    if args.phase == "preflight":
        preflight(config, records, output_dir)
        return

    encoder, head, checkpoint_payload = load_models(config, roots, device)
    if checkpoint_payload["epoch"] != 30:
        raise ValueError("unexpected selected checkpoint epoch")
    cache_root = Path(config["paths"]["feature_cache"])
    selected_conditions = ["clean"] if args.phase == "clean" else config["conditions"]
    payloads = {}
    runtime = 0.0
    peak = 0
    for condition in selected_conditions:
        payload, elapsed, condition_peak = extract_condition(
            SPECS[condition], records, manifest, config, cache_root / f"{condition}.pt",
            encoder, head, device,
        )
        payloads[condition] = payload
        runtime += elapsed
        peak = max(peak, condition_peak)
        print(f"condition={condition} seconds={elapsed:.3f} peak_bytes={condition_peak}", flush=True)

    phase3_ids, phase3_scores = load_phase3_scores()
    validate_alignment(phase3_ids, payloads["clean"]["utterance_ids"])
    maximum_difference = float(
        np.max(np.abs(phase3_scores - payloads["clean"]["scores"].numpy()))
    )
    if maximum_difference > 1e-6:
        raise RuntimeError(f"clean reproduction failed: max score difference {maximum_difference}")
    labels = np.array([record.label for record in records])
    clean_metrics = binary_metrics(labels, payloads["clean"]["scores"].numpy(), config["frozen"]["threshold"])
    reproduction = {
        "max_abs_score_difference": maximum_difference,
        "metrics": clean_metrics,
        "phase3_metrics": json.loads((REPOSITORY_ROOT / "results/metrics/baseline-dev-summary.json").read_text())["metrics"],
        "within_tolerance": True,
        "runtime_seconds": runtime,
        "peak_gpu_bytes": peak,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "clean-reproduction.json", reproduction)
    if args.phase == "clean":
        print(json.dumps(reproduction, indent=2))
        return
    summary = analyze(config, records, payloads, output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
