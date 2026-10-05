from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from audio_deepfake.augmentation import (
    NoiseAugmentedTrainingDataset,
    noise_augmentation_decision,
    select_robust_checkpoint,
    validate_paired_ids,
)
from audio_deepfake.baseline import (
    BinaryClassificationHead,
    load_development_manifest,
    load_head_checkpoint,
    save_head_checkpoint,
    write_records_csv,
)
from audio_deepfake.config import ArtifactRoots, load_config
from audio_deepfake.metrics import (
    binary_metrics,
    equal_error_rate,
    select_balanced_accuracy_threshold,
)
from audio_deepfake.utils import set_seed, write_json_atomic

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "noise_augmentation_5.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def predict(head, embeddings, batch_size, device):
    head.eval()
    values = []
    loader = DataLoader(TensorDataset(embeddings), batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for (features,) in loader:
            values.append(torch.sigmoid(head(features.to(device, non_blocking=True))).cpu())
    return torch.cat(values).numpy()


def load_encoder(config, roots, device):
    from transformers import WavLMModel

    model = WavLMModel.from_pretrained(
        config["model"]["name"], revision=config["model"]["revision"],
        cache_dir=roots.cache / "huggingface", local_files_only=True,
    )
    model.eval().requires_grad_(False).to(device)
    if model.config.hidden_size != config["model"]["hidden_size"]:
        raise ValueError("WavLM hidden size mismatch")
    return model


def augmented_signature(config, manifest, records, seed):
    return {
        "config_sha256": sha256(CONFIG_PATH),
        "manifest_sha256": sha256(manifest),
        "record_ids_sha256": hashlib.sha256(
            "\n".join(record.utterance_id for record in records).encode()
        ).hexdigest(),
        "model": config["model"]["name"],
        "revision": config["model"]["revision"],
        "augmentation": config["augmentation"],
        "seed": seed,
        "count": len(records),
    }


def extract_augmented_train(config, roots, records, manifest, seed, device):
    cache = Path(config["paths"]["feature_cache"]) / f"seed-{seed}-train.pt"
    signature = augmented_signature(config, manifest, records, seed)
    if cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if payload["signature"] != signature:
            raise ValueError(f"augmented feature-cache signature mismatch: {cache}")
        return payload, 0.0, 0
    encoder = load_encoder(config, roots, device)
    dataset = NoiseAugmentedTrainingDataset(
        records, seed, config["augmentation"]["probability"],
        config["augmentation"]["snr_db_choices"], config["preprocessing"]["sample_rate"],
        config["preprocessing"]["max_duration_seconds"],
        config["preprocessing"]["crop_strategy"],
    )
    loader = DataLoader(
        dataset, batch_size=config["feature_extraction"]["batch_size"], shuffle=False,
        num_workers=config["feature_extraction"]["num_workers"], pin_memory=True,
        persistent_workers=config["feature_extraction"]["num_workers"] > 0,
    )
    embeddings, indices = [], []
    augmented, snrs, measured = [], [], []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"augmented train seed {seed}"):
            waveform = batch["waveform"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = encoder(waveform, attention_mask=mask).last_hidden_state
                feature_mask = encoder._get_feature_vector_attention_mask(hidden.shape[1], mask)
                weights = feature_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
            embeddings.append(pooled.float().cpu())
            indices.append(batch["index"])
            augmented.extend(batch["augmented"].tolist())
            snrs.extend(batch["snr_db"].tolist())
            measured.extend(batch["measured_snr_db"].tolist())
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device)
    order = torch.cat(indices).numpy()
    if not np.array_equal(order, np.arange(len(records))):
        raise RuntimeError("augmented feature extraction changed TRAIN order")
    payload = {
        "signature": signature,
        "embeddings": torch.cat(embeddings),
        "labels": torch.tensor([record.label for record in records], dtype=torch.float32),
        "utterance_ids": [record.utterance_id for record in records],
        "augmented": augmented,
        "snr_db": snrs,
        "measured_snr_db": measured,
        "runtime_seconds": elapsed,
        "peak_gpu_bytes": peak,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(cache)
    del encoder
    torch.cuda.empty_cache()
    return payload, elapsed, peak


def load_dev_payloads(config, records):
    ids = [record.utterance_id for record in records]
    payloads = {}
    for condition in config["evaluation_conditions"]:
        path = Path(config["paths"]["phase4_cache"]) / f"{condition}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_paired_ids(ids, payload["utterance_ids"])
        if payload["signature"]["condition"]["name"] != condition:
            raise ValueError(f"condition cache mismatch: {condition}")
        payloads[condition] = payload
    return payloads


def augmentation_summary(payload, choices):
    augmented = np.asarray(payload["augmented"], dtype=bool)
    snrs = np.asarray(payload["snr_db"], dtype=float)
    measured = np.asarray(payload["measured_snr_db"], dtype=float)
    return {
        "total": len(augmented),
        "augmented": int(augmented.sum()),
        "clean": int((~augmented).sum()),
        "observed_probability": float(augmented.mean()),
        "counts_by_snr": {
            str(int(value)): int(np.sum(augmented & (snrs == value))) for value in choices
        },
        "maximum_absolute_snr_error_db": float(
            np.max(np.abs(measured[augmented] - snrs[augmented]))
        ),
    }


def train_seed(config, roots, seed, device):
    train_manifest = Path(config["paths"]["train_manifest"])
    dev_manifest = Path(config["paths"]["dev_manifest"])
    train_records = load_development_manifest(train_manifest, roots.data)
    dev_records = load_development_manifest(dev_manifest, roots.data)
    if any(record.split != "train" for record in train_records):
        raise PermissionError("Phase 5 training accepts TRAIN only")
    if any(record.split != "dev" for record in dev_records):
        raise PermissionError("Phase 5 selection accepts DEV only")
    train_payload, extraction_seconds, peak_extract = extract_augmented_train(
        config, roots, train_records, train_manifest, seed, device
    )
    validate_paired_ids([r.utterance_id for r in train_records], train_payload["utterance_ids"])
    dev_payloads = load_dev_payloads(config, dev_records)
    labels = np.array([record.label for record in dev_records])

    set_seed(seed)
    head = BinaryClassificationHead(config["model"]["hidden_size"], config["model"]["dropout"]).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    train_labels = train_payload["labels"]
    negative, positive = int((train_labels == 0).sum()), int((train_labels == 1).sum())
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative / positive], device=device)
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_payload["embeddings"], train_labels),
        batch_size=config["training"]["batch_size"], shuffle=True, generator=generator,
    )
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"]) / f"seed-{seed}"
    output_dir = Path(config["paths"]["output_dir"]) / f"seed-{seed}"
    log_path = output_dir / "epochs.jsonl"
    if log_path.exists():
        raise FileExistsError(f"refusing to overwrite run: {log_path}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    history, stale = [], 0
    best_key = None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, config["training"]["epochs"] + 1):
        head.train()
        total_loss = 0.0
        count = 0
        for features, target in loader:
            optimizer.zero_grad(set_to_none=True)
            features, target = features.to(device), target.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = criterion(head(features), target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            count += len(target)
        condition_metrics = {}
        for condition in config["selection_conditions"]:
            scores = predict(
                head, dev_payloads[condition]["embeddings"],
                config["training"]["batch_size"], device,
            )
            condition_metrics[condition] = binary_metrics(labels, scores, 0.5)
        mean_eer = float(np.mean([v["eer"] for v in condition_metrics.values()]))
        record = {
            "epoch": epoch,
            "train_loss": total_loss / count,
            "mean_selection_eer": mean_eer,
            "clean_eer": condition_metrics["clean"]["eer"],
            "condition_metrics": condition_metrics,
        }
        history.append(record)
        with log_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        path = checkpoint_dir / f"epoch-{epoch}.pt"
        save_head_checkpoint(path, head, optimizer, epoch, {
            "seed": seed, "metrics": condition_metrics,
            "mean_selection_eer": mean_eer, "config_sha256": sha256(CONFIG_PATH),
        })
        key = (mean_eer, condition_metrics["clean"]["eer"], epoch)
        if best_key is None or key < best_key:
            best_key, stale = key, 0
        else:
            stale += 1
        print(
            f"seed={seed} epoch={epoch} loss={record['train_loss']:.6f} "
            f"mean_eer={mean_eer:.6f} clean_eer={record['clean_eer']:.6f}", flush=True,
        )
        if stale >= config["training"]["early_stopping_patience"]:
            break
    training_seconds = time.perf_counter() - started
    selected_record = select_robust_checkpoint(history)
    selected_source = checkpoint_dir / f"epoch-{selected_record['epoch']}.pt"
    selected_path = checkpoint_dir / "selected.pt"
    shutil.copy2(selected_source, selected_path)
    selected_head, selected_payload = load_head_checkpoint(
        selected_path, config["model"]["hidden_size"], config["model"]["dropout"], device
    )
    if selected_payload["epoch"] != selected_record["epoch"]:
        raise RuntimeError("selected checkpoint epoch mismatch")
    clean_scores = predict(
        selected_head, dev_payloads["clean"]["embeddings"],
        config["training"]["batch_size"], device,
    )
    threshold = select_balanced_accuracy_threshold(labels, clean_scores)
    rows, metrics = [], {}
    for condition in config["evaluation_conditions"]:
        scores = predict(
            selected_head, dev_payloads[condition]["embeddings"],
            config["training"]["batch_size"], device,
        )
        metrics[condition] = binary_metrics(labels, scores, threshold)
        for index, record in enumerate(dev_records):
            rows.append({
                "utterance_id": record.utterance_id,
                "condition": condition,
                "label": record.label,
                "speaker_id": record.speaker_id,
                "attack_id": record.attack_id,
                "duration_seconds": record.duration_seconds,
                "score": float(scores[index]),
                "prediction": int(scores[index] >= threshold),
            })
    write_records_csv(output_dir / "predictions.csv", rows)
    summary = {
        "experiment_id": config["experiment"]["id"],
        "git_commit": git_commit(),
        "seed": seed,
        "selected_epoch": selected_record["epoch"],
        "selection_rule": config["training"]["checkpoint_selection"],
        "selection_mean_eer": selected_record["mean_selection_eer"],
        "threshold": threshold,
        "threshold_rule": config["training"]["threshold_selection"],
        "metrics": metrics,
        "history": history,
        "augmentation": augmentation_summary(
            train_payload, config["augmentation"]["snr_db_choices"]
        ),
        "feature_extraction_seconds": extraction_seconds,
        "training_seconds": training_seconds,
        "peak_gpu_bytes": max(peak_extract, torch.cuda.max_memory_allocated(device)),
        "checkpoint": str(selected_path),
        "checkpoint_sha256": sha256(selected_path),
        "evaluation_split_accessed": False,
    }
    write_json_atomic(output_dir / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


def metric_subset(labels, scores, threshold):
    predictions = scores >= threshold
    eer, _ = equal_error_rate(labels, scores)
    return {
        "eer": eer,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def bootstrap_deltas(labels, baseline, robust, baseline_threshold, robust_thresholds, replicates, seed):
    generator = np.random.default_rng(seed)
    negative, positive = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    values = {key: [] for key in ["eer", "roc_auc", "balanced_accuracy", "f1"]}
    for _ in range(replicates):
        indices = np.concatenate([
            generator.choice(negative, len(negative), replace=True),
            generator.choice(positive, len(positive), replace=True),
        ])
        selected = labels[indices]
        base = metric_subset(selected, baseline[indices], baseline_threshold)
        seed_metrics = [
            metric_subset(selected, scores[indices], threshold)
            for scores, threshold in zip(robust, robust_thresholds, strict=True)
        ]
        for metric, samples in values.items():
            samples.append(float(np.mean([x[metric] for x in seed_metrics])) - base[metric])
    return {
        metric: [float(x) for x in np.quantile(samples, [0.025, 0.975])]
        for metric, samples in values.items()
    }


def load_prediction_table(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def analyze(config, roots):
    seeds = config["experiment"]["final_seeds"]
    baseline_rows = load_prediction_table(
        roots.outputs / "experiments/wavlm-base-plus-robustness-v1/predictions.csv"
    )
    robust_rows = {
        seed: load_prediction_table(Path(config["paths"]["output_dir"]) / f"seed-{seed}/predictions.csv")
        for seed in seeds
    }
    summaries = {
        seed: json.loads(
            (Path(config["paths"]["output_dir"]) / f"seed-{seed}/summary.json").read_text()
        ) for seed in seeds
    }
    conditions = config["evaluation_conditions"]
    baseline_threshold = json.loads(
        (REPOSITORY_ROOT / "results/metrics/robustness-summary.json").read_text()
    )["threshold"]
    thresholds = [summaries[seed]["threshold"] for seed in seeds]
    main_rows, per_seed_rows, attack_rows, transition_rows, score_rows = [], [], [], [], []
    bootstrap = {}
    for condition_index, condition in enumerate(conditions):
        base = [r for r in baseline_rows if r["condition"] == condition]
        robust = {
            seed: [r for r in robust_rows[seed] if r["condition"] == condition] for seed in seeds
        }
        ids = [r["utterance_id"] for r in base]
        labels = np.array([int(r["label"]) for r in base])
        base_scores = np.array([float(r["score"]) for r in base])
        base_metrics = binary_metrics(labels, base_scores, baseline_threshold)
        robust_scores = []
        robust_metrics = []
        for seed, threshold in zip(seeds, thresholds, strict=True):
            validate_paired_ids(ids, [r["utterance_id"] for r in robust[seed]])
            scores = np.array([float(r["score"]) for r in robust[seed]])
            robust_scores.append(scores)
            result = binary_metrics(labels, scores, threshold)
            robust_metrics.append(result)
            per_seed_rows.append({"condition": condition, "seed": seed, **result})
        row = {"condition": condition}
        for metric in ["eer", "roc_auc", "pr_auc", "balanced_accuracy", "precision", "recall", "f1"]:
            values = np.array([result[metric] for result in robust_metrics])
            row[f"baseline_{metric}"] = base_metrics[metric]
            row[f"robust_mean_{metric}"] = float(values.mean())
            row[f"robust_std_{metric}"] = float(values.std(ddof=1))
            row[f"delta_{metric}"] = float(values.mean() - base_metrics[metric])
        main_rows.append(row)
        bootstrap[condition] = bootstrap_deltas(
            labels, base_scores, robust_scores, baseline_threshold, thresholds,
            config["uncertainty"]["replicates"], config["uncertainty"]["seed"] + condition_index,
        )
        attacks = np.array([r["attack_id"] for r in base])
        for attack in [f"A{i:02d}" for i in range(1, 7)]:
            mask = np.isin(attacks, ["-", attack])
            base_attack = binary_metrics(labels[mask], base_scores[mask], baseline_threshold)
            for seed, scores, threshold in zip(seeds, robust_scores, thresholds, strict=True):
                result = binary_metrics(labels[mask], scores[mask], threshold)
                attack_mask = attacks == attack
                attack_rows.append({
                    "condition": condition, "attack_id": attack, "seed": seed,
                    "baseline_eer": base_attack["eer"], "robust_eer": result["eer"],
                    "baseline_recall": base_attack["recall"], "robust_recall": result["recall"],
                    "baseline_false_negatives": int(np.sum(attack_mask & (base_scores < baseline_threshold))),
                    "robust_false_negatives": int(np.sum(attack_mask & (scores < threshold))),
                })
        base_correct = (base_scores >= baseline_threshold) == labels
        for seed, scores, threshold in zip(seeds, robust_scores, thresholds, strict=True):
            robust_correct = (scores >= threshold) == labels
            groups = [("all", np.ones(len(labels), dtype=bool)), ("bonafide", labels == 0), ("spoof", labels == 1)]
            groups += [(attack, attacks == attack) for attack in [f"A{i:02d}" for i in range(1, 7)]]
            for group, mask in groups:
                transition_rows.append({
                    "condition": condition, "seed": seed, "group": group,
                    "both_correct": int(np.sum(mask & base_correct & robust_correct)),
                    "baseline_correct_robust_wrong": int(np.sum(mask & base_correct & ~robust_correct)),
                    "baseline_wrong_robust_correct": int(np.sum(mask & ~base_correct & robust_correct)),
                    "both_wrong": int(np.sum(mask & ~base_correct & ~robust_correct)),
                })
        for label_name, label_value in [("bonafide", 0), ("spoof", 1)]:
            mask = labels == label_value
            score_rows.append({
                "condition": condition, "label": label_name,
                "baseline_mean": float(base_scores[mask].mean()),
                "baseline_median": float(np.median(base_scores[mask])),
                "robust_mean": float(np.mean([scores[mask].mean() for scores in robust_scores])),
                "robust_median": float(np.mean([np.median(scores[mask]) for scores in robust_scores])),
            })
    metrics_dir = REPOSITORY_ROOT / "results/metrics"
    write_records_csv(metrics_dir / "noise-augmentation-main.csv", main_rows)
    write_records_csv(metrics_dir / "noise-augmentation-per-seed.csv", per_seed_rows)
    write_records_csv(metrics_dir / "noise-augmentation-by-attack.csv", attack_rows)
    write_records_csv(metrics_dir / "noise-augmentation-transitions.csv", transition_rows)
    write_records_csv(metrics_dir / "noise-augmentation-score-shifts.csv", score_rows)
    write_json_atomic(metrics_dir / "noise-augmentation-bootstrap.json", bootstrap)
    aggregate = {
        "experiment_id": config["experiment"]["id"], "git_commit": git_commit(),
        "seeds": seeds, "thresholds": thresholds, "main": main_rows,
        "selected_epochs": {str(seed): summaries[seed]["selected_epoch"] for seed in seeds},
        "mean_eer_clean_and_noise": float(np.mean([
            row["robust_mean_eer"] for row in main_rows
            if row["condition"] in config["selection_conditions"]
        ])),
        "mean_eer_all_conditions": float(np.mean([row["robust_mean_eer"] for row in main_rows])),
        "baseline_mean_eer_clean_and_noise": float(np.mean([
            row["baseline_eer"] for row in main_rows
            if row["condition"] in config["selection_conditions"]
        ])),
        "baseline_mean_eer_all_conditions": float(np.mean([row["baseline_eer"] for row in main_rows])),
        "bootstrap_replicates": config["uncertainty"]["replicates"],
        "evaluation_split_accessed": False,
    }
    write_json_atomic(metrics_dir / "noise-augmentation-summary.json", aggregate)
    write_json_atomic(Path(config["paths"]["output_dir"]) / "summary.json", aggregate)
    print(json.dumps(aggregate, indent=2))


def preflight(config, roots):
    manifest = Path(config["paths"]["train_manifest"])
    records = load_development_manifest(manifest, roots.data)
    if any(record.split != "train" for record in records):
        raise PermissionError("preflight accepts TRAIN only")
    checks = []
    for seed in config["experiment"]["final_seeds"]:
        decisions = [
            noise_augmentation_decision(
                record.utterance_id, seed, config["augmentation"]["probability"],
                config["augmentation"]["snr_db_choices"],
            )
            for record in records
        ]
        dataset = NoiseAugmentedTrainingDataset(
            records, seed, config["augmentation"]["probability"],
            config["augmentation"]["snr_db_choices"], config["preprocessing"]["sample_rate"],
            config["preprocessing"]["max_duration_seconds"],
            config["preprocessing"]["crop_strategy"],
        )
        waveform_checks = []
        for target in [None, *config["augmentation"]["snr_db_choices"]]:
            index = next(i for i, decision in enumerate(decisions) if decision.snr_db == target)
            item = dataset[index]
            measured = float(item["measured_snr_db"])
            if target is not None and abs(measured - float(target)) > 1e-5:
                raise RuntimeError("training augmentation SNR preflight failed")
            waveform_checks.append({
                "utterance_id": records[index].utterance_id,
                "target_snr_db": target,
                "measured_snr_db": measured if target is not None else None,
                "finite": bool(torch.isfinite(item["waveform"]).all()),
            })
        checks.append({
            "seed": seed, "count": len(decisions),
            "augmented": sum(decision.applied for decision in decisions),
            "clean": sum(not decision.applied for decision in decisions),
            "snr_counts": {
                str(int(snr)): sum(decision.snr_db == snr for decision in decisions)
                for snr in config["augmentation"]["snr_db_choices"]
            },
            "waveform_checks": waveform_checks,
        })
    output = Path(config["paths"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "preflight.json", {
        "checks": checks, "train_only": True, "evaluation_split_accessed": False,
    })
    print(json.dumps(checks, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("preflight", "train", "analyze"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    roots = ArtifactRoots.from_environment()
    for name, value in {"AUDIO_DATA_ROOT": roots.data, "AUDIO_MODEL_ROOT": roots.models,
                        "AUDIO_CACHE_ROOT": roots.cache, "AUDIO_OUTPUT_ROOT": roots.outputs}.items():
        os.environ.setdefault(name, str(value))
    config = load_config(CONFIG_PATH)
    if sha256(Path(config["paths"]["baseline_checkpoint"])) != config["paths"]["baseline_checkpoint_sha256"]:
        raise ValueError("frozen baseline checkpoint hash mismatch")
    if args.phase == "preflight":
        preflight(config, roots)
    elif args.phase == "train":
        if args.seed not in config["experiment"]["final_seeds"]:
            raise ValueError("seed is not predefined")
        train_seed(config, roots, args.seed, torch.device("cuda:0"))
    else:
        analyze(config, roots)


if __name__ == "__main__":
    main()
