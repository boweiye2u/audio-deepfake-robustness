from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_curve
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from audio_deepfake.baseline import (
    AudioBatchDataset,
    AudioRecord,
    BinaryClassificationHead,
    load_development_manifest,
    load_head_checkpoint,
    save_head_checkpoint,
    write_records_csv,
)
from audio_deepfake.config import ArtifactRoots, load_config
from audio_deepfake.metrics import binary_metrics, select_balanced_accuracy_threshold
from audio_deepfake.utils import set_seed, write_json_atomic

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "baseline.yaml"


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


def balanced_subset(records: list[AudioRecord], size: int) -> list[AudioRecord]:
    if size >= len(records):
        return records
    class_size = size // 2
    negative = [record for record in records if record.label == 0][:class_size]
    positive = [record for record in records if record.label == 1][: size - class_size]
    if len(negative) != class_size or len(positive) != size - class_size:
        raise ValueError("requested balanced subset exceeds an available class")
    return sorted(negative + positive, key=lambda record: record.utterance_id)


def load_encoder(config: dict[str, Any], roots: ArtifactRoots, device: torch.device):
    from transformers import WavLMModel

    model = WavLMModel.from_pretrained(
        config["model"]["name"],
        revision=config["model"]["revision"],
        cache_dir=roots.cache / "huggingface",
    )
    model.eval().requires_grad_(False).to(device)
    if model.config.hidden_size != config["model"]["hidden_size"]:
        raise ValueError("configured hidden size does not match pretrained encoder")
    return model


def cache_signature(
    records: list[AudioRecord], config: dict[str, Any], manifest_path: Path
) -> dict[str, Any]:
    return {
        "model": config["model"]["name"],
        "revision": config["model"]["revision"],
        "manifest_sha256": sha256(manifest_path),
        "record_ids_sha256": hashlib.sha256(
            "\n".join(record.utterance_id for record in records).encode()
        ).hexdigest(),
        "sample_rate": config["preprocessing"]["sample_rate"],
        "max_duration_seconds": config["preprocessing"]["max_duration_seconds"],
        "crop_strategy": config["preprocessing"]["crop_strategy"],
        "count": len(records),
    }


def extract_embeddings(
    records: list[AudioRecord],
    manifest_path: Path,
    cache_path: Path,
    config: dict[str, Any],
    roots: ArtifactRoots,
    device: torch.device,
) -> tuple[dict[str, Any], float, int]:
    signature = cache_signature(records, config, manifest_path)
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload["signature"] != signature:
            raise ValueError(f"feature cache signature mismatch: {cache_path}")
        return payload, 0.0, 0

    model = load_encoder(config, roots, device)
    dataset = AudioBatchDataset(
        records,
        sample_rate=config["preprocessing"]["sample_rate"],
        max_duration_seconds=config["preprocessing"]["max_duration_seconds"],
        crop_strategy=config["preprocessing"]["crop_strategy"],
    )
    loader = DataLoader(
        dataset,
        batch_size=config["feature_extraction"]["batch_size"],
        shuffle=False,
        num_workers=config["feature_extraction"]["num_workers"],
        pin_memory=True,
        persistent_workers=config["feature_extraction"]["num_workers"] > 0,
    )
    embeddings: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"encode {cache_path.stem}"):
            waveform = batch["waveform"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = model(waveform, attention_mask=attention_mask).last_hidden_state
                feature_mask = model._get_feature_vector_attention_mask(
                    hidden.shape[1], attention_mask
                )
                weights = feature_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            embeddings.append(pooled.float().cpu())
            indices.append(batch["index"])
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device)
    order = torch.cat(indices).numpy()
    if not np.array_equal(order, np.arange(len(records))):
        raise RuntimeError("feature extraction changed record order")
    payload = {
        "signature": signature,
        "encoder_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "embeddings": torch.cat(embeddings),
        "labels": torch.tensor([record.label for record in records], dtype=torch.float32),
        "utterance_ids": [record.utterance_id for record in records],
        "speaker_ids": [record.speaker_id for record in records],
        "attack_ids": [record.attack_id for record in records],
        "duration_seconds": torch.tensor(
            [record.duration_seconds for record in records], dtype=torch.float32
        ),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(cache_path)
    del model
    torch.cuda.empty_cache()
    return payload, elapsed, peak


def predict(
    model: BinaryClassificationHead, embeddings: torch.Tensor, batch_size: int, device: torch.device
) -> np.ndarray:
    model.eval()
    scores: list[torch.Tensor] = []
    loader = DataLoader(TensorDataset(embeddings), batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for (features,) in loader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(features.to(device, non_blocking=True))
            scores.append(torch.sigmoid(logits).float().cpu())
    return torch.cat(scores).numpy()


def run_head_training(
    stage: str,
    train_payload: dict[str, Any],
    dev_payload: dict[str, Any],
    config: dict[str, Any],
    checkpoint_dir: Path,
    output_dir: Path,
    device: torch.device,
    epochs: int,
) -> dict[str, Any]:
    set_seed(config["experiment"]["seed"])
    model = BinaryClassificationHead(
        config["model"]["hidden_size"], config["model"]["dropout"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    train_labels = train_payload["labels"]
    negative = int((train_labels == 0).sum())
    positive = int((train_labels == 1).sum())
    pos_weight = torch.tensor([negative / positive], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    generator = torch.Generator().manual_seed(config["experiment"]["seed"])
    loader = DataLoader(
        TensorDataset(train_payload["embeddings"], train_labels),
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        generator=generator,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "epochs.jsonl"
    if log_path.exists():
        raise FileExistsError(f"refusing to overwrite run: {log_path}")

    history: list[dict[str, Any]] = []
    best_key: tuple[float, float, int] | None = None
    best_path: Path | None = None
    stale = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(features)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            count += len(labels)
        scores = predict(
            model, dev_payload["embeddings"], config["training"]["batch_size"], device
        )
        metrics = binary_metrics(dev_payload["labels"].numpy(), scores, threshold=0.5)
        record = {"epoch": epoch, "train_loss": total_loss / count, **metrics}
        history.append(record)
        with log_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        path = checkpoint_dir / f"epoch-{epoch}.pt"
        save_head_checkpoint(
            path,
            model,
            optimizer,
            epoch,
            {"stage": stage, "metrics": metrics, "config_sha256": sha256(CONFIG_PATH)},
        )
        key = (float(metrics["eer"]), -float(metrics["pr_auc"]), epoch)
        if best_key is None or key < best_key:
            best_key = key
            best_path = path
            stale = 0
        else:
            stale += 1
        print(
            f"{stage} epoch={epoch} loss={record['train_loss']:.6f} "
            f"dev_eer={metrics['eer']:.6f} dev_pr_auc={metrics['pr_auc']:.6f}",
            flush=True,
        )
        if stage == "full" and stale >= config["training"]["early_stopping_patience"]:
            break
    elapsed = time.perf_counter() - started
    if best_path is None:
        raise RuntimeError("training produced no checkpoint")
    selected = checkpoint_dir / "selected.pt"
    shutil.copy2(best_path, selected)
    selected_model, payload = load_head_checkpoint(
        selected,
        config["model"]["hidden_size"],
        config["model"]["dropout"],
        device,
    )
    scores = predict(
        selected_model, dev_payload["embeddings"], config["training"]["batch_size"], device
    )
    threshold = select_balanced_accuracy_threshold(dev_payload["labels"].numpy(), scores)
    metrics = binary_metrics(dev_payload["labels"].numpy(), scores, threshold)
    return {
        "stage": stage,
        "selected_epoch": payload["epoch"],
        "selected_checkpoint": str(selected),
        "selection_rule": config["training"]["checkpoint_selection"],
        "threshold_rule": config["training"]["threshold_selection"],
        "threshold": threshold,
        "metrics": metrics,
        "history": history,
        "training_seconds": elapsed,
        "peak_gpu_bytes_training": torch.cuda.max_memory_allocated(device),
        "train_class_counts": {"bonafide": negative, "spoof": positive},
        "pos_weight": float(pos_weight),
        "scores": scores,
    }


def write_analysis(
    result: dict[str, Any], dev_payload: dict[str, Any], config: dict[str, Any], output_dir: Path
) -> None:
    import matplotlib.pyplot as plt

    labels = dev_payload["labels"].numpy().astype(int)
    scores = result.pop("scores")
    threshold = result["threshold"]
    predictions = (scores >= threshold).astype(int)
    rows = []
    for index, utterance_id in enumerate(dev_payload["utterance_ids"]):
        rows.append(
            {
                "utterance_id": utterance_id,
                "speaker_id": dev_payload["speaker_ids"][index],
                "attack_id": dev_payload["attack_ids"][index],
                "duration_seconds": float(dev_payload["duration_seconds"][index]),
                "label": int(labels[index]),
                "score": float(scores[index]),
                "prediction": int(predictions[index]),
                "correct": int(predictions[index] == labels[index]),
                "confidence_margin": float(abs(scores[index] - threshold)),
            }
        )
    write_records_csv(output_dir / "dev-predictions.csv", rows)

    repository_metrics = REPOSITORY_ROOT / "results" / "metrics"
    repository_figures = REPOSITORY_ROOT / "results" / "figures"
    repository_metrics.mkdir(parents=True, exist_ok=True)
    repository_figures.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in result.items() if key != "history"}
    checkpoint = Path(summary["selected_checkpoint"])
    try:
        summary["selected_checkpoint"] = "~/" + checkpoint.relative_to(Path.home()).as_posix()
    except ValueError:
        summary["selected_checkpoint"] = checkpoint.name
    summary.update(
        {
            "experiment_id": config["experiment"]["id"],
            "positive_label": "spoof",
            "model": config["model"],
            "train_count": result["train_class_counts"]["bonafide"]
            + result["train_class_counts"]["spoof"],
            "dev_count": len(labels),
        }
    )
    write_json_atomic(repository_metrics / "baseline-dev-summary.json", summary)

    attack_rows = []
    for attack in [f"A{number:02d}" for number in range(1, 7)]:
        mask = np.array([value in {"-", attack} for value in dev_payload["attack_ids"]])
        metrics = binary_metrics(labels[mask], scores[mask], threshold)
        attack_rows.append({"attack_id": attack, "spoof_trials": int(labels[mask].sum()), **metrics})
    write_records_csv(repository_metrics / "baseline-dev-by-attack.csv", attack_rows)

    duration_rows = []
    durations = dev_payload["duration_seconds"].numpy()
    intervals = [(0, 2), (2, 3), (3, 4), (4, float("inf"))]
    for lower, upper in intervals:
        mask = (durations > lower) & (durations <= upper)
        if mask.sum() and len(np.unique(labels[mask])) == 2:
            metrics = binary_metrics(labels[mask], scores[mask], threshold)
            duration_rows.append(
                {"duration_bin": f"({lower},{upper}]", "trials": int(mask.sum()), **metrics}
            )
    write_records_csv(repository_metrics / "baseline-dev-by-duration.csv", duration_rows)

    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={result['metrics']['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(repository_figures / "baseline-dev-roc.png", dpi=160)
    plt.close()

    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"AP={result['metrics']['pr_auc']:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(repository_figures / "baseline-dev-pr.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(scores[labels == 0], bins=50, alpha=0.65, density=True, label="bonafide")
    plt.hist(scores[labels == 1], bins=50, alpha=0.65, density=True, label="spoof")
    plt.axvline(threshold, color="black", linestyle="--", label="DEV threshold")
    plt.xlabel("Spoof probability")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(repository_figures / "baseline-dev-score-distribution.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("smoke", "subset", "full"))
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
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 3 requires one CUDA GPU")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "0") not in {"0", ""}:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES=0 for the frozen single-GPU run")
    set_seed(config["experiment"]["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    train_manifest = Path(config["paths"]["train_manifest"])
    dev_manifest = Path(config["paths"]["dev_manifest"])
    train_records = load_development_manifest(train_manifest, roots.data)
    dev_records = load_development_manifest(dev_manifest, roots.data)
    if args.phase == "smoke":
        train_records = balanced_subset(train_records, config["stages"]["smoke_train_samples"])
        dev_records = balanced_subset(dev_records, config["stages"]["smoke_dev_samples"])
        epochs = 1
    elif args.phase == "subset":
        train_records = balanced_subset(train_records, config["stages"]["subset_train_samples"])
        dev_records = balanced_subset(dev_records, config["stages"]["subset_dev_samples"])
        epochs = config["stages"]["subset_epochs"]
    else:
        epochs = config["training"]["epochs"]

    cache_root = Path(config["paths"]["feature_cache"]) / args.phase
    output_dir = Path(config["paths"]["output_dir"]) / args.phase
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"]) / args.phase
    train_payload, train_extract_seconds, train_peak = extract_embeddings(
        train_records, train_manifest, cache_root / "train.pt", config, roots, device
    )
    dev_payload, dev_extract_seconds, dev_peak = extract_embeddings(
        dev_records, dev_manifest, cache_root / "dev.pt", config, roots, device
    )
    result = run_head_training(
        args.phase,
        train_payload,
        dev_payload,
        config,
        checkpoint_dir,
        output_dir,
        device,
        epochs,
    )
    result.update(
        {
            "experiment_id": config["experiment"]["id"],
            "git_commit": git_commit(),
            "config_sha256": sha256(CONFIG_PATH),
            "seed": config["experiment"]["seed"],
            "model_name": config["model"]["name"],
            "model_revision": config["model"]["revision"],
            "train_samples": len(train_records),
            "dev_samples": len(dev_records),
            "feature_extraction_seconds": train_extract_seconds + dev_extract_seconds,
            "peak_gpu_bytes_feature_extraction": max(train_peak, dev_peak),
            "gpu": torch.cuda.get_device_name(device),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in BinaryClassificationHead(
                    config["model"]["hidden_size"], config["model"]["dropout"]
                ).parameters()
            ),
            "encoder_parameters": train_payload["encoder_parameters"],
            "config": config,
        }
    )
    if args.phase == "full":
        write_analysis(result, dev_payload, config, output_dir)
    else:
        result.pop("scores")
    write_json_atomic(output_dir / "summary.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"history", "config"}}, indent=2, default=str))


if __name__ == "__main__":
    main()
