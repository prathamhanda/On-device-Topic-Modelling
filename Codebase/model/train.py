"""Fine-tune Transformer token classifiers for SMS keyword extraction.

Examples:
    python -m model.train
    python -m model.train --profile distilbert_accuracy
    python -m model.train --profile bert_base_high_accuracy
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.config import (
    DEFAULT_PROFILE,
    IGNORE_INDEX,
    MODEL_PROFILES,
    NUM_LABELS,
    SEED,
    checkpoint_path,
    get_profile,
    metrics_path,
    processed_data_path,
    threshold_path,
)
from model.dataset import SMSKeywordDataset
from model.metrics import (
    compute_token_metrics,
    flatten_token_arrays,
    parameter_summary,
    save_json,
    tune_keyword_threshold,
)
from model.model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(all_preds: list, all_labels: list) -> dict[str, Any]:
    """Backward-compatible metric helper used by the original training loop."""
    flat = flatten_token_arrays(all_preds, all_labels)
    return compute_token_metrics(flat["preds"], flat["labels"])


def split_dataset(dataset: SMSKeywordDataset, val_ratio: float, seed: int):
    """Create a deterministic train/validation split."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    val_size = max(1, int(round(val_ratio * len(dataset))))
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise ValueError("Dataset is too small to create a non-empty train split")
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def compute_class_weights(dataset_subset) -> list[float]:
    """Compute inverse-frequency class weights for labelled token positions."""
    counts = np.zeros(NUM_LABELS, dtype=np.int64)
    for i in range(len(dataset_subset)):
        labels = dataset_subset[i]["labels"].view(-1).tolist()
        for label in labels:
            if label != IGNORE_INDEX and 0 <= label < NUM_LABELS:
                counts[label] += 1

    total = int(counts.sum())
    if total == 0:
        return [1.0 for _ in range(NUM_LABELS)]

    weights = []
    for count in counts:
        if count == 0:
            weights.append(1.0)
        else:
            weights.append(float(total / (NUM_LABELS * count)))
    return weights


def collect_eval_outputs(model, data_loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    """Run validation and collect flat labels, argmax preds, and keyword probabilities."""
    model.eval()
    all_preds: list[list[int]] = []
    all_labels: list[list[int]] = []
    all_keyword_probs: list[list[float]] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)[..., 1]
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_keyword_probs.extend(probs.cpu().tolist())

    return flatten_token_arrays(all_preds, all_labels, all_keyword_probs)


def evaluate_predictions(
    flat_outputs: dict[str, np.ndarray],
    tune_threshold: bool,
    fixed_threshold: float | None = None,
) -> dict[str, Any]:
    """Compute argmax and optional threshold-tuned metrics from flat validation outputs."""
    labels = flat_outputs["labels"]
    argmax_preds = flat_outputs["preds"]
    keyword_probs = flat_outputs["keyword_probs"]

    argmax_metrics = compute_token_metrics(argmax_preds, labels)
    result: dict[str, Any] = {
        "decision_rule": "argmax",
        "decision_threshold": None,
        "metrics": argmax_metrics,
        "argmax_metrics": argmax_metrics,
    }

    if tune_threshold:
        tuned = tune_keyword_threshold(keyword_probs, labels, optimize_for="f1")
        result.update(
            {
                "decision_rule": "keyword_probability_threshold",
                "decision_threshold": tuned["threshold"],
                "metrics": tuned["metrics"],
                "threshold_tuning": tuned,
            }
        )
    elif fixed_threshold is not None:
        threshold_preds = (keyword_probs >= fixed_threshold).astype(int)
        result.update(
            {
                "decision_rule": "keyword_probability_threshold",
                "decision_threshold": float(fixed_threshold),
                "metrics": compute_token_metrics(threshold_preds, labels),
            }
        )

    return result


def train_model(
    profile_name: str = DEFAULT_PROFILE,
    data_path: Path | None = None,
    checkpoint_dir: Path | None = None,
    metrics_out: Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    val_ratio: float = 0.1,
    seed: int = SEED,
    force_cpu: bool = False,
    use_class_weights: bool | None = None,
    tune_threshold: bool = True,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Train a profile and save best checkpoint plus detailed validation metrics."""
    set_seed(seed)
    profile = get_profile(
        profile_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    resolved_data_path = Path(data_path or processed_data_path(profile_name))
    resolved_checkpoint_dir = Path(checkpoint_dir or checkpoint_path(profile_name))
    resolved_metrics_out = Path(metrics_out or metrics_path(profile_name, split="val"))
    resolved_threshold_path = threshold_path(profile_name)
    use_weights = profile.use_class_weights if use_class_weights is None else use_class_weights

    if not resolved_data_path.exists():
        raise FileNotFoundError(
            f"Missing {resolved_data_path}. Run `python -m data.preprocess --profile {profile_name}` first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    print(f"[INFO] Profile     : {profile_name}")
    print(f"[INFO] Model       : {profile.model_name}")
    print(f"[INFO] Data        : {resolved_data_path}")
    print(f"[INFO] Checkpoint  : {resolved_checkpoint_dir}")
    print(f"[INFO] Metrics     : {resolved_metrics_out}")
    print(f"[INFO] Device      : {device}")

    dataset = SMSKeywordDataset(resolved_data_path)
    train_ds, val_ds = split_dataset(dataset, val_ratio=val_ratio, seed=seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=profile.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=profile.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_model(profile.model_name, NUM_LABELS).to(device)
    param_info = parameter_summary(model)
    print(
        "[INFO] Parameters  : "
        f"{param_info['total_parameters']:,} total | "
        f"{param_info['trainable_parameters']:,} trainable | "
        f"{param_info['total_tensor_memory_mb']:.2f} MiB tensors"
    )

    class_weights_list = compute_class_weights(train_ds) if use_weights else None
    class_weights_tensor = None
    if class_weights_list is not None:
        class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float, device=device)
        print(f"[INFO] Class weights: {class_weights_list}")

    optimizer = AdamW(model.parameters(), lr=profile.learning_rate, weight_decay=profile.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / profile.gradient_accumulation_steps)
    total_steps = max(1, steps_per_epoch * profile.epochs)
    warmup_steps = int(profile.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = CrossEntropyLoss(weight=class_weights_tensor, ignore_index=IGNORE_INDEX)

    best_score = -1.0
    best_report: dict[str, Any] | None = None
    epochs_without_improvement = 0
    epoch_history: list[dict[str, Any]] = []

    resolved_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, profile.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer_steps = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{profile.epochs} [train]")

        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if class_weights_tensor is None:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits.reshape(-1, NUM_LABELS), labels.reshape(-1))

            (loss / profile.gradient_accumulation_steps).backward()
            total_loss += float(loss.item())

            should_step = (
                step % profile.gradient_accumulation_steps == 0 or step == len(train_loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(1, len(train_loader))
        flat_outputs = collect_eval_outputs(model, val_loader, device)
        eval_result = evaluate_predictions(flat_outputs, tune_threshold=tune_threshold)
        metrics = eval_result["metrics"]
        score = float(metrics["f1"])

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "optimizer_steps": optimizer_steps,
            "decision_rule": eval_result["decision_rule"],
            "decision_threshold": eval_result["decision_threshold"],
            "metrics": metrics,
            "argmax_metrics": eval_result["argmax_metrics"],
        }
        epoch_history.append(epoch_record)

        print(
            f"Epoch {epoch}: loss={avg_loss:.4f} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"Threshold={eval_result['decision_threshold'] if eval_result['decision_threshold'] is not None else 'argmax'}"
        )

        if score > best_score + 1e-8:
            best_score = score
            epochs_without_improvement = 0
            model.save_pretrained(resolved_checkpoint_dir)
            threshold_payload = {
                "profile": profile_name,
                "model_name": profile.model_name,
                "decision_rule": eval_result["decision_rule"],
                "decision_threshold": eval_result["decision_threshold"],
                "optimized_metric": "f1",
                "epoch": epoch,
            }
            save_json(threshold_payload, resolved_threshold_path)
            best_report = {
                "profile": asdict(profile),
                "profile_name": profile_name,
                "model_name": profile.model_name,
                "seed": seed,
                "device": str(device),
                "data_path": str(resolved_data_path),
                "checkpoint_dir": str(resolved_checkpoint_dir),
                "threshold_path": str(resolved_threshold_path),
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "val_ratio": val_ratio,
                "class_weights": class_weights_list,
                "model": param_info,
                "best_epoch": epoch,
                "decision_rule": eval_result["decision_rule"],
                "decision_threshold": eval_result["decision_threshold"],
                "metrics": metrics,
                "argmax_metrics": eval_result["argmax_metrics"],
                "epoch_history": epoch_history,
            }
            save_json(best_report, resolved_metrics_out)
            print(f"  -> Saved best model (F1={best_score:.4f}) to {resolved_checkpoint_dir}")
        else:
            epochs_without_improvement += 1
            print(
                f"  -> No improvement for {epochs_without_improvement}/"
                f"{profile.early_stopping_patience} epochs"
            )
            if epochs_without_improvement >= profile.early_stopping_patience:
                print("[INFO] Early stopping triggered")
                break

    if best_report is None:
        raise RuntimeError("Training finished without producing a validation report")

    best_report["epoch_history"] = epoch_history
    save_json(best_report, resolved_metrics_out)
    print(f"\n[DONE] Training complete. Best validation F1: {best_score:.4f}")
    print(f"[DONE] Metrics saved to {resolved_metrics_out}")
    return best_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SMS keyword extraction model")
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        choices=sorted(MODEL_PROFILES),
        help="Model profile to train",
    )
    parser.add_argument("--data-path", type=Path, default=None, help="Processed pickle path")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Where to save best model")
    parser.add_argument("--metrics-out", type=Path, default=None, help="Where to save metrics JSON")
    parser.add_argument("--epochs", type=int, default=None, help="Override profile epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override profile batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override profile learning rate")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--force-cpu", action="store_true", help="Disable CUDA even when available")
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Use the model's default unweighted cross-entropy loss",
    )
    parser.add_argument(
        "--no-threshold-tuning",
        action="store_true",
        help="Use argmax labels instead of tuning keyword probability threshold on validation",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        profile_name=args.profile,
        data_path=args.data_path,
        checkpoint_dir=args.checkpoint_dir,
        metrics_out=args.metrics_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_ratio=args.val_ratio,
        seed=args.seed,
        force_cpu=args.force_cpu,
        use_class_weights=not args.no_class_weights,
        tune_threshold=not args.no_threshold_tuning,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
