"""Evaluate a saved SMS keyword extraction checkpoint.

This script answers: "What is the current implementation's accuracy and other
parameters?"  It reports token-level accuracy, precision, recall, F1, macro F1,
weighted F1, specificity, MCC, confusion matrix, parameter count, and checkpoint
size.

Examples:
    python -m model.evaluate
    python -m model.evaluate --profile distilbert_accuracy
    python -m model.evaluate --profile bert_base_high_accuracy --split all
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForTokenClassification

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.config import (
    DEFAULT_PROFILE,
    MODEL_PROFILES,
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
    directory_size_mb,
    flatten_token_arrays,
    load_json,
    parameter_summary,
    save_json,
)
from model.train import split_dataset


def _resolve_threshold(profile_name: str, threshold: str) -> tuple[str, float | None]:
    """Resolve threshold CLI value into a decision rule and float threshold."""
    if threshold.lower() == "argmax":
        return "argmax", None
    if threshold.lower() == "auto":
        path = threshold_path(profile_name)
        if path.exists():
            payload = load_json(path)
            value = payload.get("decision_threshold")
            if value is not None:
                return "keyword_probability_threshold", float(value)
        return "argmax", None
    return "keyword_probability_threshold", float(threshold)


def collect_outputs(model, data_loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    """Collect flat labels, argmax predictions, and keyword probabilities."""
    model.eval()
    all_preds: list[list[int]] = []
    all_labels: list[list[int]] = []
    all_keyword_probs: list[list[float]] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
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


def evaluate_checkpoint(
    profile_name: str = DEFAULT_PROFILE,
    data_path: Path | None = None,
    checkpoint_dir: Path | None = None,
    split: str = "val",
    val_ratio: float = 0.1,
    seed: int = SEED,
    threshold: str = "auto",
    batch_size: int | None = None,
    force_cpu: bool = False,
    metrics_out: Path | None = None,
    save_metrics: bool = True,
) -> dict[str, Any]:
    """Evaluate a saved checkpoint and optionally persist metrics JSON."""
    profile = get_profile(profile_name)
    resolved_data_path = Path(data_path or processed_data_path(profile_name))
    resolved_checkpoint_dir = Path(checkpoint_dir or checkpoint_path(profile_name))
    resolved_metrics_out = Path(metrics_out or metrics_path(profile_name, split=split))

    if not resolved_data_path.exists():
        raise FileNotFoundError(
            f"Missing {resolved_data_path}. Run `python -m data.preprocess --profile {profile_name}` first."
        )
    if not resolved_checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Missing checkpoint {resolved_checkpoint_dir}. Run `python -m model.train --profile {profile_name}` first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    dataset = SMSKeywordDataset(resolved_data_path)
    train_ds, val_ds = split_dataset(dataset, val_ratio=val_ratio, seed=seed)
    if split == "train":
        eval_ds = train_ds
    elif split == "val":
        eval_ds = val_ds
    elif split == "all":
        eval_ds = dataset
    else:
        raise ValueError("split must be one of: train, val, all")

    loader = DataLoader(eval_ds, batch_size=batch_size or profile.batch_size, shuffle=False)
    model = AutoModelForTokenClassification.from_pretrained(resolved_checkpoint_dir).to(device)

    decision_rule, resolved_threshold = _resolve_threshold(profile_name, threshold)
    print(f"[INFO] Profile    : {profile_name}")
    print(f"[INFO] Model      : {profile.model_name}")
    print(f"[INFO] Checkpoint : {resolved_checkpoint_dir}")
    print(f"[INFO] Data       : {resolved_data_path}")
    print(f"[INFO] Split      : {split} ({len(eval_ds)} samples)")
    print(f"[INFO] Decision   : {decision_rule} ({resolved_threshold})")
    print(f"[INFO] Device     : {device}")

    flat = collect_outputs(model, loader, device)
    if resolved_threshold is None:
        preds = flat["preds"]
    else:
        preds = (flat["keyword_probs"] >= resolved_threshold).astype(int)

    metrics = compute_token_metrics(preds, flat["labels"])
    argmax_metrics = compute_token_metrics(flat["preds"], flat["labels"])
    param_info = parameter_summary(model)

    report = {
        "profile": asdict(profile),
        "profile_name": profile_name,
        "model_name": profile.model_name,
        "split": split,
        "seed": seed,
        "val_ratio": val_ratio,
        "data_path": str(resolved_data_path),
        "checkpoint_dir": str(resolved_checkpoint_dir),
        "checkpoint_size_mb": directory_size_mb(resolved_checkpoint_dir),
        "sample_count": len(eval_ds),
        "decision_rule": decision_rule,
        "decision_threshold": resolved_threshold,
        "metrics": metrics,
        "argmax_metrics": argmax_metrics,
        "model": param_info,
    }

    if save_metrics:
        save_json(report, resolved_metrics_out)
        print(f"[INFO] Metrics saved to {resolved_metrics_out}")

    cm = metrics["confusion_matrix"]
    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"Keyword F1        : {metrics['f1']:.4f}")
    print(f"Keyword precision : {metrics['precision']:.4f}")
    print(f"Keyword recall    : {metrics['recall']:.4f}")
    print(f"Macro F1          : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1       : {metrics['weighted_f1']:.4f}")
    print(f"MCC               : {metrics['matthews_corrcoef']:.4f}")
    print(f"Parameters        : {param_info['total_parameters']:,}")
    print(f"Checkpoint size   : {report['checkpoint_size_mb']:.2f} MiB")
    print(
        "Confusion matrix  : "
        f"TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved SMS keyword extractor checkpoint")
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        choices=sorted(MODEL_PROFILES),
        help="Model profile to evaluate",
    )
    parser.add_argument("--data-path", type=Path, default=None, help="Processed pickle path")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Checkpoint directory")
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--threshold",
        default="auto",
        help="auto, argmax, or numeric keyword-probability threshold such as 0.47",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--no-save", action="store_true", help="Print only; do not write metrics JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_checkpoint(
        profile_name=args.profile,
        data_path=args.data_path,
        checkpoint_dir=args.checkpoint_dir,
        split=args.split,
        val_ratio=args.val_ratio,
        seed=args.seed,
        threshold=args.threshold,
        batch_size=args.batch_size,
        force_cpu=args.force_cpu,
        metrics_out=args.metrics_out,
        save_metrics=not args.no_save,
    )


if __name__ == "__main__":
    main()
