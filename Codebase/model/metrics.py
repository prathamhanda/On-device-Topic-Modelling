"""Reusable evaluation utilities for token-level keyword extraction metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from model.config import IGNORE_INDEX


def flatten_token_arrays(preds: Any, labels: Any, keyword_probs: Any | None = None) -> dict[str, np.ndarray]:
    """Flatten token predictions/labels and drop ignored positions.

    Args:
        preds: Nested list/array with predicted label IDs.
        labels: Nested list/array with gold label IDs. Positions with -100 are ignored.
        keyword_probs: Optional nested list/array with probability/logit-derived score for label 1.

    Returns:
        Dict with flat ``preds``, ``labels``, and optionally ``keyword_probs`` arrays.
    """
    pred_arr = np.asarray(preds)
    label_arr = np.asarray(labels)
    if pred_arr.shape != label_arr.shape:
        raise ValueError(f"preds shape {pred_arr.shape} != labels shape {label_arr.shape}")

    mask = label_arr != IGNORE_INDEX
    flat = {
        "preds": pred_arr[mask].astype(int),
        "labels": label_arr[mask].astype(int),
    }
    if keyword_probs is not None:
        prob_arr = np.asarray(keyword_probs)
        if prob_arr.shape != label_arr.shape:
            raise ValueError(
                f"keyword_probs shape {prob_arr.shape} != labels shape {label_arr.shape}"
            )
        flat["keyword_probs"] = prob_arr[mask].astype(float)
    return flat


def compute_token_metrics(preds: Any, labels: Any) -> dict[str, Any]:
    """Compute detailed binary token-classification metrics.

    Metrics are computed after ignored labels have already been removed.  Label 1
    is the keyword class; label 0 is the non-keyword class.
    """
    y_pred = np.asarray(preds).astype(int)
    y_true = np.asarray(labels).astype(int)

    if y_true.size == 0:
        raise ValueError("No labelled tokens found. Check preprocessing and labels.")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    keyword_precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    keyword_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    keyword_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    non_keyword_f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    negative_predictive_value = tn / (tn + fn) if (tn + fn) else 0.0
    keyword_iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

    try:
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
    except ValueError:
        balanced_acc = 0.0
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except ValueError:
        mcc = 0.0

    total = int(y_true.size)
    support_non_keyword = int(per_class_support[0])
    support_keyword = int(per_class_support[1])

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_acc),
        "precision": float(keyword_precision),
        "recall": float(keyword_recall),
        "f1": float(keyword_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "non_keyword_precision": float(per_class_precision[0]),
        "non_keyword_recall": float(per_class_recall[0]),
        "non_keyword_f1": float(non_keyword_f1),
        "keyword_precision": float(per_class_precision[1]),
        "keyword_recall": float(per_class_recall[1]),
        "keyword_f1": float(per_class_f1[1]),
        "specificity": float(specificity),
        "negative_predictive_value": float(negative_predictive_value),
        "keyword_iou": float(keyword_iou),
        "matthews_corrcoef": float(mcc),
        "support": {
            "total_labelled_tokens": total,
            "non_keyword": support_non_keyword,
            "keyword": support_keyword,
            "keyword_ratio": float(support_keyword / total),
        },
        "confusion_matrix": {
            "labels": ["non_keyword", "keyword"],
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.astype(int).tolist(),
        },
    }


def tune_keyword_threshold(
    keyword_probs: Any,
    labels: Any,
    thresholds: np.ndarray | None = None,
    optimize_for: str = "f1",
) -> dict[str, Any]:
    """Find the best probability threshold for the KEYWORD class on validation data."""
    probs = np.asarray(keyword_probs, dtype=float)
    y_true = np.asarray(labels, dtype=int)
    if probs.shape != y_true.shape:
        raise ValueError(f"keyword_probs shape {probs.shape} != labels shape {y_true.shape}")

    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)

    best: dict[str, Any] | None = None
    for threshold in thresholds:
        y_pred = (probs >= threshold).astype(int)
        metrics = compute_token_metrics(y_pred, y_true)
        score = metrics.get(optimize_for)
        if score is None:
            raise ValueError(f"Metric '{optimize_for}' is not available for threshold tuning")
        candidate = {
            "threshold": float(threshold),
            "optimized_metric": optimize_for,
            "optimized_score": float(score),
            "metrics": metrics,
        }
        if best is None or candidate["optimized_score"] > best["optimized_score"]:
            best = candidate
    assert best is not None
    return best


def parameter_summary(model: torch.nn.Module) -> dict[str, Any]:
    """Count model parameters and estimate full-precision memory."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Estimate from actual tensor dtype where possible instead of assuming float32.
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "frozen_parameters": int(total_params - trainable_params),
        "parameter_memory_mb": round(parameter_bytes / (1024 * 1024), 2),
        "buffer_memory_mb": round(buffer_bytes / (1024 * 1024), 2),
        "total_tensor_memory_mb": round((parameter_bytes + buffer_bytes) / (1024 * 1024), 2),
    }


def directory_size_mb(path: Path) -> float:
    """Return recursive directory/file size in MiB."""
    path = Path(path)
    if not path.exists():
        return 0.0
    if path.is_file():
        return round(path.stat().st_size / (1024 * 1024), 2)
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return round(total / (1024 * 1024), 2)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save JSON with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON dictionary."""
    return json.loads(Path(path).read_text())
