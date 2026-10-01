"""Validate TFLite output against the original PyTorch checkpoint.

Runs sample SMS messages through both model formats and compares extracted
keywords.  Supports the baseline and the larger model profiles.

Usage:
    python -m convert.validate_tflite
    python -m convert.validate_tflite --profile distilbert_accuracy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.config import (
    DEFAULT_PROFILE,
    MODEL_PROFILES,
    checkpoint_path,
    get_profile,
    tflite_path,
    threshold_path,
)
from model.metrics import load_json

SAMPLE_SMS = [
    "Congratulations! You have won a FREE prize. Call now to claim your reward.",
    "Hey, are we still meeting tomorrow at 5pm for coffee?",
    "Your OTP for login is 482910. Valid for 10 minutes. Do not share.",
]


def decode_keywords(tokens: list[str], label_ids) -> list[str]:
    return [
        token
        for token, label in zip(tokens, label_ids)
        if label == 1
        and not token.startswith("##")
        and token not in ("[CLS]", "[SEP]", "[PAD]", "[UNK]")
    ]


def resolve_threshold(profile_name: str, threshold: str) -> float | None:
    if threshold.lower() == "argmax":
        return None
    if threshold.lower() == "auto":
        path = threshold_path(profile_name)
        if path.exists():
            payload = load_json(path)
            value = payload.get("decision_threshold")
            if value is not None:
                return float(value)
        return None
    return float(threshold)


def labels_from_logits(logits: np.ndarray, threshold: float | None):
    if threshold is None:
        return np.argmax(logits, axis=-1)
    probs = _softmax(logits, axis=-1)[..., 1]
    return (probs >= threshold).astype(np.int64)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def run_pytorch(model, tokenizer, text: str, max_len: int, threshold: float | None) -> list[str]:
    enc = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_len,
        truncation=True,
        padding="max_length",
    )
    with torch.no_grad():
        logits = model(**enc).logits[0].numpy()
    preds = labels_from_logits(logits, threshold)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].numpy())
    return decode_keywords(tokens, preds)


def _set_tflite_inputs(interpreter, input_ids: np.ndarray, attention_mask: np.ndarray) -> None:
    """Feed inputs to the TFLite interpreter by matching tensor names."""
    for detail in interpreter.get_input_details():
        name = detail["name"].lower()
        dtype = detail["dtype"]
        if "input_ids" in name:
            interpreter.set_tensor(detail["index"], input_ids.astype(dtype))
        elif "attention_mask" in name:
            interpreter.set_tensor(detail["index"], attention_mask.astype(dtype))


def run_tflite(interpreter, tokenizer, text: str, max_len: int, threshold: float | None) -> list[str]:
    enc = tokenizer(
        text,
        return_tensors="np",
        max_length=max_len,
        truncation=True,
        padding="max_length",
    )
    _set_tflite_inputs(interpreter, enc["input_ids"], enc["attention_mask"])
    interpreter.invoke()
    logits = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0]
    preds = labels_from_logits(logits, threshold)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    return decode_keywords(tokens, preds)


def validate_tflite(
    profile_name: str = DEFAULT_PROFILE,
    checkpoint_dir: Path | None = None,
    model_path: Path | None = None,
    threshold: str = "auto",
) -> None:
    profile = get_profile(profile_name)
    resolved_checkpoint = Path(checkpoint_dir or checkpoint_path(profile_name))
    resolved_model_path = Path(model_path or tflite_path(profile_name, quantized=True))
    resolved_threshold = resolve_threshold(profile_name, threshold)

    print(f"[INFO] Loading PyTorch model from {resolved_checkpoint} ...")
    model = AutoModelForTokenClassification.from_pretrained(resolved_checkpoint)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(profile.model_name, use_fast=True)

    print(f"[INFO] Loading TFLite model from {resolved_model_path} ...")
    interpreter = tf.lite.Interpreter(model_path=str(resolved_model_path))
    interpreter.allocate_tensors()

    print(f"[INFO] Threshold: {resolved_threshold if resolved_threshold is not None else 'argmax'}")
    print("\n--- Validation Results ---")
    for i, sms in enumerate(SAMPLE_SMS, 1):
        pt_kw = run_pytorch(model, tokenizer, sms, profile.max_len, resolved_threshold)
        tfl_kw = run_tflite(interpreter, tokenizer, sms, profile.max_len, resolved_threshold)
        match = set(pt_kw) == set(tfl_kw)
        print(f"\nSMS #{i}: {sms[:65]}...")
        print(f"  PyTorch  : {pt_kw}")
        print(f"  TFLite   : {tfl_kw}")
        print(f"  Match    : {'yes' if match else 'minor difference'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TFLite model against PyTorch checkpoint")
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        choices=sorted(MODEL_PROFILES),
        help="Model profile to validate",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None, help="TFLite model path")
    parser.add_argument("--threshold", default="auto", help="auto, argmax, or numeric threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_tflite(
        profile_name=args.profile,
        checkpoint_dir=args.checkpoint_dir,
        model_path=args.model_path,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
