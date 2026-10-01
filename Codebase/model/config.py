"""Central configuration for training, evaluation, and conversion.

The default profile keeps the original MobileBERT implementation intact.  The
additional profiles make it easy to train slightly larger, higher-capacity
models and compare their accuracy/size/latency trade-offs with the baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# Labels
NUM_LABELS = 2  # 0 = non-keyword, 1 = keyword
ID2LABEL = {0: "NON_KEYWORD", 1: "KEYWORD"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
IGNORE_INDEX = -100

# Reproducibility
SEED = 42

# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelProfile:
    """Hyperparameter bundle for a train/evaluate/export implementation."""

    name: str
    model_name: str
    max_len: int
    batch_size: int
    epochs: int
    learning_rate: float
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 2
    use_class_weights: bool = True
    description: str = ""


DEFAULT_PROFILE = "mobilebert_baseline"

MODEL_PROFILES: dict[str, ModelProfile] = {
    # Original mobile-first implementation.
    "mobilebert_baseline": ModelProfile(
        name="mobilebert_baseline",
        model_name="google/mobilebert-uncased",
        max_len=64,
        batch_size=32,
        epochs=5,
        learning_rate=2e-5,
        gradient_accumulation_steps=1,
        early_stopping_patience=2,
        description="Original compact MobileBERT baseline; best for mobile TFLite deployment.",
    ),
    # Slightly larger than MobileBERT and often stronger on token classification.
    "distilbert_accuracy": ModelProfile(
        name="distilbert_accuracy",
        model_name="distilbert-base-uncased",
        max_len=96,
        batch_size=16,
        epochs=6,
        learning_rate=3e-5,
        gradient_accumulation_steps=1,
        early_stopping_patience=2,
        description="Larger accuracy-focused model with moderate training/inference cost.",
    ),
    # Highest-capacity option in this repository.
    "bert_base_high_accuracy": ModelProfile(
        name="bert_base_high_accuracy",
        model_name="bert-base-uncased",
        max_len=96,
        batch_size=8,
        epochs=6,
        learning_rate=2e-5,
        gradient_accumulation_steps=2,
        early_stopping_patience=2,
        description="Largest built-in model; expected to give the best validation F1 when resources permit.",
    ),
}


def get_profile(profile_name: str = DEFAULT_PROFILE, **overrides: Any) -> ModelProfile:
    """Return a model profile with optional dataclass field overrides."""
    if profile_name not in MODEL_PROFILES:
        valid = ", ".join(sorted(MODEL_PROFILES))
        raise ValueError(f"Unknown profile '{profile_name}'. Valid profiles: {valid}")
    profile = MODEL_PROFILES[profile_name]
    clean_overrides = {k: v for k, v in overrides.items() if v is not None}
    return replace(profile, **clean_overrides) if clean_overrides else profile


def profile_output_dir(profile_name: str = DEFAULT_PROFILE) -> Path:
    """Output directory for a profile.

    The baseline keeps the original top-level output paths for backward
    compatibility.  Larger variants get isolated subdirectories.
    """
    if profile_name == DEFAULT_PROFILE:
        path = OUTPUT_DIR
    else:
        path = OUTPUT_DIR / profile_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_data_path(profile_name: str = DEFAULT_PROFILE) -> Path:
    """Pickle path for tokenized samples for a profile tokenizer."""
    if profile_name == DEFAULT_PROFILE:
        return DATA_DIR / "processed.pkl"
    return DATA_DIR / f"processed_{profile_name}.pkl"


def checkpoint_path(profile_name: str = DEFAULT_PROFILE) -> Path:
    """Best HuggingFace checkpoint directory for a profile."""
    return profile_output_dir(profile_name) / "best_model"


def threshold_path(profile_name: str = DEFAULT_PROFILE) -> Path:
    """Saved decision threshold for keyword probability decoding."""
    return profile_output_dir(profile_name) / "decision_threshold.json"


def metrics_path(profile_name: str = DEFAULT_PROFILE, split: str = "val") -> Path:
    """JSON evaluation metrics path for a profile/split."""
    return profile_output_dir(profile_name) / f"metrics_{split}.json"


def tflite_path(profile_name: str = DEFAULT_PROFILE, quantized: bool = False) -> Path:
    """TFLite export path for a profile."""
    suffix = "dynamic_quant" if quantized else "float32"
    if profile_name == DEFAULT_PROFILE:
        filename = (
            "sms_keyword_extractor_dynamic_quant.tflite"
            if quantized
            else "sms_keyword_extractor.tflite"
        )
    else:
        filename = f"sms_keyword_extractor_{profile_name}_{suffix}.tflite"
    return profile_output_dir(profile_name) / filename


# Backward-compatible constants used by existing scripts/imports.
_PROFILE = get_profile(DEFAULT_PROFILE)
MODEL_NAME = _PROFILE.model_name
MAX_LEN = _PROFILE.max_len
BATCH_SIZE = _PROFILE.batch_size
EPOCHS = _PROFILE.epochs
LEARNING_RATE = _PROFILE.learning_rate
WARMUP_RATIO = _PROFILE.warmup_ratio
WEIGHT_DECAY = _PROFILE.weight_decay
CHECKPOINT_PATH = checkpoint_path(DEFAULT_PROFILE)
TFLITE_PATH = tflite_path(DEFAULT_PROFILE, quantized=False)
TFLITE_QUANTIZED_PATH = tflite_path(DEFAULT_PROFILE, quantized=True)
