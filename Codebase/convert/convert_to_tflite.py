"""Convert a trained PyTorch checkpoint to TFLite.

The default profile preserves the original MobileBERT export.  The converter now
also accepts the larger DistilBERT/BERT profiles when TensorFlow support exists
for the selected HuggingFace architecture.

Usage:
    python -m convert.convert_to_tflite
    python -m convert.convert_to_tflite --profile distilbert_accuracy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorflow as tf
import os
from transformers import TFAutoModelForTokenClassification

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.config import (
    DEFAULT_PROFILE,
    MODEL_PROFILES,
    NUM_LABELS,
    checkpoint_path,
    get_profile,
    tflite_path,
    processed_data_path,
)


def load_tf_model(checkpoint_dir: Path) -> TFAutoModelForTokenClassification:
    """Re-load PyTorch weights into the equivalent TensorFlow token classifier."""
    print(f"[INFO] Loading PyTorch weights into TF model from {checkpoint_dir} ...")
    return TFAutoModelForTokenClassification.from_pretrained(
        str(checkpoint_dir),
        num_labels=NUM_LABELS,
        from_pt=True,
    )


def build_converter(model: TFAutoModelForTokenClassification, max_len: int, quantize: bool):
    """Build a TFLiteConverter from a fixed-shape serving function."""

    @tf.function(
        input_signature=[
            tf.TensorSpec([1, max_len], tf.int32, name="input_ids"),
            tf.TensorSpec([1, max_len], tf.int32, name="attention_mask"),
        ]
    )
    def serving_fn(input_ids, attention_mask):
        outputs = model(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            training=False,
        )
        return outputs.logits

    concrete_fn = serving_fn.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    return converter


def representative_data_generator(profile_name: str = DEFAULT_PROFILE, num_samples: int = 100):
    """Yield representative samples (input_ids, attention_mask) from the processed pickle.

    Uses the pickle produced by `data/preprocess.py` for calibration.
    """
    import pickle

    path = Path(processed_data_path(profile_name))
    if not path.exists():
        raise FileNotFoundError(f"Representative data not found at {path}. Run data.preprocess first.")

    with open(path, "rb") as f:
        samples = pickle.load(f)

    count = 0
    for s in samples:
        if count >= num_samples:
            break
        input_ids = tf.convert_to_tensor([s["input_ids"]], dtype=tf.int32)
        attention_mask = tf.convert_to_tensor([s["attention_mask"]], dtype=tf.int32)
        yield [input_ids, attention_mask]
        count += 1



def convert_profile(
    profile_name: str = DEFAULT_PROFILE,
    checkpoint_dir: Path | None = None,
    float_out: Path | None = None,
    quantized_out: Path | None = None,
    skip_float: bool = False,
    skip_quantized: bool = False,
    export_max_len: int | None = None,
) -> dict[str, Path]:
    """Convert one profile checkpoint to float32 and/or dynamic-quantized TFLite."""
    profile = get_profile(profile_name)
    resolved_checkpoint_dir = Path(checkpoint_dir or checkpoint_path(profile_name))
    if not resolved_checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Missing checkpoint {resolved_checkpoint_dir}. Run `python -m model.train --profile {profile_name}` first."
        )

    model = load_tf_model(resolved_checkpoint_dir)
    outputs: dict[str, Path] = {}
    used_max_len = export_max_len or profile.max_len

    if not skip_float:
        resolved_float_out = Path(float_out or tflite_path(profile_name, quantized=False))
        print(f"[INFO] Converting to TFLite float32 (max_len={used_max_len}) ...")
        tflite_bytes = build_converter(model, used_max_len, quantize=False).convert()
        resolved_float_out.parent.mkdir(parents=True, exist_ok=True)
        resolved_float_out.write_bytes(tflite_bytes)
        outputs["float32"] = resolved_float_out
        print(f"  -> {resolved_float_out} ({len(tflite_bytes) / 1e6:.1f} MB)")

    if not skip_quantized:
        resolved_quantized_out = Path(quantized_out or tflite_path(profile_name, quantized=True))
        print(f"[INFO] Converting to TFLite dynamic-range quantized (max_len={used_max_len}) ...")
        tflite_quant_bytes = build_converter(model, used_max_len, quantize=True).convert()
        resolved_quantized_out.parent.mkdir(parents=True, exist_ok=True)
        resolved_quantized_out.write_bytes(tflite_quant_bytes)
        outputs["dynamic_quantized"] = resolved_quantized_out
        print(f"  -> {resolved_quantized_out} ({len(tflite_quant_bytes) / 1e6:.1f} MB)")

    # Optionally produce a fully integer (int8) quantized model if requested via env var
    if os.environ.get("FULL_INT8_QUANT", "0") == "1":
        resolved_int8_out = Path(tflite_path(profile_name, quantized=True)).with_name(
            tflite_path(profile_name, quantized=True).stem + "_int8.tflite"
        )
        print(f"[INFO] Converting to TFLite full integer (int8) quantized (max_len={used_max_len}) ...")
        converter_int8 = build_converter(model, used_max_len, quantize=True)
        # Use representative dataset for calibration
        converter_int8.representative_dataset = lambda: representative_data_generator(profile_name)
        converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_int8.inference_input_type = tf.int8
        converter_int8.inference_output_type = tf.int8
        tflite_int8_bytes = converter_int8.convert()
        resolved_int8_out.parent.mkdir(parents=True, exist_ok=True)
        resolved_int8_out.write_bytes(tflite_int8_bytes)
        outputs["int8_quantized"] = resolved_int8_out
        print(f"  -> {resolved_int8_out} ({len(tflite_int8_bytes) / 1e6:.1f} MB)")

    print("\n[DONE] Conversion complete.")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SMS keyword extractor checkpoint to TFLite")
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        choices=sorted(MODEL_PROFILES),
        help="Model profile to convert",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--float-out", type=Path, default=None)
    parser.add_argument("--quantized-out", type=Path, default=None)
    parser.add_argument("--skip-float", action="store_true")
    parser.add_argument("--skip-quantized", action="store_true")
    parser.add_argument("--export-max-len", type=int, default=None, help="Override max token length for exported TFLite (smaller -> faster)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_profile(
        profile_name=args.profile,
        checkpoint_dir=args.checkpoint_dir,
        float_out=args.float_out,
        quantized_out=args.quantized_out,
        skip_float=args.skip_float,
        skip_quantized=args.skip_quantized,
        export_max_len=args.export_max_len,
    )


if __name__ == "__main__":
    main()
