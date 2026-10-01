# File Guide - SMS Keyword Extractor

This guide explains every important file in the improved project.

## Root files

### `requirements.txt`
Lists the Python dependencies needed for training, evaluation, conversion, and inference.

### `README.md`
Main setup and usage documentation. It now includes the multi-profile training/evaluation workflow.

### `MODEL_TRADEOFF_REPORT.md`
Detailed trade-off report comparing the original MobileBERT baseline with the larger DistilBERT and BERT-base implementations.

### `DESIGN_AND_ACCURACY.md`
Original design/performance notes, with an updated note pointing to the new multi-model comparison workflow.

## `data/` - data pipeline

### `data/download_data.py`
Downloads the UCI SMS Spam Collection dataset and saves it as `data/sms_data.csv` with columns `label` and `text`.

Run:

```bash
python -m data.download_data
```

### `data/preprocess.py`
Cleans SMS text, creates RAKE pseudo-labels, tokenizes with the selected profile tokenizer, aligns labels to first sub-tokens, and saves a processed pickle.

Run one command per profile you want to train:

```bash
python -m data.preprocess --profile mobilebert_baseline
python -m data.preprocess --profile distilbert_accuracy
python -m data.preprocess --profile bert_base_high_accuracy
```

Outputs:

```text
data/processed.pkl
data/processed_distilbert_accuracy.pkl
data/processed_bert_base_high_accuracy.pkl
```

## `model/` - training, metrics, and comparison

### `model/config.py`
Central configuration file. It defines labels, paths, and the three model profiles:

| Profile | Model | Purpose |
|---|---|---|
| `mobilebert_baseline` | `google/mobilebert-uncased` | Original compact mobile-first implementation |
| `distilbert_accuracy` | `distilbert-base-uncased` | Larger balanced model for better accuracy |
| `bert_base_high_accuracy` | `bert-base-uncased` | Highest built-in accuracy target |

### `model/dataset.py`
PyTorch `Dataset` class that loads the processed pickle and returns tensors for `input_ids`, `attention_mask`, and `labels`.

### `model/model.py`
Builds a HuggingFace `AutoModelForTokenClassification`. This keeps MobileBERT support while enabling larger Transformer models through the same training code.

### `model/metrics.py`
Reusable metric utilities. Computes:

- Accuracy and balanced accuracy
- Keyword precision, recall, F1, and IoU
- Macro and weighted precision/recall/F1
- Specificity and negative predictive value
- Matthews correlation coefficient
- Confusion matrix
- Parameter counts and checkpoint size
- Validation threshold tuning

### `model/train.py`
Profile-aware training loop. It supports class-weighted loss, gradient accumulation, early stopping, validation-threshold tuning, and detailed metrics JSON output.

Run:

```bash
python -m model.train --profile mobilebert_baseline
python -m model.train --profile distilbert_accuracy
python -m model.train --profile bert_base_high_accuracy
```

### `model/evaluate.py`
Standalone checkpoint evaluator. Use this to find the current implementation's accuracy and other parameters.

Run:

```bash
python -m model.evaluate --profile mobilebert_baseline
```

### `model/compare_models.py`
Reads or generates metrics for all implemented profiles and writes comparison outputs.

Run:

```bash
python -m model.compare_models --evaluate-missing
```

Outputs:

```text
outputs/model_comparison_val.csv
outputs/model_comparison_val.md
```

## `convert/` - TFLite conversion

### `convert/convert_to_tflite.py`
Converts a trained PyTorch checkpoint into float32 and dynamic-range quantized TFLite models. It now accepts `--profile`.

Run:

```bash
python -m convert.convert_to_tflite --profile mobilebert_baseline
```

### `convert/validate_tflite.py`
Compares PyTorch and TFLite predictions on sample SMS messages for a selected profile.

Run:

```bash
python -m convert.validate_tflite --profile mobilebert_baseline
```

## `inference/` - user-facing prediction

### `inference/predict.py`
Loads a quantized TFLite model and extracts keywords from a single SMS or interactive prompt. It supports all profiles and automatically uses the saved validation threshold when available.

Run:

```bash
python -m inference.predict --profile mobilebert_baseline --text "Your OTP is 482910. Valid for 10 minutes."
python -m inference.predict --interactive
```

## `outputs/` - generated at runtime

Created automatically when scripts run. It stores checkpoints, metrics, thresholds, TFLite files, and model comparison reports.
