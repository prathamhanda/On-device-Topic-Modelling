# Detailed Trade-off Report for All Implementations

This report documents the original implementation and the larger accuracy-focused implementations added to the project. It also explains how to measure the current accuracy and model parameters using the new evaluation scripts.

## 1. Implementations included

| Implementation | Profile name | Base model | Main goal | Relative size | Expected use case |
|---|---|---|---|---|---|
| Original baseline | `mobilebert_baseline` | `google/mobilebert-uncased` | Small, mobile-friendly keyword extraction | Smallest of the three | Android/iOS on-device inference where app size and latency matter |
| Larger balanced model | `distilbert_accuracy` | `distilbert-base-uncased` | Better accuracy with moderate compute | Larger than MobileBERT | Server inference, desktop inference, or mobile apps with more size/latency budget |
| High-accuracy model | `bert_base_high_accuracy` | `bert-base-uncased` | Highest built-in accuracy target | Largest of the three | Accuracy-first deployments where latency and model size are less restrictive |

## 2. What changed from the original code

The original project trained only MobileBERT and reported validation F1, precision, and recall. The improved code keeps that path working but adds a profile system so the same pipeline can train and compare several models.

New code modules:

| File | Purpose |
|---|---|
| `model/metrics.py` | Shared metric utilities: accuracy, balanced accuracy, F1, precision, recall, macro/weighted metrics, MCC, specificity, confusion matrix, parameter count, checkpoint size |
| `model/evaluate.py` | Standalone evaluation script for the current checkpoint |
| `model/compare_models.py` | Compares MobileBERT, DistilBERT, and BERT-base metrics into CSV and Markdown |
| `model/config.py` | Adds named model profiles and profile-specific paths |
| `model/model.py` | Replaces MobileBERT-only construction with `AutoModelForTokenClassification` |
| `data/preprocess.py` | Adds profile-aware preprocessing so each model uses its matching tokenizer |
| `convert/convert_to_tflite.py` | Adds profile support for TFLite export |
| `inference/predict.py` | Adds profile support and validation-threshold based decoding |

## 3. How to find the accuracy and parameters of the current implementation

Run the original baseline workflow:

```bash
python -m data.download_data
python -m data.preprocess --profile mobilebert_baseline
python -m model.train --profile mobilebert_baseline
python -m model.evaluate --profile mobilebert_baseline
```

The evaluation script prints and saves the current implementation metrics. The JSON output is saved to:

```text
outputs/metrics_val.json
```

The report includes:

| Metric group | Metrics included |
|---|---|
| Token accuracy | `accuracy`, `balanced_accuracy` |
| Keyword-class quality | `precision`, `recall`, `f1`, `keyword_iou` |
| Overall class quality | `macro_f1`, `weighted_f1`, per-class precision/recall/F1 |
| Error analysis | confusion matrix with `tn`, `fp`, `fn`, `tp` |
| Robust binary classification | `specificity`, `negative_predictive_value`, `matthews_corrcoef` |
| Model parameters | total parameters, trainable parameters, frozen parameters |
| Storage estimate | tensor memory and saved checkpoint size |

## 4. Accuracy-oriented improvements

### 4.1 Larger model profiles

The new profiles increase model capacity while preserving the same token-labeling objective.

| Profile | Why it may improve accuracy | Cost |
|---|---|---|
| `distilbert_accuracy` | More standard BERT-like capacity than the compact mobile-first baseline; good contextual representation for short SMS text | Larger checkpoint and slower inference than MobileBERT |
| `bert_base_high_accuracy` | Full BERT-base capacity; more layers and parameters can model subtle keyword context better | Highest memory, training time, and latency among built-in options |

### 4.2 Profile-specific tokenization

The previous preprocessing always used the MobileBERT tokenizer. That is safe for MobileBERT but not ideal for all models. The new `data.preprocess` command uses the tokenizer for the selected profile and saves a separate pickle per profile:

```text
data/processed.pkl
data/processed_distilbert_accuracy.pkl
data/processed_bert_base_high_accuracy.pkl
```

This prevents training a model on token IDs from the wrong tokenizer.

### 4.3 Detailed validation metrics

Token-level accuracy alone can be misleading because a model can get many non-keyword tokens right while missing the actual keywords. The new evaluation reports both accuracy and keyword-class metrics. For this project, `keyword_f1` is usually the most important model-selection metric because it balances missed keywords and false keyword extractions.

### 4.4 Class-weighted training

The training loop can use inverse-frequency class weights for labelled token positions. This helps when one label dominates the training set. It can be disabled with:

```bash
python -m model.train --profile mobilebert_baseline --no-class-weights
```

### 4.5 Keyword probability threshold tuning

Instead of always using `argmax`, the training loop can tune a validation threshold for the keyword probability. This is useful when the model is calibrated such that the best keyword F1 occurs at a threshold other than 0.50. The selected threshold is saved to:

```text
outputs/decision_threshold.json
outputs/distilbert_accuracy/decision_threshold.json
outputs/bert_base_high_accuracy/decision_threshold.json
```

Inference and TFLite validation use this threshold automatically when available.

Disable threshold tuning with:

```bash
python -m model.train --profile distilbert_accuracy --no-threshold-tuning
```

## 5. Trade-off table

The exact metrics must be measured on your trained checkpoints because they depend on the train/validation split, hardware, dependency versions, and pseudo-label quality. Use `python -m model.compare_models --evaluate-missing` to produce measured values.

| Dimension | `mobilebert_baseline` | `distilbert_accuracy` | `bert_base_high_accuracy` |
|---|---|---|---|
| Primary objective | Mobile deployment | Accuracy/size balance | Highest accuracy target |
| Expected keyword F1 | Good | Better than baseline in many settings | Best candidate if enough data and compute are available |
| Parameter count | Lowest | Medium | Highest |
| Training time | Fastest | Medium | Slowest |
| Inference latency | Fastest | Medium | Slowest |
| TFLite/app size | Best | Larger | Largest |
| Memory use | Lowest | Medium | Highest |
| Mobile suitability | Excellent | Possible with budget | Usually poor unless device budget is generous |
| Server suitability | Good | Very good | Very good |
| Recommended default | Yes, for mobile | Yes, for better quality | Yes, for accuracy-first experiments |

## 6. Recommended comparison workflow

Train and evaluate all profiles:

```bash
python -m data.download_data

python -m data.preprocess --profile mobilebert_baseline
python -m model.train --profile mobilebert_baseline
python -m model.evaluate --profile mobilebert_baseline

python -m data.preprocess --profile distilbert_accuracy
python -m model.train --profile distilbert_accuracy
python -m model.evaluate --profile distilbert_accuracy

python -m data.preprocess --profile bert_base_high_accuracy
python -m model.train --profile bert_base_high_accuracy
python -m model.evaluate --profile bert_base_high_accuracy

python -m model.compare_models --evaluate-missing
```

Generated comparison files:

```text
outputs/model_comparison_val.csv
outputs/model_comparison_val.md
```

## 7. How to choose an implementation

Choose `mobilebert_baseline` when the model must run fully on-device with the smallest practical footprint. It is the safest deployment option for Android/iOS.

Choose `distilbert_accuracy` when the baseline extracts too few keywords or too many false keywords, and the product can accept a larger model. This is the best first upgrade because it improves capacity without jumping all the way to BERT-base.

Choose `bert_base_high_accuracy` when the quality target matters more than deployment size and latency. It is best for offline batch evaluation, server-side keyword extraction, or experiments that define the upper bound for this architecture family.

## 8. Important limitation

The training labels are RAKE pseudo-labels, not manually annotated ground truth. Reported validation metrics measure how well the neural model learns and generalizes from the RAKE-generated supervision. For production-quality evaluation, create a manually labelled holdout set of SMS messages and run:

```bash
python -m model.evaluate --profile <profile_name> --data-path path/to/manual_holdout.pkl --split all
```

That manual holdout score is the strongest estimate of real-world keyword extraction quality.
