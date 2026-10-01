# Experiments, Results & Evaluation

This document records dataset, preprocessing, training hyperparameters, and results observed during experiments. It also includes guidance for evaluating topic-extraction variants (future work).

Dataset
- UCI SMS Spam Collection (downloaded to `data/sms_data.csv`).
- Pseudo-labels: RAKE-generated keyword pseudo-labels were used to train token-level keyword extractors.

Preprocessing
- Text cleaning: lowercase, remove URLs, collapse whitespace.
- RAKE extracts word-level keywords; labels are aligned to tokenizer first sub-token.

Training
- Profiles implemented in `model/config.py`:
  - `mobilebert_baseline` : `google/mobilebert-uncased` (compact, mobile-first)
  - `distilbert_accuracy` : `distilbert-base-uncased` (accuracy-focused)
  - `bert_base_high_accuracy` : `bert-base-uncased` (largest)
- Typical hyperparameters are in `model/config.py`; training supports class-weighted loss, gradient accumulation, and threshold tuning.

Metrics
- Token-level metrics reported by the pipeline include: accuracy, balanced accuracy, keyword precision/recall/F1, macro/weighted F1, MCC, confusion matrix, parameter counts and checkpoint size.
- For topic modeling (future work) we recommend document-level metrics (accuracy, precision, recall, macro-F1) over a manually-labeled holdout.

Results (example)
- The project includes example results (see attachments). Example experimental snapshot:

| profile | model_name | accuracy | keyword_f1 | params_m | checkpoint_mb | decision_threshold |
|---|---:|---:|---:|---:|---:|---:|
| mobilebert_baseline | google/mobilebert-uncased | 0.8947 | 0.8820 | 24.58 | 93.91 | 0.49 |
| distilbert_accuracy | distilbert-base-uncased | 0.9744 | 0.9701 | 66.36 | 253.17 | 0.44 |

Note: Your local run earlier produced `mobilebert_baseline` metrics of Accuracy 0.9089 and Keyword F1 0.8957 (see `outputs/metrics_val.json`). Differences can come from seed, train/val split, or small code/config differences.

Figures
- Example outputs screenshot: `docs/images/example_outputs.png` — contains sample SMS inputs and extracted keywords with confidences.
- Profile comparison table: `docs/images/profile_table.png` — summary of accuracy and keyword F1 for profiles.

How to reproduce an experiment
1. Ensure `data/processed.pkl` exists for the profile: `python -m data.preprocess --profile <profile>`
2. Train: `python -m model.train --profile <profile>`
3. Evaluate: `python -m model.evaluate --profile <profile>`
4. Convert & validate: `python -m convert.convert_to_tflite --profile <profile>` and `python -m convert.validate_tflite --profile <profile>`

Future evaluation for topic-extraction (recommended)
- Build a manually-labeled holdout set of notifications (title + description) with topic labels.
- Evaluate per-document topic predictions vs ground truth: accuracy, precision, recall, F1 (micro/macro), confusion matrix.
- Benchmark on-device latency & memory for each profile/TFLite variant.
