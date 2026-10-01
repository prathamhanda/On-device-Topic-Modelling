# Draft Paper: On-Device Topic/Keyword Extraction for Short Notifications

## Abstract

We study lightweight on-device keyword extraction for short notification text, and outline a path toward topic extraction suitable for mobile devices. Using RAKE weak supervision over the UCI SMS Spam Collection, we train compact token classifiers (MobileBERT baseline) and larger variants (DistilBERT). We convert models to TFLite and measure fidelity and performance. We report token-level keyword metrics and propose a roadmap to extend this work to topic modeling for notification ranking.

## 1. Introduction

Smart notification experiences require understanding the content of short texts (notifications, SMS). We target on-device methods that are small, fast, and privacy-preserving. The current codebase implements token-level keyword extraction as a first-stage signal for topic / intent derivation.

## 2. Related Work

- Keyword extraction: RAKE, TF-IDF, supervised token classifiers.
- Topic modeling for short text: LDA variants, embedding-based clustering (Sentence-BERT, BERTopic), supervised classifiers.
- On-device NLP: TFLite quantized Transformer models, mobile-first architectures (MobileBERT, DistilBERT).

## 3. Dataset

We used the public UCI SMS Spam Collection to demonstrate the pipeline and RAKE pseudo-labeling. For production topic extraction, a notification-specific labeled dataset is required (title + content → topics).

## 4. Methods

4.1 RAKE weak supervision

We extract candidate keywords per message using RAKE and align word-level labels to tokenized sub-tokens (first sub-token marked positive).

4.2 Model profiles

- `mobilebert_baseline` — `google/mobilebert-uncased` small, mobile-friendly.
- `distilbert_accuracy` — `distilbert-base-uncased`, larger and more accurate.

4.3 Training details

Training supports class-weighted CrossEntropy, gradient accumulation, warmup scheduling, and threshold tuning on validation for keyword probability decoding. Hyperparameters are in `model/config.py`.

## 5. Experiments and Results

We trained and evaluated the profiles with RAKE labels. Representative results (see `DOCS/EXPERIMENTS.md` and attached figures):

- MobileBERT baseline: keyword F1 ≈ 0.88–0.90 (small checkpoint, ~24M params).
- DistilBERT: keyword F1 ≈ 0.97 (larger checkpoint, ~66M params).

We converted the trained PyTorch checkpoints to both float and dynamic-range quantized TFLite. Quantized TFLite retained close keyword extraction fidelity in sample checks (see `convert/validate_tflite.py`).

## 6. Discussion

Keyword extraction is a useful primitive for notifications, enabling downstream topic mapping, entity recognition, or intent classification. However, true topic labels require a different dataset and evaluation protocol. Two practical approaches:

1) Supervised topic classifier: collect labeled notifications and fine-tune a small Transformer on document-level topic labels; export to TFLite.
2) Embedding + clustering: compute compact embeddings (distil/SBERT) and use nearest-centroid/cluster assignment for topics — lightweight at inference if centroids are pre-computed.

## 7. On-device Integration & Benchmarks

- The repo includes TFLite exports and a Python inference wrapper. To finalize on-device integration we recommend an Android sample that loads the dynamic-quant TFLite, runs inference, and measures latency/memory across devices.
- Benchmark metrics to collect: inference latency (ms), peak RAM, binary size, and keyword-topic accuracy on a labeled holdout.

## 8. Conclusion & Next Steps

We provide a reproducible keyword extraction pipeline and TFLite exports suitable for prototyping on-device notification understanding. To reach the project goal of topic modeling and notification ranking, the primary outstanding work is dataset collection & labeling, a topic-modeling prototype (supervised or embedding-based), and an Android demo to measure real on-device performance.

## Appendix — Transcribed Examples and Profile Table

Below are text transcriptions of example outputs and the profile comparison table as shown in the project screenshots.

Example outputs (console-style):

```
===============================================================
Device    : CPU
Model     : Fine-tuned DistilBERT
Task      : SMS Keyword Extraction
Dataset   : UCI SMS Spam Collection + RAKE weak supervision
Best Score: Accuracy 97.44% | Keyword F1 97.01%
===============================================================

Example 1
---------------------------------------------------------------
Input SMS:
Your OTP is 482910. Valid for 10 minutes.

Extracted Keywords:
 - OTP                 confidence=0.98
 - 482910              confidence=0.97
 - 10 minutes          confidence=0.94

Example 2
---------------------------------------------------------------
Input SMS:
Your HDFC account was debited Rs 5000 at Amazon Pay.

Extracted Keywords:
 - HDFC account        confidence=0.96
 - debited             confidence=0.95
 - Rs 5000             confidence=0.97
 - Amazon Pay          confidence=0.93

Example 3
---------------------------------------------------------------
Input SMS:
Recharge successful for mobile number 9876543210.

Extracted Keywords:
 - Recharge successful confidence=0.96
 - mobile number       confidence=0.94
 - 9876543210          confidence=0.98

Example 4
---------------------------------------------------------------
Input SMS:
Congratulations! You have won a free iPhone. Click the link to claim now.

Extracted Keywords:
 - Congratulations      confidence=0.95
 - free iPhone          confidence=0.97
 - Click the link       confidence=0.94
 - claim now            confidence=0.92

Example 5
---------------------------------------------------------------
Input SMS:
Reminder: your electricity bill of Rs 1240 is due tomorrow.

Extracted Keywords:
 - electricity bill     confidence=0.96
 - Rs 1240             confidence=0.97
 - due tomorrow        confidence=0.95
===============================================================
```

Profile comparison (metrics table):

| profile               | model_name                     | status | accuracy | keyword_f1 | keyword_precision | keyword_recall | macro_f1 | params_m | checkpoint_mb | decision_threshold |
|-----------------------|--------------------------------|--------|--------:|-----------:|------------------:|---------------:|---------:|--------:|--------------:|-------------------:|
| mobilebert_baseline   | google/mobilebert-uncased      | ok     |  0.8947 |     0.8820 |            0.8424 |         0.9255 |   0.8935 |   24.58 |         93.91 |               0.49 |
| distilbert_accuracy   | distilbert-base-uncased        | ok     |  0.9744 |     0.9701 |            0.9623 |         0.9781 |   0.9739 |   66.36 |        253.17 |               0.44 |

*End transcription.*

## Acknowledgements

This work reuses public datasets and open-source tooling (HuggingFace Transformers, TensorFlow, PyTorch, RAKE).

---