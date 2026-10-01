# Samsung PRISM Worklet: On-Device Topic Modelling for Small Text Data

## Overview

Notifications are among the most frequently used mobile features, but they can quickly become noisy and overwhelming for users. In real-world mobile environments, notifications come from a wide variety of apps, languages, locations, and user preferences. As a result, the same notification list can have very different relevance depending on the individual user and the context in which it appears.

This project addresses a key challenge in mobile AI: understanding the content of short text notifications and extracting meaningful topics or interest signals in a lightweight, on-device, privacy-preserving manner.

The goal is not just to detect the presence of keywords, but to infer what a user is currently interested in based on the notification text and description. These inferred topics can then be used to rank or prioritize notifications, improving the overall user experience without requiring cloud-based processing.

---

## Problem Statement

• Notifications in mobile devices are a widely known feature and are heavily used by apps to direct users to their app features and updates.
• Sometimes the notification stream can become overwhelming for the user, creating a need to simplify and prioritize the notification list.
• The content of notifications varies significantly based on app type, location, language, user preferences, and device context.
• To rank notifications for a user, several factors may be considered, including app preference, contact preference, and content preference.
• We focus specifically on content preference and try to derive the topics a user is currently interested in based on notification text and description.

Examples:

- Notification: “Ferns City | Storms – Thunderstorms possible in the next few hours”
  - Topics: Ferns City, Thunderstorms, weather

- Notification: “Greater Bengaluru Authority replaces BBMP”
  - Topics: BBMP, Bengaluru, politics, Greater Bengaluru Authority

This work explores the on-device implementation of topic modeling for small text data. The focus is on building a compact, efficient, and interpretable approach that can operate locally on the device while respecting resource constraints.

---

## Motivation

The mobile notification experience is highly personalized. A user may care more about weather alerts, travel updates, banking notices, or local civic updates than about generic promotional messages. If the system understands the semantic content of notifications, it can help simplify the user interface and reorder notifications based on topical relevance.

However, notification text is often:

- Short and noisy
- Ambiguous or context-dependent
- Inconsistent in style and vocabulary
- Written in different languages or abbreviations
- Highly variable in semantics across apps

This makes conventional topic modeling methods difficult, especially when they are constrained by mobile performance, memory, and latency requirements.

---

## Project Goal

The project aims to build an on-device topic modeling pipeline for short text notifications using lightweight language models and efficient inference strategies.

The system should:

- understand notification content from short text inputs
- extract meaningful keywords and topics
- operate efficiently on-device
- maintain low memory and computational overhead
- support fast inference with mobile-friendly deployment formats such as TFLite
- provide reliable evaluation using standard metrics such as accuracy, precision, recall, and F1 score

---

## Core Challenges

### 1. Small text, high variability
Small text blocks do not provide enough context for traditional bag-of-words or large statistical models to work reliably. Short notifications may contain only a few words, abbreviations, named entities, or domain-specific phrases.

### 2. On-device constraints
Mobile inference must respect:

- limited CPU and memory
- low battery consumption
- fast response times
- small model size
- compatibility with edge deployment frameworks

### 3. Topic extraction quality
The system must identify not just isolated keywords but also meaningful topical signals such as city names, weather concepts, political entities, or app-specific categories.

### 4. Evaluation under real deployment conditions
The solution must be evaluated with objective metrics and benchmarked against runtime constraints. This includes both quality metrics and system metrics such as memory footprint, inference latency, and model footprint.

---

## Proposed Approach

This project uses a lightweight NLP pipeline for short-text understanding built around on-device keyword extraction and downstream topic inference.

### High-level pipeline

1. Input notification text
2. Clean and normalize the text
3. Extract candidate keywords or topic-bearing tokens
4. Map extracted signals to topic categories or semantic concepts
5. Rank or prioritize notifications based on inferred topics and user relevance
6. Run inference on-device using a mobile-optimized model deployment

### Why keyword extraction is useful
Keyword extraction is a practical first step toward topic understanding because it can capture the most informative terms in short text. These terms often correspond to the underlying topic, event, or subject.

In this implementation, the model is trained as a token-level classification task to identify keyword-like tokens from text. This helps form the foundation for downstream topic modeling and notification ranking.

---

## Technical Implementation

This repository implements a mobile-oriented NLP workflow for short text using transformer-based token classification and TFLite export.

### Model strategy

The solution uses compact transformer-based models trained on short-text pseudo-labeled data. The project supports multiple profiles, including:

- MobileBERT baseline for compact mobile deployment
- DistilBERT for a stronger accuracy/performance trade-off
- Larger transformer variants for higher-accuracy experimentation

These models are designed to extract salient keywords and topic-bearing terms from short messages.

### Deployment strategy

To support Android on-device inference, the trained model is converted to TFLite, enabling:

- smaller model size
- fast execution on mobile hardware
- lower memory footprint than full floating-point inference
- compatibility with mobile AI deployment pipelines

---

## Workflow

### 1. Data collection and preprocessing
The project uses public short-text datasets such as the UCI SMS collection and applies preprocessing to normalize the text and generate weak labels for keyword extraction.

### 2. Pseudo-label generation
Weak supervision is created using keyword extraction strategies such as RAKE. These pseudo-labels help assign keyword-like targets to short texts without requiring large-scale manual annotation.

### 3. Model training
A token-classification model is trained to predict whether each token is keyword-bearing or not.

### 4. Evaluation
The trained model is evaluated using standard classification metrics, including:

- Accuracy
- Precision
- Recall
- F1 Score
- Additional metrics such as macro F1, confusion matrix, and balanced accuracy where applicable

### 5. TFLite conversion
The trained model is converted to TFLite for mobile deployment, allowing evaluation of runtime performance and portability.

### 6. On-device inference
The exported model is used for on-device keyword and topic extraction from notification-like text samples.

---

## Example Notification Interpretation

The following examples illustrate the intended behavior of the topic-modeling pipeline:

### Example 1
Notification:

> Ferns City | Storms – Thunderstorms possible in the next few hours

Inferred topics:

- Ferns City
- Thunderstorms
- weather

### Example 2
Notification:

> Greater Bengaluru Authority replaces BBMP

Inferred topics:

- BBMP
- Bengaluru
- politics
- Greater Bengaluru Authority

These examples demonstrate that the system should capture topical concepts from short notifications, not just generic words or app names.

---

## Evaluation Metrics

To assess model quality, the project measures the following metrics:

- Accuracy: proportion of correctly predicted labels over total predictions
- Precision: measure of how many predicted keywords are actually relevant
- Recall: measure of how many true relevant keywords were correctly identified
- F1 Score: harmonic mean of precision and recall
- Additional classification metrics for deeper analysis depending on the implementation

For real-world mobile deployment, the following system-level metrics are also important:

- inference latency
- model size
- peak memory usage
- TFLite conversion footprint
- battery impact
- deployment suitability on Android devices

---

## Why this matters for Samsung PRISM

This work aligns closely with Samsung PRISM goals by focusing on intelligent, resource-aware AI for mobile devices. The challenge is to build a system that can derive user-relevant topics from short, noisy, and heterogeneous notification content while staying efficient enough to run on-device.

This is important because on-device intelligence enables:

- better privacy preservation
- faster response times
- improved personalization
- less reliance on cloud services
- a smoother and more relevant notification experience

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── download_data.py
│   ├── preprocess.py
│   ├── sms_data.csv
│   └── SMSSpamCollection
├── model/
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── model.py
│   ├── train.py
│   └── compare_models.py
├── convert/
│   ├── convert_to_tflite.py
│   └── validate_tflite.py
├── inference/
│   └── predict.py
├── outputs/
│   ├── best_model/
│   ├── metrics_val.json
│   ├── decision_threshold.json
│   └── *.tflite
├── tests/
│   └── test_inference_behavior.py
├── paper/
│   └── DRAFT.md
└── README.md
```

---

## Setup

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

## Model Profiles

The project supports multiple configuration profiles depending on the target trade-off between speed, memory, and accuracy.

| Profile | Base model | Purpose | Notes |
|---|---|---|---|
| `mobilebert_baseline` | `google/mobilebert-uncased` | Original compact implementation | Best balance for mobile/TFLite deployment |
| `distilbert_accuracy` | `distilbert-base-uncased` | More accurate and still relatively light | Good middle ground |
| `bert_base_high_accuracy` | `bert-base-uncased` | Highest quality target | More expensive in latency and memory |

---

## End-to-End Usage

### 1. Download the dataset

```bash
python -m data.download_data
```

### 2. Preprocess the dataset for a profile

The tokenizer must match the model profile. Run preprocessing once per profile you want to train.

```bash
# Original MobileBERT baseline
python -m data.preprocess --profile mobilebert_baseline

# Larger models
python -m data.preprocess --profile distilbert_accuracy
python -m data.preprocess --profile bert_base_high_accuracy
```

### 3. Train a model

```bash
python -m model.train --profile mobilebert_baseline
python -m model.train --profile distilbert_accuracy
python -m model.train --profile bert_base_high_accuracy
```

The best checkpoint is chosen according to validation F1 score for keyword extraction. Metrics are saved as JSON in the `outputs/` directory.

### 4. Evaluate a trained model

```bash
python -m model.evaluate --profile mobilebert_baseline
python -m model.evaluate --profile distilbert_accuracy
python -m model.evaluate --profile bert_base_high_accuracy
```

This reports metrics such as:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1 score
- Macro F1 and weighted F1
- Confusion matrix
- Model parameter count
- Estimated checkpoint size

### 5. Compare model profiles

```bash
python -m model.compare_models --evaluate-missing
```

This generates comparison tables and reports for the implemented profiles.

### 6. Convert the trained model to TFLite

```bash
python -m convert.convert_to_tflite --profile mobilebert_baseline
python -m convert.convert_to_tflite --profile distilbert_accuracy
python -m convert.convert_to_tflite --profile bert_base_high_accuracy
```

The baseline exports are kept in standard output paths, while larger profiles are written to their respective output folders.

### 7. Validate TFLite output against the original model

```bash
python -m convert.validate_tflite --profile mobilebert_baseline
```

This checks whether the TFLite export preserves keyword extraction behavior comparable to the original PyTorch checkpoint.

### 8. Run inference

```bash
python -m inference.predict --profile mobilebert_baseline --text "Your OTP is 482910. Valid for 10 minutes."
python -m inference.predict --profile distilbert_accuracy --text "Your card was debited Rs 500 at ATM."
python -m inference.predict --interactive
```

---

## How Keyword Extraction Works

This is a token classification task. The model assigns a binary label to each token or wordpiece:

```text
Input  : your otp is 482910 valid 10 minutes
Tokens : [your] [otp] [is] [482910] [valid] [10] [minutes]
Labels :   0     1    0      1       1      0      1
Output : otp, 482910, valid, minutes
```

Labels are generated automatically using weak supervision such as RAKE-based pseudo-labeling, so the pipeline can work without a fully manually annotated dataset.

---

## Useful Training Overrides

```bash
python -m model.train --profile bert_base_high_accuracy --batch-size 4 --epochs 8
python -m model.train --profile distilbert_accuracy --no-threshold-tuning
python -m model.train --profile mobilebert_baseline --force-cpu
```

---

## Example Output

An example inference result may look like:

```text
Input SMS:
Your OTP is 482910. Valid for 10 minutes.

Extracted Keywords:
- OTP
- 482910
- valid
- minutes
```

This is the core primitive behind turning short notification text into topic-bearing signals.

---

## Expected Outcome

This project aims to build a lightweight, mobile-friendly pipeline that can:

- extract important topic-bearing keywords from notification text
- infer meaningful user interests from text content
- support ranking and prioritization of notifications
- run efficiently on-device with low memory and latency
- evaluate quality using accuracy, precision, recall, and F1 score

---

## Future Work

The next steps include:

- collecting a notification-specific dataset with topic annotations
- extending keyword extraction to full topic classification
- combining content preference with app and contact preference for ranking
- benchmarking on Android devices for real latency and memory measurements
- comparing lightweight on-device methods against alternative topic-modeling baselines

---

## Conclusion

This work addresses a real and practical mobile AI challenge: understanding the content of short notifications and identifying topics relevant to the user while staying efficient enough to run on-device. By combining a compact transformer-based approach with TFLite deployment and standard evaluation metrics, the project provides a useful foundation for smarter and more personalized notification ranking.

---

## Setup

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Download the data

```bash
python -m data.download_data
```

### 2. Preprocess data

```bash
python -m data.preprocess --profile mobilebert_baseline
```

### 3. Train a model

```bash
python -m model.train --profile mobilebert_baseline
```

### 4. Evaluate the model

```bash
python -m model.evaluate --profile mobilebert_baseline
```

### 5. Compare profiles

```bash
python -m model.compare_models --evaluate-missing
```

### 6. Convert to TFLite

```bash
python -m convert.convert_to_tflite --profile mobilebert_baseline
```

### 7. Run inference

```bash
python -m inference.predict --profile mobilebert_baseline --text "Your OTP is 482910. Valid for 10 minutes."
```

---

## Expected Outcome

The project aims to produce a lightweight mobile-friendly NLP pipeline that can:

- extract important terms from short notification text
- infer relevant topics from the content
- enable more intelligent prioritization and ranking of notifications
- work efficiently on-device under memory and latency constraints
- provide quantifiable evaluation using standard quality metrics and deployment benchmarks

---

## Future Work

This project is an initial step toward a more complete notification intelligence system. The next stages include:

- collecting a mobile-notification-specific dataset with human-annotated topics
- extending keyword extraction into document-level topic classification
- building a ranking model that uses topic relevance, app preference, and contact preference together
- evaluating performance on Android devices with real latency and memory measurements
- comparing lightweight transformer models against alternative topic-modeling strategies

---

## Conclusion

This work tackles a practical and high-impact mobile AI problem: understanding the content of short notifications and identifying the topics that matter most to the user. By focusing on on-device performance, memory efficiency, and measurable quality, the project aims to push forward a realistic solution for smarter notification ranking and user-centric content prioritization.

The resulting approach is especially relevant for mobile contexts where responsiveness, privacy, and resource efficiency are critical requirements.
