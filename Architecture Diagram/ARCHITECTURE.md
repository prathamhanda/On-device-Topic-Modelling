**Architecture**

Overview
 - This repository implements a mobile-first keyword extraction pipeline for SMS/notification text. The flow is: data collection -> RAKE pseudo-labeling -> token-level fine-tuning (HuggingFace) -> export to TFLite -> on-device inference and post-processing.

Diagrams

System architecture diagram (PNG): [diagrams/system_arch.png](diagrams/system_arch.png)

Model & export pipeline (PNG): [diagrams/model_pipeline.png](diagrams/model_pipeline.png)

Models used
 - `mobilebert_baseline`: `google/mobilebert-uncased` (fine-tuned for token classification)
 - `distilbert_accuracy`: `distilbert-base-uncased` (optional higher-accuracy profile)
 - `bert_base_high_accuracy`: `bert-base-uncased` (highest capacity option)

Fine-tuning details
 - Task: token classification (2 labels — NON_KEYWORD / KEYWORD)
 - Supervision: weak supervision using RAKE pseudo-labels plus optional manual labels
 - Training: HuggingFace `AutoModelForTokenClassification` fine-tuned with class-weighted loss, scheduler, and early-stopping (see `model/train.py`)
 - Export: best checkpoint saved to `outputs/best_model/` then converted to TF and TFLite using `convert/convert_to_tflite.py`

External repositories & resources
 - HuggingFace model hubs:
   - https://huggingface.co/google/mobilebert-uncased
   - https://huggingface.co/distilbert-base-uncased
   - https://huggingface.co/bert-base-uncased
 - Libraries used: HuggingFace Transformers, PyTorch, TensorFlow (TFLite), rake-nltk.

Files of interest
 - `model/config.py` — model profiles and paths
 - `data/preprocess.py` — RAKE weak labels and tokenization
 - `model/train.py` — fine-tuning script
 - `convert/convert_to_tflite.py` — export script (from PT -> TF -> TFLite)
 - `inference/predict.py` — TFLite runtime extraction and CLI
