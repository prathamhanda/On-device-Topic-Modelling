# Implementation & Run Guide

This document explains how to reproduce the full pipeline, run single-SMS inference, and where experimental artifacts live.

**Environment**
- **Python:** 3.11 (venv recommended)
- **Install:** `pip install -r requirements.txt` inside the project venv

Quick setup (from repo root):
```bash
# create venv and install deps
. /opt/homebrew/bin/python3.11 -m venv .venv
. . .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Full pipeline (reproducible)**₹   
```bash
# 1) download data
.venv/bin/python -m data.download_data

# 2) preprocess (one profile at a time)
.venv/bin/python -m data.preprocess --profile mobilebert_baseline

# 3) train
.venv/bin/python -m model.train --profile mobilebert_baseline

# 4) evaluate
.venv/bin/python -m model.evaluate --profile mobilebert_baseline

# 5) convert to TFLite (float + dynamic-quant)
.venv/bin/python -m convert.convert_to_tflite --profile mobilebert_baseline

# 6) validate TFLite vs PyTorch
.venv/bin/python -m convert.validate_tflite --profile mobilebert_baseline

# 7) inference (single SMS)
.venv/bin/python -m inference.predict --profile mobilebert_baseline --text "Your OTP is 482910. Valid for 10 minutes."
```

Programmatic usage (Python API)
```python
from inference.predict import SMSKeywordExtractor
extractor = SMSKeywordExtractor(profile_name="mobilebert_baseline")
print(extractor.extract_keywords("Your OTP is 482910. Valid for 10 minutes."))
```

Artifacts produced by pipeline
- `data/processed.pkl` (tokenized samples for default profile)
- `outputs/best_model/` (HuggingFace checkpoint)
- `outputs/metrics_val.json` (validation metrics)
- `outputs/sms_keyword_extractor.tflite` (float)
- `outputs/sms_keyword_extractor_dynamic_quant.tflite` (dynamic-range quantized)

Figures (attached images)
- Place the example output screenshot at `docs/images/example_outputs.png` and the profile comparison table at `docs/images/profile_table.png` so they are embedded in the report and paper.

Notes
- If converting to TFLite fails with a Keras/transformers compatibility message, install `tf-keras` in the venv: `.venv/bin/python -m pip install tf-keras`.
- On macOS with LibreSSL you may see an `urllib3 NotOpenSSLWarning` — it's a harmless runtime warning.
