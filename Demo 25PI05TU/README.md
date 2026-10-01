Demo recording quick-start
=========================

1) Activate venv

```bash
cd /Users/priyanshigoyal/Desktop/Samsung PRISM
source .venv/bin/activate
```

2) Run interactive CLI (for live typing during recording)

```bash
python -m inference.predict --interactive
# type SMS text at the prompt, press Enter; type 'quit' to exit
```

3) Run programmatic demo (useful to record a single command run)

```bash
python3 demo/demo_run.py
```

4) (Optional) Export a smaller TFLite for low-latency inference

```bash
# Export dynamic-quant TFLite with max token length = 32 (smaller graph -> faster)
python -m convert.convert_to_tflite --profile mobilebert_baseline --skip-float --export-max-len 32

# For full int8 quantization (requires representative data):
FULL_INT8_QUANT=1 python -m convert.convert_to_tflite --profile mobilebert_baseline --skip-float --export-max-len 32
```

5) Sample output :

```
Loaded profile: mobilebert_baseline
TFLite: outputs/sms_keyword_extractor_dynamic_quant.tflite

SMS: Your OTP is 482910. Valid for 10 mins.
Keywords: ['ot', '48', 'valid', '10', 'min']
Elapsed: 27.4 ms

SMS: Your bill of Rs 1200 is due tomorrow.
Keywords: ['bill', 'rs', '1200', 'due', 'tomorrow']
Elapsed: 18.8 ms
```
