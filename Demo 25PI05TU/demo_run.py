"""Simple demo runner for SMSKeywordExtractor.

Usage:
    source .venv/bin/activate
    python demo/demo_run.py

Prints extracted keywords and per-inference latency (ms) for a few examples.
"""
from time import perf_counter
import sys
from pathlib import Path

# ensure repo root is importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.predict import SMSKeywordExtractor


SAMPLES = [
    "Your OTP is 482910. Valid for 10 mins.",
    "Your bill of Rs 1200 is due tomorrow.",
    "Win a free phone—reply YES to enter the contest!",
    "Doctor appointment tomorrow 10AM. Please arrive 10 minutes early.",
]


def run_demo(profile: str = "mobilebert_baseline"):
    ext = SMSKeywordExtractor(profile_name=profile)
    print(f"Loaded profile: {profile}\nTFLite: {ext.tflite_path}\n")

    for text in SAMPLES:
        t0 = perf_counter()
        kws = ext.extract_keywords(text)
        topics = ext.extract_topics(text)
        dt = (perf_counter() - t0) * 1000.0
        print(f"SMS: {text}")
        print(f"Keywords: {kws}")
        print(f"Topics: {topics}")
        print(f"Elapsed: {dt:.1f} ms\n")


if __name__ == "__main__":
    run_demo()
