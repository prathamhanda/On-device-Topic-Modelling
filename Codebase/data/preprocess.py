"""Preprocess SMS data for token-level keyword extraction.

Steps:
  1. Clean raw text
  2. Use RAKE to auto-generate keyword pseudo-labels
  3. Tokenize with the selected profile tokenizer
  4. Align word-level keyword labels to the first sub-token of each word
  5. Save processed samples as pickle

Examples:
    python -m data.preprocess
    python -m data.preprocess --profile distilbert_accuracy
    python -m data.preprocess --profile bert_base_high_accuracy
"""
from __future__ import annotations

import argparse
import pickle
import re
import string
import sys
from pathlib import Path

import nltk
import pandas as pd
from rake_nltk import Rake
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.config import DEFAULT_PROFILE, MODEL_PROFILES, get_profile, processed_data_path

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)  # required by NLTK >= 3.9

DATA_DIR = Path(__file__).parent
SMS_CSV = DATA_DIR / "sms_data.csv"
OUT_PATH = processed_data_path(DEFAULT_PROFILE)

# Backward-compatible defaults
MODEL_NAME = get_profile(DEFAULT_PROFILE).model_name
MAX_LEN = get_profile(DEFAULT_PROFILE).max_len


def clean_text(text: str) -> str:
    """Normalize SMS text while preserving useful numbers and punctuation context."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_keywords_rake(text: str) -> set[str]:
    """Return a set of word-level RAKE pseudo-keywords for an SMS."""
    r = Rake(min_length=1, max_length=3)
    r.extract_keywords_from_text(text)
    keywords: set[str] = set()
    for phrase in r.get_ranked_phrases():
        for word in phrase.split():
            cleaned = word.lower().strip(string.punctuation)
            if cleaned:
                keywords.add(cleaned)
    return keywords


def align_labels(tokenizer, words: list[str], keyword_set: set[str], max_len: int) -> dict:
    """Tokenize words and produce token-level binary labels.

    Label mapping:
      - 1 on the first sub-token of a keyword word
      - 0 on the first sub-token of a non-keyword word
      - -100 on special tokens, padding, and later sub-tokens so loss ignores them
    """
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        max_length=max_len,
        truncation=True,
        padding="max_length",
    )
    if not hasattr(encoding, "word_ids"):
        raise ValueError(
            "The selected tokenizer must be a fast tokenizer with word_ids() support. "
            "Use AutoTokenizer(..., use_fast=True)."
        )

    word_ids = encoding.word_ids()
    labels: list[int] = []
    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        elif word_id != prev_word_id:
            word = words[word_id].lower().strip(string.punctuation)
            labels.append(1 if word in keyword_set else 0)
        else:
            labels.append(-100)
        prev_word_id = word_id

    encoding["labels"] = labels
    return encoding


def preprocess(
    profile_name: str = DEFAULT_PROFILE,
    model_name: str | None = None,
    max_len: int | None = None,
    sms_csv: Path = SMS_CSV,
    out_path: Path | None = None,
) -> Path:
    """Create a tokenized pickle file for a profile/model tokenizer."""
    profile = get_profile(profile_name)
    resolved_model_name = model_name or profile.model_name
    resolved_max_len = max_len or profile.max_len
    resolved_out_path = out_path or processed_data_path(profile_name)

    if not sms_csv.exists():
        raise FileNotFoundError(
            f"Missing {sms_csv}. Run `python -m data.download_data` first or pass --sms-csv."
        )

    df = pd.read_csv(sms_csv)
    if "text" not in df.columns:
        raise ValueError(f"{sms_csv} must contain a 'text' column")

    print(f"[INFO] Profile       : {profile_name}")
    print(f"[INFO] Tokenizer     : {resolved_model_name}")
    print(f"[INFO] Max length    : {resolved_max_len}")
    print(f"[INFO] Source CSV    : {sms_csv}")
    print(f"[INFO] Output pickle : {resolved_out_path}")

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, use_fast=True)
    samples: list[dict] = []
    skipped_empty = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        text = clean_text(str(row["text"]))
        words = text.split()
        if not words:
            skipped_empty += 1
            continue

        keyword_set = extract_keywords_rake(text)
        enc = align_labels(tokenizer, words, keyword_set, resolved_max_len)
        samples.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": enc["labels"],
                # Extra metadata is ignored by the Dataset class but useful for debugging.
                "text": text,
                "words": words,
                "rake_keywords": sorted(keyword_set),
                "profile": profile_name,
                "tokenizer": resolved_model_name,
                "max_len": resolved_max_len,
            }
        )

    resolved_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_out_path, "wb") as f:
        pickle.dump(samples, f)

    print(f"[INFO] Saved {len(samples)} samples to {resolved_out_path}")
    if skipped_empty:
        print(f"[INFO] Skipped {skipped_empty} empty messages")
    return resolved_out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess SMS data for a model profile")
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        choices=sorted(MODEL_PROFILES),
        help="Model profile whose tokenizer/max_len should be used",
    )
    parser.add_argument("--model-name", default=None, help="Override HuggingFace tokenizer name")
    parser.add_argument("--max-len", type=int, default=None, help="Override max token length")
    parser.add_argument("--sms-csv", type=Path, default=SMS_CSV, help="Input CSV with a text column")
    parser.add_argument("--out-path", type=Path, default=None, help="Output pickle path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess(
        profile_name=args.profile,
        model_name=args.model_name,
        max_len=args.max_len,
        sms_csv=args.sms_csv,
        out_path=args.out_path,
    )


if __name__ == "__main__":
    main()
