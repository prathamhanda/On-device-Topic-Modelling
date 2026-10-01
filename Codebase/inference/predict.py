"""SMS Keyword Extractor user-facing inference script.

Loads a TFLite model and extracts keywords from any SMS text.

Usage:
    python -m inference.predict --text "Your OTP is 482910. Valid for 10 mins."
    python -m inference.predict --profile distilbert_accuracy --text "Your card was debited Rs 500"
    python -m inference.predict --interactive
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.config import (
    DEFAULT_PROFILE,
    MODEL_PROFILES,
    get_profile,
    processed_data_path,
    threshold_path,
    tflite_path,
)
from model.metrics import load_json

COMMON_TYPO_MAP = {
    "amnt": "amount",
    "amout": "amount",
    "amountt": "amount",
    "bal": "balance",
    "balnce": "balance",
    "blance": "balance",
    "crd": "card",
    "cardd": "card",
    "ot": "otp",
    "otpp": "otp",
    "otpt": "otp",
    "pwd": "password",
    "passwrd": "password",
    "passord": "password",
    "validte": "valid",
    "validatn": "valid",
    "vaild": "valid",
    "vald": "valid",
    "mins": "minutes",
    "minitues": "minutes",
    "minuts": "minutes",
    "pls": "please",
    "plz": "please",
    "ur": "your",
    "fr": "for",
    "msg": "message",
    "txn": "transaction",
    "trnsaction": "transaction",
    "transactn": "transaction",
    "paymnt": "payment",
    "pymt": "payment",
    "pmt": "payment",
    "isn": "is",
    "isnt": "is",
    "thet": "the",
    "teh": "the",
    "thn": "then",
    "tell": "tell",
    "wll": "will",
    "wil": "will",
    "shld": "should",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "of",
    "on",
    "or",
    "our",
    "please",
    "should",
    "so",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "too",
    "up",
    "was",
    "we",
    "were",
    "will",
    "with",
    "would",
    "your",
    "you",
    "yours",
    "am",
    "pm",
    "rs",
}

DEFAULT_KEYWORD_LEXICON = {
    "otp",
    "code",
    "amount",
    "balance",
    "credit",
    "debit",
    "payment",
    "transaction",
    "valid",
    "minutes",
    "card",
    "bank",
    "atm",
    "password",
    "pin",
    "message",
    "wallet",
    "cashback",
    "refund",
    "invoice",
    "charge",
    "reward",
    "offer",
    "trip",
    "ticket",
    "booking",
    "login",
    "account",
    "verification",
    "order",
    "delivery",
    "status",
    "bill",
    "weather",
    "storm",
    "thunderstorm",
    "rain",
    "bengaluru",
    "bbmp",
    "politics",
    "appointment",
    "doctor",
    "tomorrow",
}


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def resolve_threshold(profile_name: str, threshold: str) -> float | None:
    """Resolve threshold string to float or None for argmax."""
    if threshold.lower() == "argmax":
        return None
    if threshold.lower() == "auto":
        path = threshold_path(profile_name)
        if path.exists():
            payload = load_json(path)
            value = payload.get("decision_threshold")
            if value is not None:
                return float(value)
        return None
    return float(threshold)


class SMSKeywordExtractor:
    """Load a TFLite model once and extract keywords from SMS strings."""

    @staticmethod
    @lru_cache(maxsize=None)
    def _load_keyword_lexicon(profile_name: str) -> frozenset[str]:
        """Load the training vocabulary and normalize common SMS typos into a compact lexicon."""
        lexicon = set(DEFAULT_KEYWORD_LEXICON)
        data_path = processed_data_path(profile_name)
        if data_path.exists():
            try:
                with open(data_path, "rb") as handle:
                    samples = pickle.load(handle)
                for sample in samples:
                    for keyword in sample.get("rake_keywords", []):
                        token = re.sub(r"[^a-z0-9]+", "", str(keyword).lower())
                        if token:
                            lexicon.add(token)
            except (OSError, EOFError, ValueError, pickle.PickleError):
                pass
        return frozenset(lexicon)

    def __init__(
        self,
        profile_name: str = DEFAULT_PROFILE,
        tflite_model_path: Path | None = None,
        threshold: str = "auto",
        infer_max_len: int | None = None,
    ):
        self.profile_name = profile_name
        self.profile = get_profile(profile_name)
        self.keyword_lexicon = self._load_keyword_lexicon(profile_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.profile.model_name, use_fast=True)
        self.tflite_path = Path(tflite_model_path or tflite_path(profile_name, quantized=True))
        if not self.tflite_path.exists():
            raise FileNotFoundError(
                f"Missing TFLite model at {self.tflite_path}. "
                f"Run `python -m convert.convert_to_tflite --profile {profile_name}` first."
            )
        self.threshold = resolve_threshold(profile_name, threshold)
        self.interpreter = tf.lite.Interpreter(model_path=str(self.tflite_path))
        self.interpreter.allocate_tensors()
        self._in = self.interpreter.get_input_details()
        self._out = self.interpreter.get_output_details()
        self._infer_max_len = infer_max_len or self.profile.max_len
        self._fast_path_length = min(12, max(6, self._infer_max_len // 6))

    @staticmethod
    def _clean(text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _repair_candidate_token(token: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "", (token or "").lower())
        if not cleaned:
            return ""
        repaired = COMMON_TYPO_MAP.get(cleaned, cleaned)
        if cleaned in {"thet", "teh"}:
            return "the"
        if cleaned in {"isn", "isnt"}:
            return "is"
        if repaired in STOPWORDS and repaired not in {"the", "then", "is"}:
            return ""
        return repaired

    @staticmethod
    def _canonical_token(token: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "", token.lower())
        return COMMON_TYPO_MAP.get(cleaned, cleaned)

    @staticmethod
    def _is_approx_match(token: str, candidate: str) -> bool:
        if token == candidate:
            return True
        if abs(len(token) - len(candidate)) > 1:
            return False
        ratio = SequenceMatcher(None, token, candidate).ratio()
        return ratio >= 0.8

    def _match_keyword_token(self, token: str) -> str | None:
        canonical = self._canonical_token(token)
        if not canonical:
            return None
        lexicon = getattr(self, "keyword_lexicon", DEFAULT_KEYWORD_LEXICON)
        if canonical in lexicon:
            return canonical
        for candidate in lexicon:
            if abs(len(candidate) - len(canonical)) > 1:
                continue
            if candidate[0] != canonical[0]:
                continue
            if self._is_approx_match(canonical, candidate):
                return candidate
        return None

    @staticmethod
    def _filter_keyword_tokens(tokens: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in tokens:
            repaired = SMSKeywordExtractor._repair_candidate_token(raw)
            if not repaired:
                continue
            if repaired in STOPWORDS:
                continue
            if repaired.isdigit():
                continue
            if len(repaired) <= 2 and repaired not in {"rs", "id", "pm", "am", "otp"}:
                continue
            if repaired in seen:
                continue
            seen.add(repaired)
            cleaned.append(repaired)
        return cleaned

    def _get_fast_path_length(self) -> int:
        if not hasattr(self, "_fast_path_length"):
            self._fast_path_length = 12
        return self._fast_path_length

    def _normalize_tokens(self, text: str) -> list[str]:
        cleaned = self._clean(text)
        tokens = re.findall(r"[a-z0-9]+", cleaned)
        normalized: list[str] = []
        for token in tokens:
            canonical = self._canonical_token(token)
            if canonical and canonical not in STOPWORDS:
                normalized.append(canonical)
        return normalized

    def _fast_path_extract_keywords(self, sms_text: str) -> list[str]:
        """Fast lexicon-only match for common short SMS texts without invoking TFLite."""
        tokens = self._normalize_tokens(sms_text)
        matched: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            keyword = self._match_keyword_token(token)
            if keyword and keyword not in seen:
                seen.add(keyword)
                matched.append(keyword)
        return self._filter_keyword_tokens(matched)

    def _set_inputs(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> None:
        for detail in self._in:
            name = detail["name"].lower()
            dtype = detail["dtype"]
            expected_shape = tuple(detail["shape"])
            if "input_ids" in name:
                arr = input_ids
                if arr.shape != expected_shape:
                    pad_id = self.tokenizer.pad_token_id or 0
                    padded = np.full(expected_shape, pad_id, dtype=arr.dtype)
                    padded[:, : arr.shape[1]] = arr
                    arr = padded
                self.interpreter.set_tensor(detail["index"], arr.astype(dtype))
            elif "attention_mask" in name:
                arr = attention_mask
                if arr.shape != expected_shape:
                    padded = np.zeros(expected_shape, dtype=arr.dtype)
                    padded[:, : arr.shape[1]] = arr
                    arr = padded
                self.interpreter.set_tensor(detail["index"], arr.astype(dtype))

    def _labels_from_logits(self, logits: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            return np.argmax(logits, axis=-1)
        probs = _softmax(logits, axis=-1)[..., 1]
        return (probs >= self.threshold).astype(np.int64)

    def _decode_word_level_predictions(self, text: str, preds: np.ndarray, word_ids: list[int | None]) -> list[str]:
        token_strings = self.tokenizer.convert_ids_to_tokens(self.tokenizer(text, return_tensors="np")["input_ids"][0])
        word_to_parts: dict[int, list[str]] = {}
        word_to_positive: dict[int, bool] = {}

        for token, word_id, pred in zip(token_strings, word_ids, preds):
            if word_id is None:
                continue
            if token in ("[CLS]", "[SEP]", "[PAD]", "[UNK]"):
                continue
            word_to_parts.setdefault(word_id, []).append(token.replace("##", ""))
            if pred == 1:
                word_to_positive[word_id] = True

        decoded: list[str] = []
        for word_id in sorted(word_to_parts):
            parts = [part for part in word_to_parts[word_id] if part]
            if not parts or not word_to_positive.get(word_id, False):
                continue
            word = "".join(parts).lower()
            repaired = self._repair_candidate_token(word)
            if repaired:
                decoded.append(repaired)
        filtered = self._filter_keyword_tokens(decoded)
        if filtered:
            return filtered
        return self._filter_keyword_tokens(self._normalize_tokens(text))

    def extract_keywords(self, sms_text: str) -> list[str]:
        """Return keyword tokens found in the SMS, filtered to meaningful words."""
        text = self._clean(sms_text)
        if text:
            tokens = text.split()
            if len(tokens) <= self._get_fast_path_length():
                fast_hits = self._fast_path_extract_keywords(text)
                if fast_hits:
                    return fast_hits

        enc = self.tokenizer(
            text,
            return_tensors="np",
            max_length=self._infer_max_len,
            truncation=True,
            padding="max_length",
        )
        self._set_inputs(enc["input_ids"], enc["attention_mask"])
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(self._out[0]["index"])[0]
        preds = self._labels_from_logits(logits)

        word_ids = enc.word_ids() if hasattr(enc, "word_ids") else [None] * len(enc["input_ids"][0])
        return self._decode_word_level_predictions(text, preds, word_ids)

    def extract_topics(self, sms_text: str) -> list[str]:
        """Return a small list of likely topics or named entities from the SMS."""
        original = sms_text.strip()
        text = self._clean(sms_text)
        if not text:
            return []

        candidates: list[str] = []
        lower = original.lower()

        # Preserve the original casing for named entities and place names.
        phrase_pattern = re.compile(
            r"\b(?:[A-Z][A-Za-z0-9&.-]*|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9&.-]*|[A-Z]{2,}))*\b"
        )
        candidates.extend(phrase_pattern.findall(original))

        for chunk in re.split(r"[|/\\-–—:]+", original):
            chunk = re.sub(r"[^A-Za-z0-9\s]", " ", chunk).strip()
            if not chunk:
                continue
            for phrase in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", chunk):
                candidates.append(phrase)

        # Also include uppercase acronyms and meaningful keywords.
        candidates.extend(re.findall(r"\b[A-Z]{2,}\b", original))
        keywords = self.extract_keywords(original)
        for keyword in keywords:
            if len(keyword) > 2 and keyword not in STOPWORDS:
                candidates.append(keyword)

        for domain, markers in {
            "weather": ["storm", "storms", "thunderstorm", "thunderstorms", "rain", "weather", "cloud", "wind"],
            "politics": ["bbmp", "bengaluru", "authority", "government", "politics", "minister", "election", "council"],
            "finance": ["bill", "payment", "amount", "card", "balance", "bank", "refund"],
            "health": ["doctor", "appointment", "medicine", "hospital", "clinic", "visit"],
        }.items():
            if any(marker in lower for marker in markers):
                candidates.append(domain.title())

        final: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            cleaned = re.sub(r"\s+", " ", candidate).strip()
            cleaned = cleaned.replace("-", " ")
            if not cleaned:
                continue
            normalized = cleaned.lower()
            if normalized in STOPWORDS:
                continue
            if normalized == "bbmp":
                cleaned = "BBMP"
            elif normalized == "bengaluru":
                cleaned = "Bengaluru"
            elif normalized in {"ferns", "city", "storm", "storms", "thunderstorm", "thunderstorms"}:
                cleaned = cleaned.title()
            if normalized in seen:
                continue
            seen.add(normalized)
            final.append(cleaned)

        if len(final) < 3:
            for word in self.extract_keywords(original):
                if len(word) > 2 and word not in STOPWORDS:
                    normalized = word.lower()
                    if normalized not in seen:
                        seen.add(normalized)
                        final.append(word.title())

        return final[:5]


def main() -> None:
    parser = argparse.ArgumentParser(description="SMS Keyword Extractor (TFLite)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Single SMS text")
    group.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        choices=sorted(MODEL_PROFILES),
        help="Model profile to load",
    )
    parser.add_argument("--tflite-path", type=Path, default=None, help="Override TFLite model path")
    parser.add_argument("--threshold", default="auto", help="auto, argmax, or numeric threshold")
    parser.add_argument(
        "--infer-max-len",
        type=int,
        default=None,
        help="Override tokenizer max length at inference time to reduce latency",
    )
    args = parser.parse_args()

    extractor = SMSKeywordExtractor(
        profile_name=args.profile,
        tflite_model_path=args.tflite_path,
        threshold=args.threshold,
        infer_max_len=args.infer_max_len,
    )

    if args.interactive:
        print("SMS Keyword Extractor  |  type 'quit' to exit\n")
        while True:
            sms = input("SMS > ").strip()
            if sms.lower() in ("quit", "exit", "q"):
                break
            keywords = extractor.extract_keywords(sms)
            print(f"Keywords: {keywords}\n")
    else:
        keywords = extractor.extract_keywords(args.text)
        print(f"SMS      : {args.text}")
        print(f"Keywords : {keywords}")


if __name__ == "__main__":
    main()
