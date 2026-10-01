from inference.predict import SMSKeywordExtractor


def test_typo_normalization_for_common_sms_errors():
    extractor = SMSKeywordExtractor.__new__(SMSKeywordExtractor)
    extractor.keyword_lexicon = {
        "otp",
        "valid",
        "minutes",
        "amount",
        "balance",
        "card",
        "payment",
        "transaction",
    }

    normalized = extractor._normalize_tokens("pls check ur otp and amout validte")

    assert "otp" in normalized
    assert "amount" in normalized
    assert "validate" in normalized or "valid" in normalized


def test_fast_path_returns_lexicon_keywords_without_model_inference():
    extractor = SMSKeywordExtractor.__new__(SMSKeywordExtractor)
    extractor.keyword_lexicon = {"otp", "amount", "valid", "card"}
    extractor._fast_path_length = 10

    result = extractor._fast_path_extract_keywords("your otp is valid for amount")

    assert "otp" in result
    assert "valid" in result


def test_stopword_filtering_and_typo_repair_for_token_fragments():
    extractor = SMSKeywordExtractor.__new__(SMSKeywordExtractor)
    extractor.keyword_lexicon = {"otp", "valid", "minutes", "bill", "due", "tomorrow", "weather"}

    repaired = extractor._repair_candidate_token("thet")
    assert repaired == "the"
    assert extractor._repair_candidate_token("isn") == "is"

    cleaned = extractor._filter_keyword_tokens(["is", "otp", "isn", "valid", "the", "minutes", "a"])
    assert "is" not in cleaned
    assert "the" not in cleaned
    assert "a" not in cleaned
    assert "otp" in cleaned
    assert "valid" in cleaned
    assert "minutes" in cleaned


def test_topic_extraction_returns_coherent_topics_for_notifications():
    extractor = SMSKeywordExtractor.__new__(SMSKeywordExtractor)

    topics = extractor.extract_topics("Ferns City | Storms – Thunderstorms possible in the next few hours")
    assert "Ferns City" in topics
    assert "Thunderstorms" in topics or "weather" in topics

    topics2 = extractor.extract_topics("Greater Bengaluru Authority replaces BBMP")
    assert "BBMP" in topics2
    assert "Bengaluru" in topics2
    assert "politics" in topics2 or "Greater Bengaluru Authority" in topics2
