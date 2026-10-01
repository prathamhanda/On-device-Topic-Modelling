"""Build token-classification models for SMS keyword extraction."""
from __future__ import annotations

from transformers import AutoConfig, AutoModelForTokenClassification
from transformers.modeling_utils import PreTrainedModel

from model.config import ID2LABEL, LABEL2ID


def build_model(model_name: str, num_labels: int) -> PreTrainedModel:
    """Load a pre-trained Transformer with a token-classification head.

    The original project used ``MobileBertForTokenClassification`` directly.
    ``AutoModelForTokenClassification`` keeps MobileBERT support and also enables
    larger profiles such as DistilBERT and BERT-base without changing training
    code.
    """
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )
    return model
