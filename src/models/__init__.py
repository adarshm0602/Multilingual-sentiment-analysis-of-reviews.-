"""
Models module for Kannada Sentiment Analysis.

This module provides model downloading, loading, sentiment classification,
and training functionality using pre-trained transformer models.
"""

from .download_model import (
    ModelDownloader,
    ModelInfo,
    ModelType,
    AVAILABLE_MODELS,
    DEFAULT_MODELS,
)

from .sentiment_classifier import (
    SentimentClassifier,
    SentimentResult,
    SentimentLabel,
    classify_sentiment,
    load_classifier_from_config,
)

from .train_sentiment import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    train_sentiment_model,
)

__all__ = [
    # Model downloading
    "ModelDownloader",
    "ModelInfo",
    "ModelType",
    "AVAILABLE_MODELS",
    "DEFAULT_MODELS",
    # Sentiment classification
    "SentimentClassifier",
    "SentimentResult",
    "SentimentLabel",
    "classify_sentiment",
    "load_classifier_from_config",
    # Training
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "train_sentiment_model",
]
