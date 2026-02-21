"""
Preprocessing module for Kannada Sentiment Analysis.

This module provides text preprocessing utilities including
language detection, transliteration, translation, text normalization, and script handling.
"""

from .language_detector import LanguageDetector, DetectionResult, detect_language
from .transliterator import (
    Transliterator,
    TransliterationResult,
    TransliterationError,
    transliterate_to_kannada,
)
from .translator import (
    Translator,
    TranslationResult,
    TranslationError,
    TranslationBackend,
    translate_kannada_to_english,
    load_config_translator,
)

__all__ = [
    # Language Detection
    "LanguageDetector",
    "DetectionResult",
    "detect_language",
    # Transliteration
    "Transliterator",
    "TransliterationResult",
    "TransliterationError",
    "transliterate_to_kannada",
    # Translation
    "Translator",
    "TranslationResult",
    "TranslationError",
    "TranslationBackend",
    "translate_kannada_to_english",
    "load_config_translator",
]
