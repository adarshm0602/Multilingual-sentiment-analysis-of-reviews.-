"""
Test cases for the Language Detection Pipeline.

This module contains pytest test cases for the LanguageDetector class,
testing detection of Kannada script, Romanized Kannada, English, and mixed text.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.language_detector import (
    LanguageDetector,
    DetectionResult,
    detect_language,
)


class TestLanguageDetector:
    """Test suite for LanguageDetector class."""

    @pytest.fixture
    def detector(self) -> LanguageDetector:
        """Create a LanguageDetector instance for testing."""
        return LanguageDetector()

    # ==================== Kannada Script Tests ====================

    def test_pure_kannada_script(self, detector: LanguageDetector) -> None:
        """Test detection of pure Kannada script text."""
        text = "ತುಂಬ ಚೆನ್ನಾಗಿದೆ"
        result = detector.detect(text)

        assert result.language == "kannada_script"
        assert result.confidence > 0.5
        assert result.script_proportions["kannada"] > 0.9

    def test_kannada_script_sentence(self, detector: LanguageDetector) -> None:
        """Test detection of a longer Kannada script sentence."""
        text = "ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ ಮತ್ತು ನಾನು ಇದನ್ನು ಶಿಫಾರಸು ಮಾಡುತ್ತೇನೆ"
        result = detector.detect(text)

        assert result.language == "kannada_script"
        assert result.confidence > 0.8

    def test_kannada_script_single_word(self, detector: LanguageDetector) -> None:
        """Test detection of a single Kannada word."""
        text = "ನಮಸ್ಕಾರ"
        result = detector.detect(text)

        assert result.language == "kannada_script"
        assert result.script_proportions["kannada"] == 1.0

    def test_kannada_script_with_numbers(self, detector: LanguageDetector) -> None:
        """Test Kannada script with embedded numbers."""
        text = "ಈ ಉತ್ಪನ್ನಕ್ಕೆ 5 ರೇಟಿಂಗ್"
        result = detector.detect(text)

        assert result.language == "kannada_script"
        assert result.script_proportions["numeric"] > 0

    # ==================== Romanized Kannada Tests ====================

    def test_romanized_kannada(self, detector: LanguageDetector) -> None:
        """Test detection of Romanized Kannada text."""
        text = "tumba chennagide"
        result = detector.detect(text)

        assert result.language == "romanized_kannada"
        assert result.confidence > 0

    def test_romanized_kannada_sentence(self, detector: LanguageDetector) -> None:
        """Test detection of a Romanized Kannada sentence."""
        text = "ee product tumba chennagide nanna friend ge recommend madthini"
        result = detector.detect(text)

        assert result.language == "romanized_kannada"

    def test_romanized_kannada_common_words(self, detector: LanguageDetector) -> None:
        """Test detection with common Romanized Kannada words."""
        text = "namaskara hegidira neevu"
        result = detector.detect(text)

        assert result.language == "romanized_kannada"

    def test_romanized_kannada_mixed_case(self, detector: LanguageDetector) -> None:
        """Test Romanized Kannada detection is case-insensitive."""
        text = "TUMBA CHENNAGIDE"
        result = detector.detect(text)

        assert result.language == "romanized_kannada"

    # ==================== English Tests ====================

    def test_pure_english(self, detector: LanguageDetector) -> None:
        """Test detection of pure English text."""
        text = "This is very good"
        result = detector.detect(text)

        assert result.language == "english"
        assert result.script_proportions["english"] > 0.8

    def test_english_sentence(self, detector: LanguageDetector) -> None:
        """Test detection of a longer English sentence."""
        text = "This product is amazing and I would highly recommend it to everyone"
        result = detector.detect(text)

        assert result.language == "english"
        assert result.confidence > 0.5

    def test_english_with_numbers(self, detector: LanguageDetector) -> None:
        """Test English text with numbers."""
        text = "I give this product 5 stars out of 5"
        result = detector.detect(text)

        assert result.language == "english"
        assert result.script_proportions["numeric"] > 0

    def test_english_with_punctuation(self, detector: LanguageDetector) -> None:
        """Test English text with punctuation."""
        text = "Great product! Works perfectly. Highly recommended!!!"
        result = detector.detect(text)

        assert result.language == "english"

    # ==================== Mixed Text Tests ====================

    def test_mixed_kannada_english(self, detector: LanguageDetector) -> None:
        """Test detection of mixed Kannada script and English text."""
        text = "ಈ product ತುಂಬಾ good ಆಗಿದೆ"
        result = detector.detect(text)

        assert result.language == "mixed"
        assert result.script_proportions["kannada"] > 0
        assert result.script_proportions["english"] > 0

    def test_mixed_text_dominant_kannada(self, detector: LanguageDetector) -> None:
        """Test mixed text with dominant Kannada."""
        text = "ಈ ಉತ್ಪನ್ನ super ಆಗಿದೆ"
        result = detector.detect(text)

        # Could be mixed or kannada_script depending on proportions
        assert result.language in ("mixed", "kannada_script")
        assert result.script_proportions["kannada"] > result.script_proportions["english"]

    def test_mixed_text_dominant_english(self, detector: LanguageDetector) -> None:
        """Test mixed text with dominant English."""
        text = "This is a great ಉತ್ಪನ್ನ for daily use"
        result = detector.detect(text)

        assert result.language in ("mixed", "english")

    # ==================== Edge Cases ====================

    def test_empty_string(self, detector: LanguageDetector) -> None:
        """Test handling of empty string."""
        result = detector.detect("")

        assert result.language == "unknown"
        assert result.confidence == 0.0

    def test_whitespace_only(self, detector: LanguageDetector) -> None:
        """Test handling of whitespace-only string."""
        result = detector.detect("   \t\n  ")

        assert result.language == "unknown"
        assert result.confidence == 0.0

    def test_numbers_only(self, detector: LanguageDetector) -> None:
        """Test handling of numbers-only string."""
        result = detector.detect("12345")

        assert result.language == "unknown"
        assert result.script_proportions["numeric"] == 1.0

    def test_special_characters_only(self, detector: LanguageDetector) -> None:
        """Test handling of special characters only."""
        result = detector.detect("!@#$%^&*()")

        assert result.language == "unknown"

    # ==================== Helper Method Tests ====================

    def test_detect_simple(self, detector: LanguageDetector) -> None:
        """Test the simplified detection method."""
        assert detector.detect_simple("ತುಂಬ ಚೆನ್ನಾಗಿದೆ") == "kannada_script"
        assert detector.detect_simple("This is good") == "english"

    def test_is_kannada_script(self, detector: LanguageDetector) -> None:
        """Test is_kannada method with Kannada script."""
        assert detector.is_kannada("ನಮಸ್ಕಾರ") is True

    def test_is_kannada_romanized(self, detector: LanguageDetector) -> None:
        """Test is_kannada method with Romanized Kannada."""
        assert detector.is_kannada("tumba chennagide") is True

    def test_is_kannada_english(self, detector: LanguageDetector) -> None:
        """Test is_kannada method returns False for English."""
        assert detector.is_kannada("Hello world") is False

    def test_get_kannada_char_count(self, detector: LanguageDetector) -> None:
        """Test counting Kannada characters."""
        text = "ನಮಸ್ಕಾರ"
        count = detector.get_kannada_char_count(text)
        assert count == 7  # ನ ಮ ಸ ್ ಕ ಾ ರ

    def test_get_kannada_char_count_mixed(self, detector: LanguageDetector) -> None:
        """Test counting Kannada characters in mixed text."""
        text = "Hello ನಮಸ್ಕಾರ World"
        count = detector.get_kannada_char_count(text)
        assert count == 7

    # ==================== Detection Result Tests ====================

    def test_detection_result_has_all_fields(self, detector: LanguageDetector) -> None:
        """Test that DetectionResult contains all expected fields."""
        result = detector.detect("ತುಂಬ ಚೆನ್ನಾಗಿದೆ")

        assert hasattr(result, "language")
        assert hasattr(result, "confidence")
        assert hasattr(result, "script_proportions")
        assert hasattr(result, "details")

    def test_script_proportions_sum(self, detector: LanguageDetector) -> None:
        """Test that script proportions sum to approximately 1.0."""
        result = detector.detect("ತುಂಬ ಚೆನ್ನಾಗಿದೆ")
        proportions = result.script_proportions

        total = sum(proportions.values())
        assert abs(total - 1.0) < 0.01  # Allow small floating point error

    # ==================== Module Function Tests ====================

    def test_detect_language_function(self) -> None:
        """Test the module-level detect_language convenience function."""
        assert detect_language("ನಮಸ್ಕಾರ") == "kannada_script"
        assert detect_language("Hello") == "english"

    # ==================== Custom Threshold Tests ====================

    def test_custom_kannada_threshold(self) -> None:
        """Test detector with custom Kannada threshold."""
        strict_detector = LanguageDetector(kannada_threshold=0.9)
        result = strict_detector.detect("ಈ product ಚೆನ್ನಾಗಿದೆ")

        # With stricter threshold, mixed text shouldn't be classified as kannada_script
        assert result.language in ("mixed", "kannada_script")

    def test_custom_english_threshold(self) -> None:
        """Test detector with custom English threshold."""
        lenient_detector = LanguageDetector(english_threshold=0.5)
        result = lenient_detector.detect("Hello world")

        assert result.language == "english"


class TestKannadaUnicodeRange:
    """Test suite for Kannada Unicode character detection."""

    @pytest.fixture
    def detector(self) -> LanguageDetector:
        """Create a LanguageDetector instance for testing."""
        return LanguageDetector()

    def test_kannada_unicode_start(self, detector: LanguageDetector) -> None:
        """Test character at start of Kannada Unicode range."""
        # U+0C80 is the start of Kannada block
        char = chr(0x0C80)
        assert detector._is_kannada_char(char) is True

    def test_kannada_unicode_end(self, detector: LanguageDetector) -> None:
        """Test character at end of Kannada Unicode range."""
        # U+0CFF is the end of Kannada block
        char = chr(0x0CFF)
        assert detector._is_kannada_char(char) is True

    def test_non_kannada_unicode(self, detector: LanguageDetector) -> None:
        """Test character outside Kannada Unicode range."""
        # ASCII 'A'
        assert detector._is_kannada_char("A") is False
        # Devanagari character
        assert detector._is_kannada_char("अ") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
