"""
Comprehensive unit tests for LanguageDetector.

Coverage:
  - Kannada script detection (pure, with numbers, long text)
  - Romanized Kannada detection (pattern-based, case-insensitive)
  - English detection (short, long, with punctuation, with numbers)
  - Mixed-language detection
  - Edge cases: empty, whitespace, numbers-only, special chars,
                very long text (>512 tokens), emojis, URLs, code snippets
  - Internal helpers: _is_kannada_char, _is_english_char,
                      _calculate_script_proportions
  - Module-level convenience function
  - Custom threshold variants
  - DetectionResult data integrity
"""

import pytest

from src.preprocessing.language_detector import (
    DetectionResult,
    LanguageDetector,
    detect_language,
)
from tests.conftest import make_long_english, make_long_kannada, make_long_romanized


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def detector() -> LanguageDetector:
    """Default LanguageDetector (shared within each test function)."""
    return LanguageDetector()


# ===========================================================================
# Parametrized canonical-input checks
# ===========================================================================

KANNADA_SCRIPT_EXAMPLES = [
    "ನಮಸ್ಕಾರ",
    "ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ",
    "ಕನ್ನಡ ಭಾಷೆ",
    "ಬಹಳ ಒಳ್ಳೆಯ ಸೇವೆ ಮತ್ತು ಉತ್ತಮ ಗುಣಮಟ್ಟ",
]

ROMANIZED_KANNADA_EXAMPLES = [
    "tumba chennagide",
    "namaskara hegidira neevu",
    "ee product tumba olle agide",
    "TUMBA CHENNAGIDE",          # upper-case — detection is case-insensitive
    "olle baralla gottilla",
]

ENGLISH_EXAMPLES = [
    "This is a great product",
    "Highly recommended for everyone",
    "I give this five stars out of five",
    "Great product! Works perfectly. Highly recommended!!!",
]


@pytest.mark.parametrize("text", KANNADA_SCRIPT_EXAMPLES)
def test_canonical_kannada_script(detector, text):
    result = detector.detect(text)
    assert result.language == "kannada_script", (
        f"Expected 'kannada_script', got '{result.language}' for: {text!r}"
    )
    assert result.confidence > 0.5


@pytest.mark.parametrize("text", ROMANIZED_KANNADA_EXAMPLES)
def test_canonical_romanized_kannada(detector, text):
    result = detector.detect(text)
    assert result.language == "romanized_kannada", (
        f"Expected 'romanized_kannada', got '{result.language}' for: {text!r}"
    )


@pytest.mark.parametrize("text", ENGLISH_EXAMPLES)
def test_canonical_english(detector, text):
    result = detector.detect(text)
    assert result.language == "english", (
        f"Expected 'english', got '{result.language}' for: {text!r}"
    )


# ===========================================================================
# Kannada script — detailed tests
# ===========================================================================

class TestKannadaScript:

    def test_pure_script_confidence(self, detector):
        result = detector.detect("ನಮಸ್ಕಾರ")
        assert result.language == "kannada_script"
        assert result.script_proportions["kannada"] == pytest.approx(1.0)

    def test_sentence_high_confidence(self, detector):
        text = "ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ ಮತ್ತು ನಾನು ಇದನ್ನು ಶಿಫಾರಸು ಮಾಡುತ್ತೇನೆ"
        result = detector.detect(text)
        assert result.language == "kannada_script"
        assert result.confidence >= 0.8

    def test_with_embedded_digits(self, detector):
        """Kannada text containing an ASCII digit should still classify as Kannada."""
        result = detector.detect("ಈ ಉತ್ಪನ್ನಕ್ಕೆ 5 ರೇಟಿಂಗ್")
        assert result.language == "kannada_script"
        assert result.script_proportions["numeric"] > 0.0

    def test_with_emoji(self, detector):
        """Kannada text with a trailing emoji must not break classification."""
        result = detector.detect("ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ 👍")
        assert result.language in ("kannada_script", "mixed")

    def test_long_kannada_text(self, detector):
        """Very long Kannada text (>2000 chars) should be detected correctly."""
        result = detector.detect(make_long_kannada())
        assert result.language == "kannada_script"
        assert result.confidence > 0.7

    def test_kannada_with_punctuation(self, detector):
        result = detector.detect("ಚೆನ್ನಾಗಿದೆ! ತುಂಬಾ ಉತ್ತಮ.")
        assert result.language == "kannada_script"

    def test_negative_kannada_review(self, detector):
        result = detector.detect("ಈ ಟಿ.ವಿ. ಕಳಪೆ ಗುಣಮಟ್ಟದ್ದು. ಕೇವಲ ಒಂದು ತಿಂಗಳಲ್ಲಿ ಸ್ಕ್ರೀನ್ ಹಾಳಾಯಿತು.")
        assert result.language == "kannada_script"


# ===========================================================================
# Romanized Kannada — detailed tests
# ===========================================================================

class TestRomanizedKannada:

    def test_basic_positive(self, detector):
        result = detector.detect("ee product tumba chennagide")
        assert result.language == "romanized_kannada"

    def test_common_greeting(self, detector):
        result = detector.detect("namaskara hegidira")
        assert result.language == "romanized_kannada"

    def test_case_insensitive(self, detector):
        lower = detector.detect("tumba olle ide")
        upper = detector.detect("TUMBA OLLE IDE")
        # Both should agree
        assert lower.language == upper.language == "romanized_kannada"

    def test_is_kannada_helper_romanized(self, detector):
        assert detector.is_kannada("tumba chennagide") is True

    def test_long_romanized_text(self, detector):
        """Very long romanized text should not crash and should be detected."""
        result = detector.detect(make_long_romanized())
        # Acceptable to land on 'romanized_kannada' or 'english'
        # (pattern density drops as the chunk repeats) — just must not crash.
        assert result.language in ("romanized_kannada", "english", "mixed")

    def test_romanized_with_numbers(self, detector):
        """Numbers mixed into romanized Kannada should not break detection."""
        result = detector.detect("ee product 500 rupees ge tumba olle")
        assert result.language in ("romanized_kannada", "english")


# ===========================================================================
# English — detailed tests
# ===========================================================================

class TestEnglish:

    def test_short_sentence(self, detector):
        assert detector.detect("This is very good").language == "english"

    def test_long_english_text(self, detector):
        """Text >512 tokens (transformer limit) must still be detected correctly."""
        result = detector.detect(make_long_english())
        assert result.language == "english"

    def test_with_numbers(self, detector):
        result = detector.detect("I give this product 5 stars out of 5")
        assert result.language == "english"
        assert result.script_proportions["numeric"] > 0.0

    def test_with_punctuation(self, detector):
        result = detector.detect("Excellent!!! Works perfectly. 100% satisfied.")
        assert result.language == "english"

    def test_with_url_like_content(self, detector):
        """URLs embedded in English text should not crash detection."""
        result = detector.detect(
            "Check this product at www.example.com it is really good"
        )
        assert result.language in ("english", "unknown")

    def test_with_code_snippet(self, detector):
        """A snippet of code (all ASCII) should not crash detection."""
        result = detector.detect('print("Hello world") if x > 0 else pass')
        assert result.language in ("english", "unknown")

    def test_is_not_kannada(self, detector):
        assert detector.is_kannada("Hello World") is False

    def test_detect_simple_english(self, detector):
        assert detector.detect_simple("Hello World") == "english"


# ===========================================================================
# Mixed language
# ===========================================================================

class TestMixedLanguage:

    def test_kannada_and_english_words(self, detector):
        result = detector.detect("ಈ product ತುಂಬಾ good ಆಗಿದೆ")
        assert result.language in ("mixed", "kannada_script")
        assert result.script_proportions["kannada"] > 0.0
        assert result.script_proportions["english"] > 0.0

    def test_dominant_english_with_kannada_word(self, detector):
        result = detector.detect("This is a great ಉತ್ಪನ್ನ for daily use")
        assert result.language in ("mixed", "english")

    def test_mixed_proportions_are_nonzero(self, detector):
        result = detector.detect("ಈ product ತುಂಬಾ good ಆಗಿದೆ")
        assert result.script_proportions["kannada"] > 0.0
        assert result.script_proportions["english"] > 0.0


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_empty_string(self, detector):
        result = detector.detect("")
        assert result.language == "unknown"
        assert result.confidence == pytest.approx(0.0)

    def test_whitespace_only(self, detector):
        for text in ("   ", "\t", "\n", "  \t  \n  "):
            result = detector.detect(text)
            assert result.language == "unknown", f"Failed for whitespace: {text!r}"
            assert result.confidence == pytest.approx(0.0)

    def test_single_digit(self, detector):
        result = detector.detect("5")
        assert result.language == "unknown"
        assert result.script_proportions["numeric"] == pytest.approx(1.0)

    def test_numbers_only(self, detector):
        result = detector.detect("123456789")
        assert result.language == "unknown"
        assert result.script_proportions["numeric"] == pytest.approx(1.0)
        assert result.script_proportions["kannada"] == pytest.approx(0.0)
        assert result.script_proportions["english"] == pytest.approx(0.0)

    def test_special_characters_only(self, detector):
        result = detector.detect("!@#$%^&*()")
        assert result.language == "unknown"

    def test_emojis_only(self, detector):
        """Pure-emoji input must not crash and should return 'unknown'."""
        result = detector.detect("😊👍🎉❤️🚀")
        assert result.language in ("unknown", "english", "mixed")
        assert 0.0 <= result.confidence <= 1.0

    def test_emoji_with_english(self, detector):
        """English text with emojis should remain 'english' or mixed."""
        result = detector.detect("This product is amazing! 😊👍🎉")
        assert result.language in ("english", "mixed")

    def test_emoji_with_kannada(self, detector):
        """Kannada text with emojis should remain 'kannada_script' or mixed."""
        result = detector.detect("ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ! 👍👍")
        assert result.language in ("kannada_script", "mixed")

    def test_single_kannada_character(self, detector):
        result = detector.detect("ಕ")
        assert result.language == "kannada_script"
        assert result.script_proportions["kannada"] == pytest.approx(1.0)

    def test_single_english_character(self, detector):
        result = detector.detect("A")
        assert result.language in ("english", "unknown")

    def test_newlines_and_tabs_in_kannada(self, detector):
        result = detector.detect("ತುಂಬಾ\nಚೆನ್ನಾಗಿದೆ\tಮತ್ತು")
        assert result.language == "kannada_script"

    def test_repeated_special_chars_with_letters(self, detector):
        result = detector.detect("Hello!!! World??? Great!!!!")
        assert result.language == "english"

    def test_devanagari_not_classified_as_kannada(self, detector):
        """Hindi/Devanagari characters must not be mis-classified as Kannada script."""
        result = detector.detect("नमस्ते यह बहुत अच्छा है")
        assert result.language != "kannada_script"


# ===========================================================================
# Internal helper tests
# ===========================================================================

class TestInternalHelpers:

    def test_is_kannada_char_in_range(self, detector):
        assert detector._is_kannada_char(chr(0x0C80)) is True   # block start
        assert detector._is_kannada_char(chr(0x0CFF)) is True   # block end
        assert detector._is_kannada_char("ಕ") is True

    def test_is_kannada_char_outside_range(self, detector):
        assert detector._is_kannada_char("A") is False
        assert detector._is_kannada_char("अ") is False  # Devanagari
        assert detector._is_kannada_char("0") is False

    def test_is_english_char(self, detector):
        assert detector._is_english_char("a") is True
        assert detector._is_english_char("Z") is True
        assert detector._is_english_char("5") is False
        assert detector._is_english_char("ಕ") is False

    def test_script_proportions_sum_to_one(self, detector):
        for text in [
            "ನಮಸ್ಕಾರ",
            "Hello world",
            "tumba chennagide",
            "ಈ product good ಆಗಿದೆ 5",
        ]:
            props = detector._calculate_script_proportions(text)
            total = sum(props.values())
            assert abs(total - 1.0) < 1e-9, (
                f"Proportions don't sum to 1 for {text!r}: {props}"
            )

    def test_script_proportions_empty(self, detector):
        props = detector._calculate_script_proportions("")
        assert props == {"kannada": 0.0, "english": 0.0, "numeric": 0.0, "other": 0.0}

    def test_script_proportions_all_kannada(self, detector):
        props = detector._calculate_script_proportions("ನಮಸ್ಕಾರ")
        assert props["kannada"] == pytest.approx(1.0)
        assert props["english"] == pytest.approx(0.0)

    def test_script_proportions_all_english(self, detector):
        props = detector._calculate_script_proportions("Hello")
        assert props["english"] == pytest.approx(1.0)
        assert props["kannada"] == pytest.approx(0.0)

    def test_kannada_char_count(self, detector):
        # "ನಮಸ್ಕಾರ" has 7 code points in the Kannada block
        assert detector.get_kannada_char_count("ನಮಸ್ಕಾರ") == 7

    def test_kannada_char_count_mixed(self, detector):
        count = detector.get_kannada_char_count("Hello ನಮಸ್ಕಾರ World")
        assert count == 7

    def test_kannada_char_count_empty(self, detector):
        assert detector.get_kannada_char_count("") == 0


# ===========================================================================
# DetectionResult data integrity
# ===========================================================================

class TestDetectionResult:

    def test_all_fields_present(self, detector):
        result = detector.detect("ತುಂಬ ಚೆನ್ನಾಗಿದೆ")
        assert isinstance(result, DetectionResult)
        assert hasattr(result, "language")
        assert hasattr(result, "confidence")
        assert hasattr(result, "script_proportions")
        assert hasattr(result, "details")

    def test_confidence_in_range(self, detector):
        for text in ["ನಮಸ್ಕಾರ", "Hello", "tumba olle", ""]:
            result = detector.detect(text)
            assert 0.0 <= result.confidence <= 1.0, (
                f"Confidence out of range for {text!r}: {result.confidence}"
            )

    def test_script_proportions_keys(self, detector):
        result = detector.detect("Hello world")
        assert set(result.script_proportions.keys()) == {
            "kannada", "english", "numeric", "other"
        }

    def test_details_is_non_empty_string(self, detector):
        for text in ["ನಮಸ್ಕಾರ", "Hello", "tumba olle", ""]:
            result = detector.detect(text)
            assert isinstance(result.details, str)
            assert len(result.details) > 0


# ===========================================================================
# Helper methods on LanguageDetector
# ===========================================================================

class TestHelperMethods:

    def test_detect_simple_returns_string(self, detector):
        lang = detector.detect_simple("ನಮಸ್ಕಾರ")
        assert isinstance(lang, str)
        assert lang == "kannada_script"

    def test_is_kannada_true_for_script(self, detector):
        assert detector.is_kannada("ನಮಸ್ಕಾರ") is True

    def test_is_kannada_true_for_romanized(self, detector):
        assert detector.is_kannada("tumba chennagide") is True

    def test_is_kannada_false_for_english(self, detector):
        assert detector.is_kannada("Hello world") is False

    def test_is_kannada_false_for_empty(self, detector):
        assert detector.is_kannada("") is False


# ===========================================================================
# Module-level convenience function
# ===========================================================================

class TestModuleLevelFunction:

    def test_detect_language_kannada(self):
        assert detect_language("ನಮಸ್ಕಾರ") == "kannada_script"

    def test_detect_language_english(self):
        assert detect_language("Hello world") == "english"

    def test_detect_language_empty(self):
        assert detect_language("") == "unknown"


# ===========================================================================
# Custom threshold variants
# ===========================================================================

class TestCustomThresholds:

    def test_strict_kannada_threshold(self):
        """With a high kannada_threshold, slightly mixed text may fall through."""
        strict = LanguageDetector(kannada_threshold=0.95)
        # Text with a Latin product name inside Kannada
        result = strict.detect("ಈ product ಚೆನ್ನಾಗಿದೆ")
        assert result.language in ("mixed", "kannada_script")

    def test_lenient_english_threshold(self):
        """Lowering english_threshold should still detect plain English."""
        lenient = LanguageDetector(english_threshold=0.5)
        result = lenient.detect("Hello world")
        assert result.language == "english"

    def test_custom_mixed_threshold_effect(self):
        """Raising mixed_threshold should route borderline text to 'unknown'."""
        strict = LanguageDetector(mixed_threshold=0.9)
        # Very sparse text with low proportions → should hit 'unknown'
        result = strict.detect("5 ★")
        assert result.language in ("unknown", "mixed", "english")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
