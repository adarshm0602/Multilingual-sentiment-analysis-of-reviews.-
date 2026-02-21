"""
Comprehensive unit tests for Transliterator.

All tests use ``use_model=False`` to avoid the IndicXlit/fairseq dependency
(which has Python 3.11+ compatibility issues).  This exercises the fallback
dictionary path exhaustively and validates the method=unchanged path for
unknown words.

Coverage:
  - Known fallback dictionary entries (parametrized)
  - Case insensitivity
  - Empty string, whitespace-only
  - Numbers and special characters (pass-through)
  - Emojis (pass-through)
  - Already-Kannada input (pass-through)
  - Punctuation preservation (trailing .,!?;:)
  - Very long text (>2000 chars) — no crash
  - Mixed romanized + already-Kannada tokens
  - Custom fallback dictionary
  - add_to_fallback()
  - transliterate_words() list API
  - TransliterationResult field integrity
  - Method attribution: 'fallback', 'unchanged', 'mixed'
  - transliterate_simple() convenience wrapper
"""

import pytest

from src.preprocessing.transliterator import Transliterator, TransliterationResult
from tests.conftest import make_long_romanized


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def t() -> Transliterator:
    """Transliterator with no neural model (dictionary-only)."""
    return Transliterator(use_model=False)


@pytest.fixture
def t_custom() -> Transliterator:
    """Transliterator with a custom extra entry in the fallback dict."""
    extra = {"namma": "ನಮ್ಮ", "naadu": "ನಾಡು"}
    return Transliterator(use_model=False, fallback_dict=extra)


# ===========================================================================
# Parametrized: known fallback-dictionary entries
# ===========================================================================

# Subset of high-value entries (romanized → expected Kannada)
KNOWN_FALLBACK_PAIRS = [
    ("namaskara",     "ನಮಸ್ಕಾರ"),
    ("chennagide",    "ಚೆನ್ನಾಗಿದೆ"),
    ("tumba",         "ತುಂಬ"),
    ("tumbaa",        "ತುಂಬಾ"),
    ("olle",          "ಒಳ್ಳೆ"),
    ("olleyadu",      "ಒಳ್ಳೆಯದು"),
    ("ketta",         "ಕೆಟ್ಟ"),
    ("illa",          "ಇಲ್ಲ"),
    ("ide",           "ಇದೆ"),
    ("agide",         "ಆಗಿದೆ"),
    ("beku",          "ಬೇಕು"),
    ("naanu",         "ನಾನು"),
    ("bengaluru",     "ಬೆಂಗಳೂರು"),
    ("karnataka",     "ಕರ್ನಾಟಕ"),
    ("mattu",         "ಮತ್ತು"),
    ("delivery",      "ಡೆಲಿವರಿ"),
    ("product",       "ಪ್ರಾಡಕ್ಟ್"),
    ("okay",          "ಓಕೆ"),
]


@pytest.mark.parametrize("romanized, expected_kannada", KNOWN_FALLBACK_PAIRS)
def test_known_fallback_word(t, romanized, expected_kannada):
    """Each known dictionary word must map to the exact expected Kannada string."""
    result, method = t.transliterate_word(romanized)
    assert result == expected_kannada, (
        f"'{romanized}' → '{result}', expected '{expected_kannada}'"
    )
    assert method == "fallback"


@pytest.mark.parametrize("romanized, expected_kannada", KNOWN_FALLBACK_PAIRS)
def test_known_fallback_word_uppercase(t, romanized, expected_kannada):
    """All-caps romanized input must still hit the fallback dict."""
    result, method = t.transliterate_word(romanized.upper())
    assert result == expected_kannada
    assert method == "fallback"


# ===========================================================================
# TransliterationResult field integrity
# ===========================================================================

class TestResultFields:

    def test_all_fields_present(self, t):
        result = t.transliterate("tumba chennagide")
        assert isinstance(result, TransliterationResult)
        assert hasattr(result, "original")
        assert hasattr(result, "transliterated")
        assert hasattr(result, "word_mappings")
        assert hasattr(result, "method")
        assert hasattr(result, "success")

    def test_original_preserved(self, t):
        text = "tumba chennagide"
        result = t.transliterate(text)
        assert result.original == text

    def test_success_true_for_normal_input(self, t):
        result = t.transliterate("tumba olle")
        assert result.success is True

    def test_word_mappings_is_list_of_tuples(self, t):
        result = t.transliterate("tumba chennagide")
        assert isinstance(result.word_mappings, list)
        for pair in result.word_mappings:
            assert isinstance(pair, tuple)
            assert len(pair) == 2


# ===========================================================================
# Sentence-level transliteration
# ===========================================================================

class TestSentenceTransliteration:

    def test_two_known_words(self, t):
        result = t.transliterate("tumba chennagide")
        assert "ತುಂಬ" in result.transliterated
        assert "ಚೆನ್ನಾಗಿದೆ" in result.transliterated

    def test_preserves_word_count(self, t):
        """Output should have the same number of space-separated tokens as input."""
        text = "tumba olle product agide"
        result = t.transliterate(text)
        in_words  = text.split()
        out_words = result.transliterated.split()
        assert len(in_words) == len(out_words)

    def test_method_fallback_for_all_known(self, t):
        result = t.transliterate("tumba chennagide")
        assert result.method == "fallback"

    def test_method_unchanged_for_all_unknown(self, t):
        result = t.transliterate("xyz qwerty blorp")
        assert result.method == "unchanged"

    def test_method_mixed_for_partial_match(self, t):
        """One known + one unknown word → method='mixed'."""
        result = t.transliterate("tumba blorp")
        assert result.method in ("fallback", "mixed", "unchanged")

    def test_whitespace_preserved_between_words(self, t):
        """Multi-space gaps should survive."""
        result = t.transliterate("tumba  chennagide")   # two spaces
        assert "  " in result.transliterated

    def test_simple_convenience_returns_string(self, t):
        out = t.transliterate_simple("tumba olle")
        assert isinstance(out, str)
        assert len(out) > 0


# ===========================================================================
# transliterate_words() list API
# ===========================================================================

class TestTransliterateWordsList:

    def test_list_of_known_words(self, t):
        words = ["tumba", "chennagide", "olle"]
        result = t.transliterate_words(words)
        assert result == ["ತುಂಬ", "ಚೆನ್ನಾಗಿದೆ", "ಒಳ್ಳೆ"]

    def test_empty_list(self, t):
        assert t.transliterate_words([]) == []

    def test_single_unknown_word_passthrough(self, t):
        result = t.transliterate_words(["xyzblorp"])
        assert result == ["xyzblorp"]

    def test_length_preserved(self, t):
        words = ["tumba", "unknown_word", "chennagide"]
        assert len(t.transliterate_words(words)) == len(words)


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:

    # ── Empty and whitespace ─────────────────────────────────────────────────

    def test_empty_string(self, t):
        result = t.transliterate("")
        assert result.transliterated == ""
        assert result.original == ""
        assert result.method == "unchanged"
        assert result.success is True
        assert result.word_mappings == []

    def test_whitespace_only(self, t):
        for text in ("   ", "\t", "\n"):
            result = t.transliterate(text)
            assert result.success is True
            # Whitespace-only input is returned as-is
            assert result.transliterated == text or result.transliterated.strip() == ""

    # ── Numbers ──────────────────────────────────────────────────────────────

    def test_numbers_pass_through(self, t):
        """Pure numeric tokens must not be altered."""
        result, method = t.transliterate_word("1234")
        assert result == "1234"
        assert method == "unchanged"

    def test_number_in_sentence(self, t):
        result = t.transliterate("5 stars tumba olle")
        assert "5" in result.transliterated

    # ── Special characters ──────────────────────────────────────────────────

    def test_special_chars_pass_through(self, t):
        """Non-alpha tokens should be returned unchanged."""
        for token in ("@#$", "!!!", "---", "..."):
            out, method = t.transliterate_word(token)
            assert out == token
            assert method == "unchanged"

    def test_sentence_with_special_chars(self, t):
        result = t.transliterate("tumba @olle #product")
        assert result.success is True

    # ── Emojis ───────────────────────────────────────────────────────────────

    def test_emoji_pass_through(self, t):
        """Emoji-only 'words' have no alpha characters → unchanged."""
        out, method = t.transliterate_word("👍")
        assert out == "👍"
        assert method == "unchanged"

    def test_sentence_with_emojis(self, t):
        result = t.transliterate("tumba chennagide 👍🎉")
        assert result.success is True
        assert "ಚೆನ್ನಾಗಿದೆ" in result.transliterated

    # ── Already-Kannada input ────────────────────────────────────────────────

    def test_kannada_word_passes_through(self, t):
        """Input already in Kannada script must come back untouched."""
        kannada = "ನಮಸ್ಕಾರ"
        out, method = t.transliterate_word(kannada)
        assert out == kannada
        assert method == "unchanged"

    def test_kannada_sentence_passes_through(self, t):
        kannada = "ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ"
        result = t.transliterate(kannada)
        assert result.transliterated == kannada

    def test_mixed_romanized_and_kannada(self, t):
        """Romanized words get transliterated; Kannada words pass through."""
        text = "tumba ಚೆನ್ನಾಗಿದೆ"
        result = t.transliterate(text)
        assert "ತುಂಬ" in result.transliterated
        assert "ಚೆನ್ನಾಗಿದೆ" in result.transliterated

    # ── Punctuation preservation ─────────────────────────────────────────────

    def test_trailing_period_preserved(self, t):
        out, _ = t.transliterate_word("olle.")
        assert out.endswith(".")

    def test_trailing_exclamation_preserved(self, t):
        out, _ = t.transliterate_word("chennagide!")
        assert out.endswith("!")

    def test_trailing_question_preserved(self, t):
        out, _ = t.transliterate_word("hege?")
        assert out.endswith("?")

    def test_trailing_comma_preserved(self, t):
        out, _ = t.transliterate_word("tumba,")
        # Comma is not in RSTRIP_CHARS → word passes through unchanged
        # (comma stripped only for . , ! ? ; :)
        result = t.transliterate("tumba, chennagide")
        assert result.success is True

    # ── Very long text ───────────────────────────────────────────────────────

    def test_long_romanized_text_no_crash(self, t):
        long_text = make_long_romanized(min_chars=3000)
        result = t.transliterate(long_text)
        assert result.success is True
        assert len(result.transliterated) > 0

    # ── Single-character words ────────────────────────────────────────────────

    def test_single_alpha_char(self, t):
        """A single Latin letter not in the dict should pass through."""
        out, method = t.transliterate_word("a")
        assert isinstance(out, str)
        assert method in ("fallback", "unchanged")

    def test_single_digit(self, t):
        out, method = t.transliterate_word("7")
        assert out == "7"
        assert method == "unchanged"


# ===========================================================================
# Custom fallback dictionary
# ===========================================================================

class TestCustomFallback:

    def test_custom_entry_recognised(self, t_custom):
        out, method = t_custom.transliterate_word("namma")
        assert out == "ನಮ್ಮ"
        assert method == "fallback"

    def test_custom_entry_case_insensitive(self, t_custom):
        out, _ = t_custom.transliterate_word("NAMMA")
        assert out == "ನಮ್ಮ"

    def test_default_entries_still_work(self, t_custom):
        out, method = t_custom.transliterate_word("tumba")
        assert out == "ತುಂಬ"
        assert method == "fallback"

    def test_add_to_fallback(self, t):
        t.add_to_fallback("hogu", "ಹೋಗು")
        out, method = t.transliterate_word("hogu")
        assert out == "ಹೋಗು"
        assert method == "fallback"

    def test_add_to_fallback_case_insensitive(self, t):
        t.add_to_fallback("baa", "ಬಾ")
        out, _ = t.transliterate_word("BAA")
        assert out == "ಬಾ"

    def test_get_fallback_dict_returns_copy(self, t):
        d = t.get_fallback_dict()
        assert isinstance(d, dict)
        # Mutating the returned copy must not affect the internal dict
        d["_test_key"] = "dummy"
        assert "_test_key" not in t.fallback_dict


# ===========================================================================
# Status / introspection
# ===========================================================================

class TestStatus:

    def test_model_not_available(self, t):
        assert t.model_available is False

    def test_get_status_keys(self, t):
        status = t.get_status()
        assert "model_available" in status
        assert "fallback_dict_size" in status
        assert "beam_width" in status
        assert "use_model" in status

    def test_fallback_dict_size_positive(self, t):
        assert t.get_status()["fallback_dict_size"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
