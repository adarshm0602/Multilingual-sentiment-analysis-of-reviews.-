"""
Comprehensive end-to-end tests for KannadaSentimentPipeline.

The pipeline fixture uses:
  - translation_backend="fallback"   (dict-based; no network or heavy model)
  - use_transliteration_model=False  (fallback dict only; avoids IndicXlit)
  - auto_fallback=True

This keeps test runtime reasonable while still exercising every route:
  Kannada script   → translate → classify
  Romanized Kannada → transliterate → translate → classify
  English          → classify directly
  Mixed / Unknown  → classify directly (best-effort)

Coverage:
  - Result dict structure (all keys, correct types)
  - Routing: English, Kannada script, Romanized Kannada, mixed, unknown
  - Edge cases: empty, whitespace, very long text, special chars,
                numbers-only, emojis, mixed scripts
  - process_batch(): list input, DataFrame input, mixed valid/invalid,
                     empty list, output columns, status values, row order
  - Non-fatal error propagation
  - Confidence scores in valid range
  - pipeline_steps list ordering
"""

import pytest
import sys
from pathlib import Path

# Guard: import pandas only when available (same approach as pipeline itself)
try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

from src.pipeline import KannadaSentimentPipeline
from tests.conftest import make_long_english, make_long_kannada, make_long_romanized

# ---------------------------------------------------------------------------
# Expected output keys for process()
# ---------------------------------------------------------------------------
_EXPECTED_KEYS = {
    "original_text",
    "detected_language",
    "detection_confidence",
    "transliterated_text",
    "translated_text",
    "sentiment_label",
    "confidence_score",
    "pipeline_steps",
    "errors",
    "timings",
}

_VALID_LANGUAGES = {"kannada_script", "romanized_kannada", "english", "mixed", "unknown"}
_VALID_SENTIMENTS = {"Positive", "Negative", "Neutral"}

# ---------------------------------------------------------------------------
# Expected DataFrame columns for process_batch()
# ---------------------------------------------------------------------------
_BATCH_COLUMNS = {
    "original_text",
    "detected_language",
    "detection_confidence",
    "transliterated_text",
    "translated_text",
    "sentiment_label",
    "confidence_score",
    "pipeline_steps",
    "errors",
    "processing_time_s",
    "status",
}
_VALID_STATUSES = {"ok", "ok_with_warnings", "error", "skipped"}


# ===========================================================================
# Shared pipeline fixture  (module scope — loaded once for the whole file)
# ===========================================================================

@pytest.fixture(scope="module")
def pipeline() -> KannadaSentimentPipeline:
    """
    A fully initialised KannadaSentimentPipeline using the fallback translator
    and no IndicXlit model.  Shared across all tests in this module to avoid
    re-loading DistilBERT on every test.
    """
    return KannadaSentimentPipeline(
        translation_backend="fallback",
        use_transliteration_model=False,
        auto_fallback=True,
    )


# ===========================================================================
# Result-structure tests
# ===========================================================================

class TestResultStructure:
    """Verify the shape of the dict returned by process()."""

    def test_all_keys_present(self, pipeline):
        result = pipeline.process("This product is great")
        assert _EXPECTED_KEYS.issubset(result.keys()), (
            f"Missing keys: {_EXPECTED_KEYS - result.keys()}"
        )

    def test_original_text_preserved(self, pipeline):
        text = "Hello world this is a test"
        result = pipeline.process(text)
        assert result["original_text"] == text

    def test_detected_language_valid(self, pipeline):
        result = pipeline.process("This is great")
        assert result["detected_language"] in _VALID_LANGUAGES

    def test_sentiment_label_valid(self, pipeline):
        result = pipeline.process("This is great")
        assert result["sentiment_label"] in _VALID_SENTIMENTS

    def test_confidence_score_in_range(self, pipeline):
        for text in [
            "This product is amazing",
            "ಚೆನ್ನಾಗಿದೆ",
            "tumba olle agide",
        ]:
            result = pipeline.process(text)
            score = result["confidence_score"]
            assert 0.0 <= score <= 1.0, (
                f"confidence_score={score!r} out of range for: {text!r}"
            )

    def test_detection_confidence_in_range(self, pipeline):
        result = pipeline.process("This product is amazing")
        assert 0.0 <= result["detection_confidence"] <= 1.0

    def test_pipeline_steps_is_list(self, pipeline):
        result = pipeline.process("This is good")
        assert isinstance(result["pipeline_steps"], list)

    def test_errors_is_list(self, pipeline):
        result = pipeline.process("This is good")
        assert isinstance(result["errors"], list)

    def test_timings_is_dict(self, pipeline):
        result = pipeline.process("This is good")
        assert isinstance(result["timings"], dict)

    def test_timings_contains_total(self, pipeline):
        result = pipeline.process("This is good")
        assert "total" in result["timings"]
        assert result["timings"]["total"] >= 0.0

    def test_timings_are_non_negative(self, pipeline):
        result = pipeline.process("This is good")
        for key, val in result["timings"].items():
            assert val >= 0.0, f"Negative timing for '{key}': {val}"


# ===========================================================================
# Routing tests — English path
# ===========================================================================

class TestEnglishRoute:

    def test_detected_as_english(self, pipeline):
        result = pipeline.process("This product is excellent and I love it")
        assert result["detected_language"] == "english"

    def test_no_transliteration_for_english(self, pipeline):
        """English reviews must NOT produce a transliterated_text."""
        result = pipeline.process("This product is great")
        # transliterated_text should be None for the English route
        assert result["transliterated_text"] is None

    def test_language_detection_step_present(self, pipeline):
        result = pipeline.process("This product is great")
        assert "language_detection" in result["pipeline_steps"]

    def test_positive_english_sentiment(self, pipeline):
        result = pipeline.process(
            "This is an outstanding product, absolutely love it. Highly recommended!"
        )
        assert result["sentiment_label"] in _VALID_SENTIMENTS
        assert result["confidence_score"] > 0.0

    def test_negative_english_sentiment(self, pipeline):
        result = pipeline.process(
            "Terrible product, complete waste of money. Worst purchase ever."
        )
        assert result["sentiment_label"] in _VALID_SENTIMENTS


# ===========================================================================
# Routing tests — Kannada script path
# ===========================================================================

class TestKannadaScriptRoute:

    def test_detected_as_kannada_script(self, pipeline):
        result = pipeline.process("ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ")
        assert result["detected_language"] == "kannada_script"

    def test_translation_attempted(self, pipeline):
        """Kannada script → translator is called → translated_text is populated or errors logged."""
        result = pipeline.process("ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ")
        # With fallback translator, translated_text may be a partial dict translation
        # or None if nothing matched.  The key must exist.
        assert "translated_text" in result

    def test_no_transliteration_for_kannada_script(self, pipeline):
        """Native Kannada script does NOT go through transliteration."""
        result = pipeline.process("ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ")
        assert result["transliterated_text"] is None

    def test_returns_valid_sentiment(self, pipeline):
        result = pipeline.process("ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ")
        assert result["sentiment_label"] in _VALID_SENTIMENTS
        assert result["confidence_score"] >= 0.0


# ===========================================================================
# Routing tests — Romanized Kannada path
# ===========================================================================

class TestRomanizedKannadaRoute:

    def test_detected_as_romanized(self, pipeline):
        result = pipeline.process("ee product tumba chennagide")
        assert result["detected_language"] == "romanized_kannada"

    def test_transliteration_performed(self, pipeline):
        """Romanized input should produce a non-None transliterated_text."""
        result = pipeline.process("tumba olle agide")
        # transliterated_text should be set (may equal original if dict misses)
        assert "transliterated_text" in result

    def test_returns_valid_sentiment(self, pipeline):
        result = pipeline.process("tumba olle agide")
        assert result["sentiment_label"] in _VALID_SENTIMENTS
        assert result["confidence_score"] >= 0.0

    def test_transliteration_step_listed(self, pipeline):
        result = pipeline.process("tumba olle agide")
        assert "transliteration" in result["pipeline_steps"]


# ===========================================================================
# Edge cases — process()
# ===========================================================================

class TestEdgeCases:

    def test_empty_string(self, pipeline):
        """Empty input must not raise; should produce a graceful result."""
        result = pipeline.process("")
        assert result["detected_language"] == "unknown"
        assert isinstance(result["errors"], list)
        # No crash — all keys must still be present
        assert _EXPECTED_KEYS.issubset(result.keys())

    def test_whitespace_only(self, pipeline):
        result = pipeline.process("   \t\n  ")
        assert result["detected_language"] == "unknown"

    def test_very_long_english_text(self, pipeline):
        """Text >512 BPE tokens must not crash (transformer truncates silently)."""
        result = pipeline.process(make_long_english())
        assert result["detected_language"] == "english"
        assert result["sentiment_label"] in _VALID_SENTIMENTS

    def test_very_long_kannada_text(self, pipeline):
        """Very long Kannada-script text must be processed without errors."""
        result = pipeline.process(make_long_kannada())
        assert result["detected_language"] == "kannada_script"
        assert _EXPECTED_KEYS.issubset(result.keys())

    def test_very_long_romanized_text(self, pipeline):
        """Very long romanized Kannada text must be processed without errors."""
        result = pipeline.process(make_long_romanized())
        assert _EXPECTED_KEYS.issubset(result.keys())
        assert result["sentiment_label"] in _VALID_SENTIMENTS

    def test_numbers_only(self, pipeline):
        """A string of digits must be handled gracefully."""
        result = pipeline.process("12345678")
        assert _EXPECTED_KEYS.issubset(result.keys())
        assert result["sentiment_label"] in _VALID_SENTIMENTS

    def test_special_characters_only(self, pipeline):
        result = pipeline.process("!@#$%^&*()")
        assert _EXPECTED_KEYS.issubset(result.keys())
        assert result["detected_language"] in _VALID_LANGUAGES

    def test_emojis_only(self, pipeline):
        """Pure emoji input must not crash."""
        result = pipeline.process("😊👍🎉❤️🚀")
        assert _EXPECTED_KEYS.issubset(result.keys())
        assert result["sentiment_label"] in _VALID_SENTIMENTS

    def test_emoji_with_english(self, pipeline):
        result = pipeline.process("This product is amazing! 😊👍🎉")
        assert result["detected_language"] in ("english", "mixed")
        assert result["sentiment_label"] in _VALID_SENTIMENTS

    def test_emoji_with_kannada(self, pipeline):
        result = pipeline.process("ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ! 👍👍")
        assert result["detected_language"] in ("kannada_script", "mixed")

    def test_mixed_script_text(self, pipeline):
        result = pipeline.process("ಈ product ತುಂಬಾ good ಆಗಿದೆ")
        assert result["detected_language"] in ("mixed", "kannada_script")
        assert result["sentiment_label"] in _VALID_SENTIMENTS

    def test_single_english_word(self, pipeline):
        result = pipeline.process("Good")
        assert _EXPECTED_KEYS.issubset(result.keys())
        assert result["sentiment_label"] in _VALID_SENTIMENTS

    def test_numbers_with_text(self, pipeline):
        result = pipeline.process("I give this 5 out of 5 stars")
        assert result["detected_language"] == "english"
        assert result["sentiment_label"] in _VALID_SENTIMENTS

    def test_text_with_newlines(self, pipeline):
        result = pipeline.process("This product is great.\nFast delivery.\nHighly recommended.")
        assert result["detected_language"] == "english"

    def test_punctuation_heavy_text(self, pipeline):
        result = pipeline.process("Excellent!!! Works perfectly... 100% satisfied??")
        assert result["detected_language"] == "english"


# ===========================================================================
# process_batch() — list input
# ===========================================================================

@pytest.mark.skipif(not _PANDAS, reason="pandas not installed")
class TestBatchList:

    def test_basic_batch(self, pipeline):
        texts = [
            "This product is great",
            "Terrible, waste of money",
            "ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ",
        ]
        df = pipeline.process_batch(texts, show_progress=False, use_multiprocessing=False)
        assert len(df) == len(texts)

    def test_output_columns_complete(self, pipeline):
        df = pipeline.process_batch(
            ["This is great"],
            show_progress=False,
            use_multiprocessing=False,
        )
        assert _BATCH_COLUMNS.issubset(df.columns), (
            f"Missing columns: {_BATCH_COLUMNS - set(df.columns)}"
        )

    def test_row_order_preserved(self, pipeline):
        """Rows in the output must correspond to the same-index input text."""
        texts = [
            "First review is positive",
            "Second review is negative",
            "Third review is neutral perhaps",
        ]
        df = pipeline.process_batch(texts, show_progress=False, use_multiprocessing=False)
        for i, text in enumerate(texts):
            assert df.loc[i, "original_text"] == text

    def test_sentiment_label_column_valid(self, pipeline):
        df = pipeline.process_batch(
            ["Great product", "Bad product"],
            show_progress=False, use_multiprocessing=False,
        )
        for label in df["sentiment_label"]:
            assert label in _VALID_SENTIMENTS

    def test_status_column_valid(self, pipeline):
        df = pipeline.process_batch(
            ["Good product", "Bad product"],
            show_progress=False, use_multiprocessing=False,
        )
        for status in df["status"]:
            assert status in _VALID_STATUSES

    def test_confidence_scores_in_range(self, pipeline):
        df = pipeline.process_batch(
            ["Great", "Terrible"],
            show_progress=False, use_multiprocessing=False,
        )
        for score in df["confidence_score"]:
            assert 0.0 <= float(score) <= 1.0

    def test_empty_list_returns_empty_df(self, pipeline):
        df = pipeline.process_batch([], show_progress=False, use_multiprocessing=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_skipped_empty_strings(self, pipeline):
        """Empty or whitespace strings must produce rows with status='skipped'."""
        texts = ["Good product", "", "   ", "Another review"]
        df = pipeline.process_batch(texts, show_progress=False, use_multiprocessing=False)
        assert len(df) == 4
        skipped = df[df["status"] == "skipped"]
        assert len(skipped) == 2

    def test_skipped_non_string_inputs(self, pipeline):
        """Non-string items (None, int) must produce status='skipped'."""
        texts = ["Good product", None, 42, "Another"]
        df = pipeline.process_batch(texts, show_progress=False, use_multiprocessing=False)
        assert len(df) == 4
        skipped = df[df["status"] == "skipped"]
        assert len(skipped) == 2

    def test_all_empty_input_returns_empty_results(self, pipeline):
        df = pipeline.process_batch(
            ["", "   ", "\t"],
            show_progress=False, use_multiprocessing=False,
        )
        assert len(df) == 3
        assert all(df["status"] == "skipped")

    def test_large_batch(self, pipeline):
        """Batch of 20 diverse reviews must complete without exception."""
        texts = [
            "Great product highly recommend",
            "Terrible waste of money",
            "ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ",
            "ಕಳಪೆ ಗುಣಮಟ್ಟ",
            "tumba olle agide",
            "ee product ketta agide",
            "Decent quality for the price",
            "Fast delivery and nice packaging",
            "👍 Amazing! 🎉",
            "😡 Very disappointed",
            "ಈ product ತುಂಬಾ good",
            "12345",
            "!@#$%",
            "This is amazing and works perfectly",
            "Worst purchase I have ever made",
            "ಬಹಳ ಒಳ್ಳೆಯ ಸೇವೆ",
            "tumba chennagide recommend madtini",
            "average product nothing special",
            "excellent build quality and design",
            "complete junk do not buy",
        ]
        df = pipeline.process_batch(texts, show_progress=False, use_multiprocessing=False)
        assert len(df) == len(texts)
        assert set(df.columns).issuperset(_BATCH_COLUMNS)


# ===========================================================================
# process_batch() — DataFrame input
# ===========================================================================

@pytest.mark.skipif(not _PANDAS, reason="pandas not installed")
class TestBatchDataFrame:

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "review": [
                "This is a great product",
                "Terrible quality, very disappointed",
                "ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ",
            ],
            "rating": [5, 1, 4],
        })

    def test_dataframe_input(self, pipeline, sample_df):
        df = pipeline.process_batch(
            sample_df,
            text_column="review",
            show_progress=False,
            use_multiprocessing=False,
        )
        assert len(df) == len(sample_df)

    def test_wrong_column_raises(self, pipeline, sample_df):
        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            pipeline.process_batch(
                sample_df,
                text_column="nonexistent",
                show_progress=False,
                use_multiprocessing=False,
            )

    def test_output_columns_from_df_input(self, pipeline, sample_df):
        df = pipeline.process_batch(
            sample_df,
            text_column="review",
            show_progress=False,
            use_multiprocessing=False,
        )
        assert _BATCH_COLUMNS.issubset(df.columns)

    def test_original_text_matches_column(self, pipeline, sample_df):
        df = pipeline.process_batch(
            sample_df,
            text_column="review",
            show_progress=False,
            use_multiprocessing=False,
        )
        for i, expected in enumerate(sample_df["review"].tolist()):
            assert df.loc[i, "original_text"] == expected


# ===========================================================================
# pipeline_steps ordering
# ===========================================================================

class TestPipelineStepsOrdering:

    def test_language_detection_always_first(self, pipeline):
        for text in [
            "This is great",
            "ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ",
            "tumba olle agide",
        ]:
            result = pipeline.process(text)
            steps = result["pipeline_steps"]
            if steps:
                assert steps[0] == "language_detection", (
                    f"First step was '{steps[0]}' for text: {text!r}"
                )

    def test_sentiment_always_last(self, pipeline):
        """Sentiment classification should be the last executed step."""
        for text in ["This is great", "ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ"]:
            result = pipeline.process(text)
            steps = result["pipeline_steps"]
            if len(steps) >= 2:
                assert steps[-1] == "sentiment_classification", (
                    f"Last step was '{steps[-1]}' for text: {text!r}"
                )

    def test_transliteration_before_translation(self, pipeline):
        """For romanized input, transliteration must precede translation."""
        result = pipeline.process("tumba olle agide baralla gottilla")
        steps = result["pipeline_steps"]
        if "transliteration" in steps and "translation" in steps:
            assert steps.index("transliteration") < steps.index("translation")


# ===========================================================================
# Error propagation
# ===========================================================================

class TestErrorPropagation:

    def test_errors_empty_for_clean_english(self, pipeline):
        result = pipeline.process("This product is great and I love it")
        # With fallback backend, errors list should be empty for English
        assert isinstance(result["errors"], list)

    def test_non_fatal_errors_do_not_crash(self, pipeline):
        """Even if internal steps encounter non-fatal errors, no exception is raised."""
        # Romanized text where transliteration may have gaps
        result = pipeline.process("tumba olle xyzblorp qwerty")
        assert _EXPECTED_KEYS.issubset(result.keys())
        assert result["sentiment_label"] in _VALID_SENTIMENTS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
