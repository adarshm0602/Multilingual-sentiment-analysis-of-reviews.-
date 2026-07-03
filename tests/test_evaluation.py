"""Tests for evaluation dataset utilities."""

from pathlib import Path

import pytest

from src.evaluation.dataset import (
    load_demo_reviews,
    load_extended_reviews,
    load_multilingual_eval_set,
    rating_to_sentiment,
)


def test_rating_to_sentiment_mapping():
    assert rating_to_sentiment(5) == "Positive"
    assert rating_to_sentiment(4) == "Positive"
    assert rating_to_sentiment(3) == "Neutral"
    assert rating_to_sentiment(2) == "Negative"
    assert rating_to_sentiment(1) == "Negative"


def test_demo_reviews_count():
    df = load_demo_reviews()
    assert len(df) == 20
    assert set(df["script_type"]) == {"Kannada Script", "Romanized Kannada", "English"}


def test_multilingual_eval_set_count():
    df = load_multilingual_eval_set()
    assert len(df) == 50
    assert "ground_truth" in df.columns


@pytest.mark.skipif(
    not Path("data/raw/indic_sentiment_kn.jsonl").exists(),
    reason="IndicSentiment cache not downloaded",
)
def test_indic_sentiment_cache_loads():
    from src.evaluation.dataset import load_indic_sentiment_kn

    df = load_indic_sentiment_kn()
    assert len(df) >= 1000
    assert set(df["ground_truth"]).issubset({"Positive", "Negative", "Neutral"})
