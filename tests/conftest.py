"""
Shared pytest configuration and fixtures.

Puts the project root on sys.path so all test modules can import `src.*`
without a package install.  Also defines cross-module constants and fixtures.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is importable from every test file
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import pytest


# ---------------------------------------------------------------------------
# Long-text helpers  (shared by detector, transliterator and pipeline tests)
# ---------------------------------------------------------------------------

def make_long_english(min_tokens: int = 550) -> str:
    """
    Return an English string long enough to exceed the 512-token limit used
    by most transformer classifiers.  Each repetition is ~12 tokens.
    """
    chunk = (
        "This is an excellent product that I would highly recommend to everyone. "
        "The quality is outstanding and delivery was impressively fast. "
    )
    reps = (min_tokens // 12) + 1
    return chunk * reps


def make_long_kannada(min_chars: int = 2000) -> str:
    """Return a Kannada-script string longer than 2000 characters."""
    chunk = "ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ ಮತ್ತು ಬಹಳ ಉಪಯೋಗಕರ. "
    reps = (min_chars // len(chunk)) + 1
    return chunk * reps


def make_long_romanized(min_chars: int = 2000) -> str:
    """Return a romanized-Kannada string longer than 2000 characters."""
    chunk = "ee product tumba chennagide mattu bahala upayogakara. "
    reps = (min_chars // len(chunk)) + 1
    return chunk * reps


# ---------------------------------------------------------------------------
# Canonical sample-data fixture (available to all modules via conftest)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_reviews():
    """
    A dict of labelled example texts used in multiple test modules.

    Structure:  {category: [text, ...]}
    """
    return {
        "kannada_script_positive": [
            "ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ",
            "ಬಹಳ ಒಳ್ಳೆಯ ಸೇವೆ ಮತ್ತು ಉತ್ತಮ ಗುಣಮಟ್ಟ",
        ],
        "kannada_script_negative": [
            "ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಕಳಪೆ ಗುಣಮಟ್ಟದ್ದು",
            "ಸಂಪೂರ್ಣ ವ್ಯರ್ಥ ಖರೀದಿ ಮತ್ತು ಕೆಟ್ಟ ಸೇವೆ",
        ],
        "romanized_kannada": [
            "tumba chennagide",
            "ee product olle agide",
            "tumba ketta agide baralla",
        ],
        "english_positive": [
            "This product is excellent and I love it",
            "Highly recommended, great quality and fast delivery",
        ],
        "english_negative": [
            "Terrible product, complete waste of money",
            "Very disappointed, worst purchase ever",
        ],
        "mixed": [
            "ಈ product ತುಂಬಾ good ಆಗಿದೆ",
        ],
    }
