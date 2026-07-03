#!/usr/bin/env python3
"""Build CSV copies of labeled datasets used by Streamlit and notebooks."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_indic_sentiment_kn import download_kannada_reviews
from src.evaluation.dataset import (
    INDIC_KN_JSONL,
    export_csv_copies,
    load_indic_sentiment_kn,
    load_multilingual_eval_set,
)


def main() -> None:
    if not INDIC_KN_JSONL.exists():
        print("Downloading IndicSentiment Kannada reviews …")
        download_kannada_reviews()

    demo_csv, multilingual_csv = export_csv_copies()
    multilingual = load_multilingual_eval_set()
    benchmark = load_indic_sentiment_kn()

    from src.evaluation.dataset import load_demo_reviews

    print(f"Wrote demo CSV           : {demo_csv} ({len(load_demo_reviews())} rows)")
    print(f"Wrote multilingual CSV   : {multilingual_csv} ({len(multilingual)} rows)")
    print(f"Kannada benchmark cache  : {len(benchmark)} rows from IndicSentiment")


if __name__ == "__main__":
    main()
