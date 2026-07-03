"""Dataset loading and label utilities for evaluation and benchmarking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELED_DIR = PROJECT_ROOT / "data" / "labeled"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEMO_JSON = LABELED_DIR / "demo_reviews.json"
EXTENDED_JSON = LABELED_DIR / "extended_reviews.json"
INDIC_KN_JSONL = RAW_DIR / "indic_sentiment_kn.jsonl"
MULTILINGUAL_CSV = PROCESSED_DIR / "multilingual_eval_set.csv"
BENCHMARK_CSV = PROCESSED_DIR / "kannada_benchmark_set.csv"


def rating_to_sentiment(rating: int) -> str:
    """Map star rating to sentiment label."""
    if rating >= 4:
        return "Positive"
    if rating <= 2:
        return "Negative"
    return "Neutral"


def indic_label_to_sentiment(label: str) -> str:
    """Normalize IndicSentiment labels to project sentiment names."""
    mapping = {
        "positive": "Positive",
        "negative": "Negative",
        "neutral": "Neutral",
        "Positive": "Positive",
        "Negative": "Negative",
        "Neutral": "Neutral",
    }
    normalized = mapping.get(label.strip(), label.strip().title())
    if normalized not in {"Positive", "Negative", "Neutral"}:
        raise ValueError(f"Unknown IndicSentiment label: {label!r}")
    return normalized


def _load_json_reviews(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as fh:
        rows = json.load(fh)
    return pd.DataFrame(rows)


def load_demo_reviews() -> pd.DataFrame:
    """Load the 20-review demo set."""
    return _load_json_reviews(DEMO_JSON)


def load_extended_reviews() -> pd.DataFrame:
    """Load the 30-review extended multilingual set."""
    return _load_json_reviews(EXTENDED_JSON)


def load_multilingual_eval_set(add_ground_truth: bool = True) -> pd.DataFrame:
    """Load the 50-review hand-labeled multilingual evaluation set."""
    demo = load_demo_reviews()
    extended = load_extended_reviews()
    df = pd.concat([demo, extended], ignore_index=True)
    if add_ground_truth:
        df["ground_truth"] = df["rating"].map(rating_to_sentiment)
    return df


def load_indic_sentiment_kn(path: Optional[Path] = None) -> pd.DataFrame:
    """Load Kannada product reviews from cached IndicSentiment JSONL."""
    jsonl_path = path or INDIC_KN_JSONL
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"IndicSentiment Kannada cache not found at {jsonl_path}. "
            "Run: python scripts/download_indic_sentiment_kn.py"
        )

    rows: List[Dict] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            label = record.get("LABEL")
            if not label:
                continue
            rows.append(
                {
                    "review_id": f"indic_{line_no}",
                    "review": record["INDIC REVIEW"],
                    "english_review": record.get("ENGLISH REVIEW", ""),
                    "product_category": record.get("CATEGORY", ""),
                    "sub_category": record.get("SUB-CATEGORY", ""),
                    "product": record.get("PRODUCT", ""),
                    "brand": record.get("BRAND", ""),
                    "ground_truth": indic_label_to_sentiment(record["LABEL"]),
                    "script_type": "Kannada Script",
                    "source": "ai4bharat/IndicSentiment",
                    "split": record.get("_split", "unknown"),
                }
            )
    return pd.DataFrame(rows)


def export_csv_copies() -> Tuple[Path, Path]:
    """Write CSV copies used by Streamlit and notebooks."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    demo_csv = PROJECT_ROOT / "data" / "demo_data.csv"
    demo = load_demo_reviews()
    demo.to_csv(demo_csv, index=False)

    multilingual = load_multilingual_eval_set()
    multilingual.to_csv(MULTILINGUAL_CSV, index=False)

    if INDIC_KN_JSONL.exists():
        benchmark = load_indic_sentiment_kn()
        benchmark.to_csv(BENCHMARK_CSV, index=False)

    return demo_csv, MULTILINGUAL_CSV
