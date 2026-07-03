#!/usr/bin/env python3
"""Download Kannada reviews from ai4bharat/IndicSentiment and cache locally."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT = PROJECT_ROOT / "data" / "raw" / "indic_sentiment_kn.jsonl"
DATASET_REPO = "ai4bharat/IndicSentiment"
SPLITS = ("validation", "test")


def download_kannada_reviews(output_path: Path = OUTPUT) -> Path:
    """Fetch validation + test Kannada splits and write a unified JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in SPLITS:
        remote = f"data/{split}/kn.json"
        local = hf_hub_download(DATASET_REPO, remote, repo_type="dataset")
        with open(local, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["_split"] = split
                rows.append(record)

    with output_path.open("w", encoding="utf-8") as fh:
        for record in rows:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} Kannada reviews to {output_path}")
    return output_path


if __name__ == "__main__":
    download_kannada_reviews()
