#!/usr/bin/env python3
"""Evaluate the full NLP pipeline on the multilingual hand-labeled set."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_eval_dataset import main as build_datasets
from src.evaluation.dataset import load_multilingual_eval_set
from src.pipeline import KannadaSentimentPipeline

OUTPUT_FALLBACK = PROJECT_ROOT / "data" / "processed" / "pipeline_evaluation_fallback.json"
OUTPUT_NLLB = PROJECT_ROOT / "data" / "processed" / "pipeline_evaluation_nllb.json"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "pipeline_evaluation_results.json"


def _per_script_metrics(df: pd.DataFrame) -> dict:
    out = {}
    for script, sub in df.groupby("script_type"):
        yt = sub["ground_truth"].tolist()
        yp = sub["predicted_label"].tolist()
        out[script] = {
            "count": int(len(sub)),
            "correct": int(sub["correct"].sum()),
            "accuracy": float(accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        }
    return out


def run_pipeline_evaluation(
    translation_backend: str = "nllb",
    use_transliteration_model: bool = False,
) -> dict:
    df = load_multilingual_eval_set()

    local_model = PROJECT_ROOT / "data" / "models" / "distilbert-sst2"
    classifier_model_path = str(local_model) if local_model.exists() else None

    pipeline = KannadaSentimentPipeline(
        translation_backend=translation_backend,
        use_transliteration_model=use_transliteration_model,
        classifier_model_path=classifier_model_path,
    )

    rows = []
    t0 = time.perf_counter()
    for _, row in df.iterrows():
        result = pipeline.process(row["review"])
        predicted = result["sentiment_label"]
        ground_truth = row["ground_truth"]
        rows.append(
            {
                "review_id": row["review_id"],
                "script_type": row["script_type"],
                "ground_truth": ground_truth,
                "predicted_label": predicted,
                "correct": predicted == ground_truth,
                "confidence_score": result.get("confidence_score", 0.0),
                "detected_language": result.get("detected_language"),
                "processing_time_s": result.get("processing_time_s", 0.0),
            }
        )

    results_df = pd.DataFrame(rows)
    wall_time = time.perf_counter() - t0

    y_true = results_df["ground_truth"].tolist()
    y_pred = results_df["predicted_label"].tolist()
    labels = ["Positive", "Negative", "Neutral"]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "translation_backend": translation_backend,
        "dataset_size": int(len(df)),
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "correct": int(results_df["correct"].sum()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": per_class,
        "per_script_type": _per_script_metrics(results_df),
        "wall_time_s": float(wall_time),
        "avg_time_per_review_s": float(wall_time / len(df)),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "reviews": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("nllb", "fallback", "marian"),
        default="nllb",
        help="Translation backend for pipeline evaluation",
    )
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Skip NLLB and use dictionary fallback (fast, for CI)",
    )
    args = parser.parse_args()

    build_datasets()
    backend = "fallback" if args.fallback_only else args.backend

    print(f"Running pipeline evaluation ({backend} backend) on 50 multilingual reviews …")
    results = run_pipeline_evaluation(translation_backend=backend)

    output_path = OUTPUT_FALLBACK if backend == "fallback" else OUTPUT_NLLB
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    # Keep a stable alias for the most recent run
    with OUTPUT.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"\nOverall accuracy : {results['overall_accuracy']:.1%} ({results['correct']}/{results['dataset_size']})")
    print(f"Macro F1         : {results['macro_f1']:.1%}")
    print(f"Weighted F1      : {results['weighted_f1']:.1%}")
    print("\nPer script type:")
    for script, metrics in results["per_script_type"].items():
        print(
            f"  {script:20s}  acc={metrics['accuracy']:.1%}  "
            f"({metrics['correct']}/{metrics['count']})  macro-F1={metrics['macro_f1']:.1%}"
        )
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
