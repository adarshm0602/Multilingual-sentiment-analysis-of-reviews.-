#!/usr/bin/env python3
"""Benchmark Naive Bayes, Logistic Regression, and Linear SVM on Kannada reviews."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_eval_dataset import main as build_datasets
from src.evaluation.classifier_benchmark import run_classifier_benchmark, save_benchmark_results
from src.evaluation.dataset import INDIC_KN_JSONL, load_indic_sentiment_kn, load_multilingual_eval_set

OUTPUT = PROJECT_ROOT / "data" / "processed" / "classifier_benchmark_results.json"


def _print_summary(results: dict) -> None:
    print("\n=== Classifier Benchmark (Kannada Script) ===")
    print(f"Dataset size : {results['dataset_size']} labeled reviews")
    print(f"Train / test : {results['train_size']} / {results['test_size']}")
    print()
    for name, metrics in results["classifiers"].items():
        print(
            f"  {name:22s}  accuracy={metrics['accuracy']:.1%}  "
            f"macro-F1={metrics['macro_f1']:.1%}  weighted-F1={metrics['weighted_f1']:.1%}"
        )
    print()
    best = results["best_classifier"].replace("_", " ").title()
    baseline = results["baseline_classifier"].replace("_", " ").title()
    print(
        f"Best model   : {best} ({results['best_macro_f1']:.1%} macro-F1, "
        f"{results['classifiers'][results['best_classifier']]['weighted_f1']:.1%} weighted-F1)"
    )
    print(
        f"NB baseline  : {baseline} ({results['baseline_macro_f1']:.1%} macro-F1)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("indic", "multilingual", "both"),
        default="both",
        help="Which dataset(s) to benchmark",
    )
    args = parser.parse_args()

    if not INDIC_KN_JSONL.exists():
        build_datasets()

    all_results = {"benchmarks": {}}

    if args.dataset in ("indic", "both"):
        kn_df = load_indic_sentiment_kn()
        indic_results = run_classifier_benchmark(kn_df)
        all_results["benchmarks"]["kannada_script_indic_sentiment"] = indic_results
        _print_summary(indic_results)

    if args.dataset in ("multilingual", "both"):
        multi_df = load_multilingual_eval_set()
        multi_results = run_classifier_benchmark(multi_df)
        all_results["benchmarks"]["multilingual_hand_labeled"] = multi_results
        print("\n=== Classifier Benchmark (50-review multilingual set) ===")
        _print_summary(multi_results)

    save_benchmark_results(all_results, OUTPUT)
    print(f"\nSaved results to {OUTPUT}")


if __name__ == "__main__":
    main()
