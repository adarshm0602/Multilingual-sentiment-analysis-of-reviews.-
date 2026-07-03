"""TF-IDF + sklearn classifier benchmarking for Kannada sentiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2


@dataclass
class ClassifierResult:
    name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    positive_f1: float
    negative_f1: float
    neutral_f1: float
    per_class: Dict[str, Dict[str, float]]


def _per_class_metrics(
    y_true: List[str], y_pred: List[str], labels: List[str]
) -> Dict[str, Dict[str, float]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }


def _class_f1(per_class: Dict[str, Dict[str, float]], label: str) -> float:
    return float(per_class.get(label, {}).get("f1", 0.0))


def _build_model(name: str):
    if name == "naive_bayes":
        return MultinomialNB()
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=DEFAULT_RANDOM_STATE)
    if name == "linear_svm":
        return LinearSVC(random_state=DEFAULT_RANDOM_STATE)
    raise ValueError(f"Unknown classifier: {name}")


def _make_pipeline(classifier_name: str) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 4),
                    min_df=2,
                    max_features=50000,
                ),
            ),
            ("clf", _build_model(classifier_name)),
        ]
    )


def run_classifier_benchmark(
    df: pd.DataFrame,
    text_col: str = "review",
    label_col: str = "ground_truth",
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    classifiers: Optional[List[str]] = None,
) -> Dict:
    """
    Benchmark sklearn classifiers on a labeled review DataFrame.

    Uses a stratified train/test split and character n-gram TF-IDF features,
    which work better for Kannada script than word-level tokenization.
    """
    if classifiers is None:
        classifiers = ["naive_bayes", "logistic_regression", "linear_svm"]

    work = df[[text_col, label_col]].dropna().copy()
    work[label_col] = work[label_col].astype(str)
    labels = sorted(work[label_col].unique())

    x_train, x_test, y_train, y_test = train_test_split(
        work[text_col].tolist(),
        work[label_col].tolist(),
        test_size=test_size,
        random_state=random_state,
        stratify=work[label_col],
    )

    results: List[ClassifierResult] = []
    for clf_name in classifiers:
        pipe = _make_pipeline(clf_name)
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)

        per_class = _per_class_metrics(y_test, y_pred, labels)
        results.append(
            ClassifierResult(
                name=clf_name,
                accuracy=float(accuracy_score(y_test, y_pred)),
                macro_f1=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                weighted_f1=float(
                    f1_score(y_test, y_pred, average="weighted", zero_division=0)
                ),
                positive_f1=_class_f1(per_class, "Positive"),
                negative_f1=_class_f1(per_class, "Negative"),
                neutral_f1=_class_f1(per_class, "Neutral"),
                per_class=per_class,
            )
        )

    ranked = sorted(results, key=lambda r: r.macro_f1, reverse=True)
    best = ranked[0]
    baseline = next((r for r in results if r.name == "naive_bayes"), ranked[-1])

    return {
        "dataset_size": int(len(work)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "labels": labels,
        "classifiers": {r.name: asdict(r) for r in results},
        "best_classifier": best.name,
        "best_macro_f1": best.macro_f1,
        "best_weighted_f1": best.weighted_f1,
        "baseline_classifier": baseline.name,
        "baseline_macro_f1": baseline.macro_f1,
        "classification_report": classification_report(
            y_test,
            _make_pipeline(best.name).fit(x_train, y_train).predict(x_test),
            zero_division=0,
        ),
    }


def save_benchmark_results(results: Dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    return output_path
