# Multilingual Sentiment Analysis of Kannada Product Reviews

> An end-to-end NLP pipeline that classifies the sentiment of product reviews written in
> **Kannada script**, **Romanized Kannada** (Kanglish), or **English** — with a
> Streamlit web application for interactive exploration and batch processing.

**Live demo:** [multilingual-review-analysis6666.streamlit.app](https://multilingual-review-analysis6666.streamlit.app/)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Problem Statement & Motivation](#problem-statement--motivation)
3. [System Architecture](#system-architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Dataset](#dataset)
7. [Model Performance](#model-performance)
8. [Known Limitations](#known-limitations)
9. [Future Work](#future-work)
10. [Project Structure](#project-structure)
11. [References](#references)
12. [License](#license)

---

## Quick Start

```bash
git clone <repo-url>
cd "Multilingual-sentiment-analysis-of-reviews.-"

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
python scripts/build_eval_dataset.py
python src/models/download_model.py --models distilbert-sst2   # optional; auto-downloads on first use

streamlit run app/streamlit_app.py
```

Open **http://localhost:8501**.

> **Important:** Use the project `venv` (Python 3.10–3.12). Running Streamlit from system
> Python (e.g. 3.13) without activating the venv will cause missing-module errors such as
> `ModuleNotFoundError: No module named 'langdetect'`.

---

## Problem Statement & Motivation

E-commerce platforms such as Amazon, Flipkart, and Meesho serve hundreds of millions of
Indian consumers, yet the vast majority of their sentiment analysis systems are
**English-only**. This creates a significant blind spot: a large fraction of user
reviews are written in regional scripts or in transliterated Kanglish style
(e.g. *"tumba ketta product, waste of money"*).

Karnataka alone has over 60 million Kannada speakers. Reviews can appear in three
distinct surface forms that require different pre-processing paths:

| Surface Form | Example | Challenges |
|---|---|---|
| Native Kannada script | *ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ* | Requires Kannada→English translation before English sentiment model |
| Romanized Kannada (Kanglish) | *tumba chennagide product* | Mixed-language; must be transliterated to Kannada, then translated |
| English | *Excellent quality, fast delivery* | Can be classified directly |

This project builds a unified pipeline that automatically identifies the input type and
routes it through the appropriate pre-processing steps before sentiment classification.

---

## System Architecture

The pipeline is a four-stage cascade. Each stage is independently testable and replaceable.

```
INPUT TEXT
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1 · Language Detection                                       │
│  • Unicode-block analysis (Kannada U+0C80–U+0CFF → script %)       │
│  • langdetect for romanized / English disambiguation                │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────────┐
          │                 │                     │
    kannada_script   romanized_kannada         english
          │                 │                     │
          │          ┌──────┴──────┐              │
          │          ▼             │              │
          │  Stage 2 · Transliteration (3-tier)  │
          │  IndicXlit → dict (166 words) → ITRANS │
          │                 │                     │
          ├─────────────────┘                     │
          ▼                                       │
  Stage 3 · Translation (NLLB-200 default)       │
          │                                       │
          └───────────────────────────────────────┤
                                                  ▼
                          Stage 4 · Sentiment (DistilBERT SST-2)
                          → Positive / Negative / Neutral + confidence
```

### Key Components

| Component | Class | Description |
|-----------|-------|-------------|
| Language Detector | `LanguageDetector` | Unicode-block ratio + `langdetect`; returns `kannada_script`, `romanized_kannada`, `english`, `mixed`, or `unknown` |
| Transliterator | `Transliterator` | 3-tier: IndicXlit → 166-word dict → rule-based ITRANS→Kannada |
| Translator | `Translator` | 5 backends: NLLB-200 (default), IndicTrans2, MarianMT, Google Translate, fallback dictionary |
| Sentiment Classifier | `SentimentClassifier` | DistilBERT fine-tuned on Stanford SST-2 |
| Pipeline Orchestrator | `KannadaSentimentPipeline` | Routes text, times each stage, supports single and batch inference |

> IndicXlit requires `fairseq`, which is incompatible with Python 3.11+. The pipeline
> automatically falls through to the dictionary and ITRANS tiers.

---

## Installation

### Prerequisites

- Python **3.10 – 3.12** (3.13 is not tested; use the project venv)
- macOS, Linux, or WSL2 on Windows
- ~1.5 GB disk space (DistilBERT ~270 MB + NLLB-200 ~600 MB + dependencies)
- 4 GB+ RAM (8 GB recommended)

### Steps

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/build_eval_dataset.py
```

`build_eval_dataset.py` exports `data/demo_data.csv` and downloads the Kannada benchmark
cache (`data/raw/indic_sentiment_kn.jsonl`, 1,154 reviews) if not already present.

### Model downloads

| Model | Size | Cached at |
|-------|------|-----------|
| DistilBERT-SST-2 | ~270 MB | `data/models/distilbert-sst2/` |
| NLLB-200 distilled 600M | ~600 MB | `~/.cache/huggingface/hub/` |

Pre-download the sentiment model (recommended before first Streamlit run):

```bash
python src/models/download_model.py --models distilbert-sst2
```

NLLB-200 downloads on first translation. Wait for the *"Loading NLLB-200 600M…"* spinner
on the Batch Upload tab before clicking **Process All**.

### Verify installation

```bash
python -m pytest tests/ -v
```

Expected: **216 passed**.

---

## Usage

### Web Application

```bash
source venv/bin/activate
streamlit run app/streamlit_app.py
```

**Sidebar — Translation Engine**

| Option | Description |
|--------|-------------|
| **NLLB-200 600M** *(default)* | Offline, high quality — ~600 MB |
| IndicTrans2 | Offline, best quality — ~3 GB, gated access |
| MarianMT | Offline, lower quality |
| Google Translate | Online — needs `GOOGLE_TRANSLATE_API_KEY` |
| Dictionary Fallback | Offline, instant — 166-word vocabulary |

**Single Review tab** — paste or sample a review, click **Analyze Sentiment**.

**Batch Upload tab** — load demo CSV, upload your own file, or process all with live progress and charts.

### Command-Line Pipeline

```python
from src.pipeline import KannadaSentimentPipeline

pipeline = KannadaSentimentPipeline(
    translation_backend="nllb",
    use_transliteration_model=False,
    auto_fallback=True,
)

result = pipeline.process("ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ")
print(result["sentiment_label"])    # Positive
print(result["detected_language"])  # kannada_script
print(result["translated_text"])
```

### Batch Processing

```python
import pandas as pd
from src.pipeline import KannadaSentimentPipeline

pipeline = KannadaSentimentPipeline(translation_backend="nllb")
df_in = pd.read_csv("data/demo_data.csv")
df_out = pipeline.process_batch(df_in, text_column="review")
df_out.to_csv("results.csv", index=False)
```

### Evaluation & Benchmarks

**Classifier benchmark** (Naive Bayes, Logistic Regression, Linear SVM):

```bash
python scripts/run_classifier_benchmark.py
```

**End-to-end pipeline evaluation**:

```bash
python scripts/run_pipeline_evaluation.py --fallback-only   # fast
python scripts/run_pipeline_evaluation.py --backend nllb    # ~25 min on CPU
```

**Notebooks:**

```bash
jupyter notebook notebooks/04_classifier_benchmark.ipynb   # sklearn benchmark
jupyter notebook notebooks/03_evaluation.ipynb             # pipeline charts
```

**Output files**

| File | Contents |
|------|----------|
| `data/processed/classifier_benchmark_results.json` | NB / LR / Linear SVM metrics |
| `data/processed/pipeline_evaluation_nllb.json` | Full pipeline, NLLB backend |
| `data/processed/pipeline_evaluation_fallback.json` | Full pipeline, dict backend |
| `data/processed/pipeline_evaluation_results.json` | Alias for most recent pipeline run |

---

## Dataset

### Multilingual Evaluation Set (50 reviews, hand-labeled)

Curated for all three script types. Source: `data/labeled/`, exported by `scripts/build_eval_dataset.py`.

| Subset | Reviews | Source |
|--------|---------|--------|
| Demo | 20 | `data/labeled/demo_reviews.json` → `data/demo_data.csv` |
| Extended | 30 | `data/labeled/extended_reviews.json` |
| **Total** | **50** | `data/processed/multilingual_eval_set.csv` |

| Script Type | Count |
|-------------|-------|
| Kannada Script | 20 |
| Romanized Kannada | 15 |
| English | 15 |

Ground truth: star ratings → Positive (4–5), Negative (1–2), Neutral (3).

### Kannada Benchmark Set (1,154 reviews)

Real Kannada product reviews from [ai4bharat/IndicSentiment](https://huggingface.co/datasets/ai4bharat/IndicSentiment) (validation + test splits).

```bash
python scripts/download_indic_sentiment_kn.py   # or via build_eval_dataset.py
```

| Metric | Value |
|--------|-------|
| Total reviews | **1,154** |
| Labels | Positive 580 / Negative 574 (neutral rows dropped) |

### Pre-trained Models

| Model | Purpose |
|-------|---------|
| `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment classification |
| `facebook/nllb-200-distilled-600M` | Kannada → English translation (default) |
| `ai4bharat/IndicXlit` | Neural transliteration (optional tier) |
| `indic-transliteration` | Rule-based ITRANS fallback |
| `langdetect` | Language identification signal |

All inference runs **locally on CPU** — no paid API calls at runtime (except optional Google Translate).

---

## Model Performance

### Classifier Benchmark — Kannada Script (1,154 reviews)

TF-IDF character n-grams (2–4), stratified 80/20 split (923 train / 231 test).

| Classifier | Accuracy | Macro F1 | Weighted F1 |
|------------|----------|----------|-------------|
| **Linear SVM** | **82.7%** | **82.7%** | **82.7%** |
| Logistic Regression | 82.7% | 82.7% | 82.7% |
| Naive Bayes (baseline) | 81.4% | 81.3% | 81.3% |

Linear SVM and Logistic Regression tie at **82.7% macro-F1**, beating Naive Bayes by **+1.4 pp**.

### End-to-End Pipeline — 50 Multilingual Reviews

Full cascade: detection → transliteration → translation → DistilBERT.

#### Dictionary fallback backend

| Metric | Value |
|--------|-------|
| Overall accuracy | **84.0%** (42/50) |
| Kannada Script | **75.0%** (15/20) |
| Romanized Kannada | **86.7%** (13/15) |
| English | **93.3%** (14/15) |
| Weighted F1 | **82.1%** |

#### NLLB-200 backend (recommended)

| Metric | Value |
|--------|-------|
| Overall accuracy | **92.0%** (46/50) |
| Kannada Script | **95.0%** (19/20) |
| Romanized Kannada | **86.7%** (13/15) |
| English | **93.3%** (14/15) |
| Weighted F1 | **90.1%** |
| Avg latency | ~31 s/review on CPU (includes model load) |

NLLB-200 improves Kannada-script accuracy from **75% → 95%** vs. the dictionary fallback,
confirming translation quality as the main bottleneck for native-script reviews.

---

## Known Limitations

1. **Romanized Kannada is hard to detect** — `langdetect` treats Latin-script Kanglish as English. Dictionary heuristics help but edge cases can skip transliteration.

2. **Neutral class is weak** — DistilBERT-SST-2 is binary (Positive/Negative). Neutral is inferred heuristically; only 2 neutral examples in the 50-review eval set.

3. **Translate-then-classify** — Kannada sentiment goes through English translation first. Translation errors propagate. A native Kannada or multilingual model (IndicBERT, MuRIL) would remove this bottleneck.

4. **IndicXlit unavailable on Python 3.11+** — Neural transliteration tier is skipped; dictionary + ITRANS fallbacks are used.

5. **Small multilingual eval set (50 reviews)** — Pipeline metrics have wide confidence intervals. The 1,154-review sklearn benchmark provides more stable classifier estimates.

6. **NLLB latency on CPU** — ~5–30 s per Kannada/Romanized review without GPU.

7. **Jupyter multiprocessing** — `process_batch()` falls back to threads in notebooks due to macOS `spawn` restrictions.

---

## Future Work

- Lightweight Kanglish vs. English detector (fastText or small CNN)
- Fine-tune IndicBERT / MuRIL for native Kannada sentiment
- Expand the 50-review multilingual eval set with more neutral examples
- GPU acceleration for NLLB and DistilBERT
- FastAPI REST endpoint for e-commerce integration
- Support additional Indic languages (Telugu, Tamil, Malayalam, Hindi)

---

## Project Structure

```
.
├── app/
│   └── streamlit_app.py              # Streamlit web UI
├── config.yaml                       # Default backend: nllb
├── data/
│   ├── demo_data.csv                 # 20-review demo (generated)
│   ├── labeled/
│   │   ├── demo_reviews.json         # Source of truth (20 reviews)
│   │   └── extended_reviews.json     # Source of truth (30 reviews)
│   ├── raw/
│   │   └── indic_sentiment_kn.jsonl  # 1,154 Kannada reviews (cached)
│   ├── processed/
│   │   ├── classifier_benchmark_results.json
│   │   ├── pipeline_evaluation_nllb.json
│   │   ├── pipeline_evaluation_fallback.json
│   │   └── multilingual_eval_set.csv # generated
│   └── models/
│       └── distilbert-sst2/          # Cached locally after download
├── notebooks/
│   ├── 02_model_testing.ipynb
│   ├── 03_evaluation.ipynb           # Pipeline evaluation + charts
│   ├── 04_classifier_benchmark.ipynb   # sklearn benchmark
│   └── translation_quality_evaluation.py
├── scripts/
│   ├── build_eval_dataset.py
│   ├── download_indic_sentiment_kn.py
│   ├── run_classifier_benchmark.py
│   └── run_pipeline_evaluation.py
├── src/
│   ├── evaluation/
│   │   ├── dataset.py
│   │   └── classifier_benchmark.py
│   ├── models/
│   │   ├── sentiment_classifier.py
│   │   ├── download_model.py
│   │   └── train_sentiment.py
│   ├── preprocessing/
│   │   ├── language_detector.py
│   │   ├── translator.py
│   │   └── transliterator.py
│   ├── pipeline.py
│   └── utils/
├── tests/                            # 216 unit tests
├── requirements.txt
└── packages.txt                      # Streamlit Cloud system deps
```

---

## References

### Models & Libraries

1. **DistilBERT / SST-2** — Sanh et al. (2019). arXiv:1910.01108. `distilbert-base-uncased-finetuned-sst-2-english`
2. **NLLB-200** — NLLB Team et al. (2022). arXiv:2207.04672. `facebook/nllb-200-distilled-600M`
3. **IndicXlit** — Madhani et al. (2022). arXiv:2205.03018. AI4Bharat
4. **indic-transliteration** — Vasuki (2019–2024). PyPI: `indic-transliteration`
5. **IndicTrans2** — Gala et al. (2023). arXiv:2305.16307. AI4Bharat
6. **langdetect** — Nakatani (2010). PyPI: `langdetect`
7. **Hugging Face Transformers** — Wolf et al. (2020). EMNLP 2020
8. **Streamlit** — https://streamlit.io

### Datasets

9. **Stanford Sentiment Treebank (SST-2)** — Socher et al. (2013). EMNLP 2013
10. **IndicSentiment** — AI4Bharat. Kannada product reviews for classifier benchmarking. https://huggingface.co/datasets/ai4bharat/IndicSentiment

---

## License

This project is developed for academic purposes as part of a university final project.
Pre-trained model weights are subject to their respective licenses (Apache 2.0 for DistilBERT;
CC-BY-NC for NLLB-200; MIT/CC for AI4Bharat models).
