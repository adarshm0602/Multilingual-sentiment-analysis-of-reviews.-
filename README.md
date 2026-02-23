# Multilingual Sentiment Analysis of Kannada Product Reviews

> An end-to-end NLP pipeline that classifies the sentiment of product reviews written in
> **Kannada script**, **Romanized Kannada** (Kanglish), or **English** — with a
> Streamlit web application for interactive exploration and batch processing.

---

## Table of Contents

1. [Problem Statement & Motivation](#problem-statement--motivation)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Command-Line Pipeline](#command-line-pipeline)
   - [Batch Processing](#batch-processing)
   - [Web Application](#web-application)
   - [Evaluation Notebook](#evaluation-notebook)
5. [Dataset](#dataset)
6. [Model Performance](#model-performance)
7. [Known Limitations](#known-limitations)
8. [Future Work](#future-work)
9. [References](#references)

---

## Problem Statement & Motivation

E-commerce platforms such as Amazon, Flipkart, and Meesho serve hundreds of millions of
Indian consumers, yet the vast majority of their sentiment analysis systems are
**English-only**.  This creates a significant blind spot: a large fraction of user
reviews are written in regional scripts or in transliterated "Hinglish/Kanglish" style
(e.g. *"tumba ketta product, waste of money"*).

Karnataka alone has over 60 million Kannada speakers.  Reviews can appear in three
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

The pipeline is a four-stage cascade.  Each stage is independently testable and
replaceable.

```
INPUT TEXT
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1 · Language Detection                                       │
│                                                                     │
│  • Unicode-block analysis (Kannada U+0C80–U+0CFF → script %)       │
│  • langdetect for romanized / English disambiguation                │
│  • Outputs: { language, confidence, script_proportions }            │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────────┐
          │                 │                     │
    kannada_script   romanized_kannada         english
          │                 │                (or mixed/unknown)
          │          ┌──────┴──────┐               │
          │          ▼             │               │
          │  ┌───────────────────┐ │               │
          │  │  Stage 2 ·        │ │               │
          │  │  Transliteration  │ │               │
          │  │                   │ │               │
          │  │  Tier 1: IndicXlit│ │               │
          │  │  (ai4bharat) *    │ │               │
          │  │  Tier 2: curated  │ │               │
          │  │  dict (166 words) │ │               │
          │  │  Tier 3: rule-    │ │               │
          │  │  based ITRANS→    │ │               │
          │  │  Kannada scheme   │ │               │
          │  └──────┬────────────┘ │               │
          │         │ Kannada      │               │
          │         └──────────────┘               │
          │                 │                      │
          ├─────────────────┘                      │
          │                                        │
          ▼                                        │
  ┌──────────────────────┐                         │
  │  Stage 3 · Translate │                         │
  │                      │                         │
  │  NLLB-200 600M       │                         │
  │  (default, offline)  │                         │
  │  or IndicTrans2      │                         │
  │  or MarianMT         │                         │
  │  or Google Translate │                         │
  │  or fallback dict    │                         │
  └──────┬───────────────┘                         │
         │ English                                 │
         └─────────────────────────────────────────┤
                                                   │
                                                   ▼
                                   ┌───────────────────────────┐
                                   │  Stage 4 · Sentiment      │
                                   │  Classification           │
                                   │                           │
                                   │  DistilBERT (SST-2)       │
                                   │  distilbert-base-uncased- │
                                   │  finetuned-sst-2-english  │
                                   │  (runs locally on CPU)    │
                                   │                           │
                                   │  → Positive / Negative /  │
                                   │    Neutral + confidence   │
                                   └───────────────────────────┘
```

> \* IndicXlit requires `fairseq`, which is incompatible with Python 3.11+.
>   The pipeline automatically falls through to the dictionary and scheme tiers.

### Key Components

| Component | Class | Description |
|-----------|-------|-------------|
| Language Detector | `LanguageDetector` | Unicode-block ratio + `langdetect`; returns `kannada_script`, `romanized_kannada`, `english`, `mixed`, or `unknown` |
| Transliterator | `Transliterator` | 3-tier: IndicXlit neural model → curated 166-word dict → rule-based ITRANS→Kannada (`indic-transliteration`); Romanized → Kannada |
| Translator | `Translator` | 5 backends: NLLB-200 (default, offline), IndicTrans2 (offline, gated), MarianMT (offline), Google Translate (online), fallback dictionary |
| Sentiment Classifier | `SentimentClassifier` | DistilBERT fine-tuned on Stanford SST-2; 3-class output (Positive / Negative / Neutral) |
| Pipeline Orchestrator | `KannadaSentimentPipeline` | Routes text, times each stage, collects errors; supports single and batch inference |

---

## Installation

### Prerequisites

- Python **3.10 – 3.12**
- macOS, Linux, or WSL2 on Windows
- ~1.5 GB disk space (DistilBERT ~270 MB + NLLB-200 ~600 MB + dependencies)
- 4 GB+ RAM (8 GB recommended)

### 1 · Clone the repository

```bash
git clone <repo-url>
cd "Multilingual-sentiment-analysis-of-reviews.-"
```

### 2 · Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate.bat       # Windows CMD
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on IndicXlit** — The `ai4bharat-transliteration` package requires
> `fairseq`, which has compatibility issues on Python 3.11+.  The pipeline
> automatically falls through to the built-in 166-word dictionary and then the
> `indic-transliteration` rule-based scheme; no manual intervention is needed.

### 4 · First-run model downloads

Two models are downloaded automatically on first use and cached locally:

| Model | Size | Cached at |
|-------|------|-----------|
| DistilBERT-SST-2 | ~270 MB | `data/models/distilbert-sst2/` |
| NLLB-200 distilled 600M | ~600 MB | `~/.cache/huggingface/hub/` |

Both downloads require an internet connection on first run only.  All subsequent
runs are fully offline.

> **Tip** — When you first open the **Batch Upload** tab, a spinner reads
> *"Loading NLLB-200 600M…"*.  Wait for it to complete before clicking
> **Process All** to avoid a silent multi-minute delay mid-batch.

### 5 · Verify the installation

```bash
python3 -m pytest tests/ -v
```

Expected output: **212 passed** in under 15 seconds.

---

## Usage

### Command-Line Pipeline

```python
from src.pipeline import KannadaSentimentPipeline

# Default: NLLB-200 offline translation (high quality, ~600 MB first download)
pipeline = KannadaSentimentPipeline(
    translation_backend="nllb",       # or "marian" / "google" / "fallback"
    use_transliteration_model=False,  # True requires IndicXlit (Python ≤3.10)
    auto_fallback=True,
)

# Kannada script — translated by NLLB, then classified
result = pipeline.process("ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ")
print(result["sentiment_label"])      # Positive
print(result["confidence_score"])     # 0.9998
print(result["detected_language"])    # kannada_script
print(result["translated_text"])      # This product is very good
print(result["pipeline_steps"])       # ['language_detection', 'translation', 'sentiment_classification']
print(result["timings"])              # {'language_detection': 0.0001, 'translation': 5.2, ...}

# Romanized Kannada — transliterated then translated
result = pipeline.process("tumba chennagide, olle product")
print(result["sentiment_label"])      # Positive
print(result["transliterated_text"]) # ತುಂಬ ಚೆನ್ನಾಗಿದೆ, ಒಳ್ಳೆ ಪ್ರಾಡಕ್ಟ್

# English — routed directly to sentiment classifier (no translation needed)
result = pipeline.process("Terrible packaging, damaged on arrival.")
print(result["sentiment_label"])      # Negative
```

### Batch Processing

```python
import pandas as pd
from src.pipeline import KannadaSentimentPipeline

pipeline = KannadaSentimentPipeline(translation_backend="nllb")

# From a list of strings
texts = [
    "ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ",
    "tumba ketta product",
    "Fast delivery, great quality!",
]
df = pipeline.process_batch(texts, show_progress=True)
print(df[["original_text", "sentiment_label", "confidence_score", "status"]])

# From a CSV / DataFrame
df_in = pd.read_csv("data/demo_data.csv")
df_out = pipeline.process_batch(
    df_in,
    text_column="review",         # column containing the review text
    n_workers=4,                  # parallel workers (default: min(4, cpu//2))
    chunk_size=10,                # items per submission wave
    use_multiprocessing=True,     # False → thread pool
)
df_out.to_csv("results.csv", index=False)
```

**Output DataFrame columns**

| Column | Type | Description |
|--------|------|-------------|
| `original_text` | str | Unmodified input |
| `detected_language` | str | `kannada_script` / `romanized_kannada` / `english` / `mixed` / `unknown` |
| `detection_confidence` | float | 0–1 |
| `transliterated_text` | str \| None | Only for romanized input |
| `translated_text` | str \| None | Only for Kannada script input |
| `sentiment_label` | str | `Positive` / `Negative` / `Neutral` |
| `confidence_score` | float | 0–1 |
| `pipeline_steps` | list | Stages executed |
| `errors` | list | Any non-fatal warnings |
| `processing_time_s` | float | Wall-clock time per row |
| `status` | str | `ok` / `ok_with_warnings` / `error` / `skipped` |

### Web Application

```bash
# From the project root
venv/bin/streamlit run app/streamlit_app.py
```

Then open **http://localhost:8501** in your browser.

**Sidebar — Translation Engine**

Choose from five backends:

| Option | Description |
|--------|-------------|
| **NLLB-200 600M** *(default)* | Offline, free, high quality — Meta's NLLB model, ~600 MB |
| IndicTrans2 | Offline, highest quality — AI4Bharat model, ~3 GB, gated access |
| MarianMT | Offline, free, lower quality — Helsinki-NLP opus-mt-kn-en |
| Google Translate | Online — requires `GOOGLE_TRANSLATE_API_KEY` environment variable |
| Dictionary Fallback | Offline, instant — 166-entry vocabulary, lowest accuracy |

**Single Review tab**

1. Type or paste a review, or click a sample button (Kannada / Romanized / English)
2. Click **Analyze Sentiment**
3. Results show: detected language, pipeline steps, sentiment badge, confidence
   gauge, transliteration card, and translation card

**Batch Upload tab**

1. Click **Load Demo Data** to use the built-in 20-review CSV, **or** upload your
   own `.csv` / `.xlsx` file
2. Select the column containing review text
3. Click **Process All** — a live progress bar shows per-review status
4. View the **Summary Dashboard** (sentiment pie chart, language bar chart,
   confidence histogram) and the coloured **Full Results Table**
5. Download all results as CSV with the **Download** button

### Evaluation Notebook

```bash
# Execute all cells (requires venv with nbconvert)
venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=300 \
    notebooks/03_evaluation.ipynb

# Or open interactively
venv/bin/jupyter notebook notebooks/03_evaluation.ipynb
```

The notebook produces:
- Per-language accuracy tables
- Confusion matrix heat-map
- Processing-time violin plots by script type
- Error-pattern analysis scatter plot
- Language-detection accuracy stacked bar chart
- Saved results at `data/processed/evaluation_results.json`

---

## Dataset

### Evaluation Dataset (50 reviews, hand-labeled)

The evaluation set was constructed manually to ensure balanced coverage across
language types and sentiment classes.

| Subset | Reviews | Source |
|--------|---------|--------|
| Demo dataset | 20 | `data/demo_data.csv` — crafted examples covering Electronics, Clothing, Home & Kitchen, Books, and Sports |
| Extended set | 30 | Additional examples covering product categories not in the demo set |
| **Total** | **50** | |

**Script-type distribution**

| Script Type | Count | % |
|-------------|-------|---|
| Kannada Script (native) | 20 | 40% |
| Romanized Kannada | 15 | 30% |
| English | 15 | 30% |

**Sentiment distribution** (ground-truth labels derived from star ratings)

| Sentiment | Count | % | Rating mapping |
|-----------|-------|---|----------------|
| Positive | 24 | 48% | ★★★★☆ – ★★★★★ (4–5 stars) |
| Negative | 24 | 48% | ★☆☆☆☆ – ★★☆☆☆ (1–2 stars) |
| Neutral | 2 | 4% | ★★★☆☆ (3 stars) |

### Pre-trained Models Used

| Model | Source | Purpose |
|-------|--------|---------|
| `distilbert-base-uncased-finetuned-sst-2-english` | HuggingFace Hub | Sentiment classification |
| `facebook/nllb-200-distilled-600M` | Meta AI / HuggingFace Hub | Kannada → English translation (default) |
| `ai4bharat/IndicXlit` | AI4Bharat | Romanized Kannada → Kannada transliteration (neural tier) |
| `indic-transliteration` | Sanskrit Computational Linguistics | Rule-based ITRANS → Kannada script (fallback tier) |
| `ai4bharat/IndicTrans2` | AI4Bharat | Kannada → English translation (optional, high quality) |
| langdetect 1.0.9 | Google Language Detection | Language identification signal |

> All models run **locally** on CPU — no inference API calls are made at runtime.

---

## Model Performance

> Evaluation conducted on the 50-review hand-labeled dataset using the
> **fallback translation backend** (dictionary-based; no neural translation).
> Results reflect the end-to-end pipeline including language detection,
> transliteration, translation, and sentiment classification.
> Switching to the NLLB-200 backend is expected to raise Kannada-script accuracy
> significantly by replacing the 166-word fallback dictionary with full neural MT.

### Overall Accuracy

| Metric | Value |
|--------|-------|
| Overall accuracy | **74.0%** |
| Total reviews | 50 |
| Correct predictions | 37 / 50 |

### Per-Language Accuracy

| Script Type | Reviews | Correct | Accuracy | Avg Confidence | Avg Latency |
|-------------|---------|---------|----------|----------------|-------------|
| Kannada Script | 20 | 15 | **75.0%** | 94.5% | 111.5 ms |
| Romanized Kannada | 15 | 8 | **53.3%** | 98.4% | 41.0 ms |
| English | 15 | 14 | **93.3%** | 99.98% | 20.4 ms |

> With the NLLB-200 backend the average latency for Kannada-script reviews rises to
> ~5 s per review (neural translation), but translation quality improves substantially
> compared to the fallback dictionary.

### Per-Class Metrics

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| Positive | 0.875 | 0.583 | **0.700** | 24 |
| Negative | 0.676 | 0.958 | **0.793** | 24 |
| Neutral | 0.000 | 0.000 | 0.000 | 2 |
| **Macro avg** | 0.517 | 0.514 | **0.498** | 50 |

### Confusion Matrix

```
                 Predicted
              Pos   Neg   Neu
Actual Pos  [  14    10     0 ]
       Neg  [   1    23     0 ]
       Neu  [   1     1     0 ]
```

The model has a **negative-skew** bias: it confidently identifies negative reviews
(96% recall) but struggles to distinguish positive Kannada-translated reviews from
negative ones, likely due to imperfect fallback translation.

### Language Detection Accuracy

| Script Type | Detection Accuracy | Notes |
|-------------|-------------------|-------|
| Kannada Script | **100.0%** | Unicode block detection is deterministic |
| English | **100.0%** | langdetect performs well on pure English |
| Romanized Kannada | **26.7%** | Frequently mis-detected as English (73% of cases) |

### Processing Latency (end-to-end, per review)

| Backend | Mean | Median | Min | Max |
|---------|------|--------|-----|-----|
| Fallback dict | 63 ms | 21 ms | 17 ms | 1863 ms (cold-start) |
| NLLB-200 | ~4.2 s | ~5 s | ~18 ms (English) | ~6 s |

> The 1.86 s maximum for the fallback backend is the **one-time model warm-up**
> on first call.  NLLB-200 translates each Kannada/Romanized review in ~5 s;
> English reviews bypass translation and run in ~20 ms regardless of backend.

---

## Known Limitations

1. **Romanized Kannada detection is unreliable (26.7% accuracy)**
   `langdetect` treats Kanglish text as English because both share the Latin script.
   The current heuristic (dictionary word-match threshold) catches only 4 of 15
   romanized reviews.  False-negatives route through the English path, skipping
   transliteration.

2. **Neutral class is effectively unsupported**
   The underlying DistilBERT-SST-2 model is a binary classifier (Positive /
   Negative).  Neutral is inferred from low-confidence borderline outputs, but the
   evaluation set has only 2 neutral examples — making this class unreliable.

3. **No native Kannada sentiment model**
   The pipeline translates Kannada → English and then applies an English-only
   sentiment model.  Translation errors propagate to the final label.  A model
   fine-tuned directly on Kannada or a multilingual model (IndicBERT, MuRIL) would
   eliminate this source of error.

4. **IndicXlit incompatibility with Python 3.11+**
   The `fairseq` dependency required by IndicXlit does not support Python 3.11+.
   The neural transliteration tier is therefore unavailable; the system falls back
   to the curated 166-word dictionary and then the `indic-transliteration` rule-based
   scheme automatically.

5. **ITRANS rule-based transliteration is imperfect for colloquial words**
   The `indic-transliteration` scheme assumes standard ITRANS encoding.  Informal
   Kanglish spellings (e.g. *"tumba"* vs. ITRANS *"tumba"*) may produce unexpected
   Unicode output.  The curated dictionary takes priority for known loanwords.

6. **Small evaluation set (50 reviews)**
   Performance estimates carry wide confidence intervals.  Results should be
   interpreted as indicative rather than definitive.

7. **Multiprocessing spawn limitation in Jupyter**
   `process_batch()` automatically falls back to `ThreadPoolExecutor` when called
   from an interactive session (Jupyter / IPython) due to macOS `spawn` context
   restrictions.

8. **NLLB-200 latency (~5 s/review)**
   Running on CPU, NLLB-200 takes ~5 seconds per Kannada review.  A 20-review
   demo batch completes in roughly 90 seconds.  GPU acceleration would reduce this
   to under 0.5 s/review.

---

## Future Work

- **Romanized Kannada detection**: Train a lightweight classifier (fastText or a
  small CNN) on a labeled corpus of Kanglish vs. English text to replace the current
  heuristic.

- **Native Kannada sentiment model**: Fine-tune IndicBERT or MuRIL directly on a
  Kannada-language sentiment dataset (e.g., from Indic NLP datasets or scraped from
  Flipkart/Amazon reviews) to eliminate the translate-then-classify bottleneck.

- **Larger labeled dataset**: Crowdsource or scrape 1,000+ labeled Kannada reviews
  across all three script types for more reliable evaluation.

- **Neutral class**: Collect sufficient neutral examples and fine-tune a 3-class
  sentiment head to handle ambiguous reviews properly.

- **GPU acceleration**: Add `device="cuda"` support for NLLB-200 and DistilBERT to
  reduce per-review latency from ~5 s to under 0.5 s.

- **REST API**: Expose the pipeline as a FastAPI service so it can be integrated
  into e-commerce backends without requiring Python client code.

- **Confidence calibration**: Apply temperature scaling to the sentiment softmax
  outputs — the model is currently over-confident (mean confidence 96% even on
  incorrect predictions).

- **Support additional Indic languages**: The architecture generalises naturally to
  Telugu, Tamil, Malayalam, and Hindi with language-specific transliteration
  dictionaries and translation model swap.

---

## Project Structure

```
.
├── .streamlit/
│   └── config.toml               # Streamlit server config (runOnSave=false, etc.)
├── app/
│   └── streamlit_app.py          # Streamlit web application
├── config.yaml                   # Global configuration (default backend: nllb)
├── data/
│   ├── demo_data.csv             # 20-review labelled demo dataset
│   ├── models/
│   │   └── distilbert-sst2/     # Cached DistilBERT sentiment model (~270 MB)
│   └── processed/
│       ├── evaluation_results.json  # Full pipeline evaluation (50 reviews)
│       └── translation_quality.csv  # Translation quality scores
├── notebooks/
│   ├── 02_model_testing.ipynb   # Sentiment classifier exploration
│   ├── 03_evaluation.ipynb      # End-to-end evaluation with charts
│   └── translation_quality_evaluation.py  # Standalone translation evaluator
├── pytest.ini                   # Test discovery configuration
├── requirements.txt
├── src/
│   ├── models/
│   │   └── sentiment_classifier.py   # DistilBERT wrapper + SentimentResult
│   ├── pipeline.py                   # KannadaSentimentPipeline orchestrator
│   ├── preprocessing/
│   │   ├── language_detector.py      # LanguageDetector + DetectionResult
│   │   ├── translator.py             # Translator (5 backends) + TranslationResult
│   │   └── transliterator.py         # Transliterator (3-tier) + TransliterationResult
│   └── utils/
└── tests/
    ├── conftest.py               # Shared fixtures and helpers
    ├── test_language_detector.py # 76 unit tests
    ├── test_pipeline.py          # 59 end-to-end tests
    └── test_transliterator.py    # 77 unit tests
```

---

## References

### Models & Libraries

1. **DistilBERT / SST-2**
   Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019).
   *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.*
   arXiv:1910.01108. HuggingFace model: `distilbert-base-uncased-finetuned-sst-2-english`.

2. **NLLB-200**
   NLLB Team, Costa-jussà, M. R., Cross, J., Çelebi, O., et al. (2022).
   *No Language Left Behind: Scaling Human-Centered Machine Translation.*
   arXiv:2207.04672. Meta AI. HuggingFace model: `facebook/nllb-200-distilled-600M`.

3. **IndicXlit**
   Madhani, Y., Parthan, S., Bedekar, P., Khapra, M. M., et al. (2022).
   *Aksharantar: Advancing Romanization for South Asian Languages.*
   arXiv:2205.03018. AI4Bharat.

4. **indic-transliteration**
   Vishvas Vasuki (2019–2024).
   *indic_transliteration — Transliteration of Indic scripts.*
   PyPI: `indic-transliteration`. https://github.com/indic-transliteration/indic_transliteration_py

5. **IndicTrans2**
   Gala, J., Chitale, P. A., AK, R., Raghavan, V., Doddapaneni, S.,
   Kumar, A., … & Khapra, M. M. (2023).
   *IndicTrans2: Towards High-Quality and Accessible Machine Translation Models
   for all 22 Scheduled Indian Languages.*
   arXiv:2305.16307. AI4Bharat.

6. **langdetect**
   Nakatani, S. (2010). *langdetect* — Language detection library ported from
   Google's language-detection library.  PyPI: `langdetect`.

7. **Indic NLP Library**
   Kunchukuttan, A. (2020).
   *The IndicNLP Library.* AI4Bharat.
   https://github.com/anoopkunchukuttan/indic_nlp_library

### Datasets & Benchmarks

8. **Stanford Sentiment Treebank (SST-2)**
   Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., &
   Potts, C. (2013).
   *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.*
   EMNLP 2013.

9. **AI4Bharat Indic NLP Datasets**
   AI4Bharat (2022). *IndicNLP Catalog* — resources for Indian languages.
   https://github.com/AI4Bharat/indicnlp_catalog

### Frameworks

10. **Hugging Face Transformers**
    Wolf, T., Debut, L., Sanh, V., et al. (2020).
    *Transformers: State-of-the-Art Natural Language Processing.*
    EMNLP 2020 (Systems Demonstrations). arXiv:1910.03771.

11. **Streamlit**
    Streamlit Inc. (2019–2024). *Streamlit — The fastest way to build data apps.*
    https://streamlit.io

12. **PyTorch**
    Paszke, A., Gross, S., Massa, F., et al. (2019).
    *PyTorch: An Imperative Style, High-Performance Deep Learning Library.*
    NeurIPS 2019. arXiv:1912.01703.

---

## License

This project is developed for academic purposes as part of a university final project.
All pre-trained model weights are used subject to their respective licenses
(Apache 2.0 for DistilBERT / Transformers; CC-BY-NC for NLLB-200; MIT/CC for AI4Bharat models).
