"""
Kannada Sentiment Analysis Pipeline.

This module is the top-level orchestrator for the multilingual sentiment
analysis system.  It wires together the four processing stages — language
detection, transliteration, translation, and sentiment classification —
and provides both a single-item API (:meth:`KannadaSentimentPipeline.process`)
and a parallel batch API (:meth:`KannadaSentimentPipeline.process_batch`).

Routing logic
-------------
Each input text is routed through a different preprocessing chain depending
on its detected language / script:

::

    Input text
        │
        ▼  Stage 1: LanguageDetector
        ├─► "kannada_script"    → Stage 3 (translate) → Stage 4 (classify)
        ├─► "romanized_kannada" → Stage 2 (transliterate) → Stage 3 → Stage 4
        ├─► "english"           → Stage 4 directly
        └─► "mixed" / "unknown" → Stage 4 (best-effort, warning logged)

Parallel batch processing
-------------------------
:meth:`~KannadaSentimentPipeline.process_batch` uses
:class:`~concurrent.futures.ProcessPoolExecutor` by default.  Each worker
process receives only lightweight constructor kwargs; the ML models are
**not** pickled — they are re-initialised inside each worker via the
module-level :func:`_worker_initializer` / :func:`_worker_process_one`
pattern (required for ``spawn``-based multiprocessing on macOS / Windows).

In interactive sessions (Jupyter, IPython) where ``spawn`` cannot import
``__main__``, the implementation falls back automatically to a
:class:`~concurrent.futures.ThreadPoolExecutor`.

Public symbols
--------------
* :class:`PipelineError`              — unrecoverable pipeline exception.
* :class:`KannadaSentimentPipeline`   — main pipeline class.
* :func:`load_pipeline_from_config`   — factory using ``config.yaml``.

Module-level helpers (not part of the public API but needed for pickling)
--------------------------------------------------------------------------
* :func:`_worker_initializer`  — called once per worker process at start-up.
* :func:`_worker_process_one`  — processes a single ``(index, text)`` pair.

Example
-------
>>> from src.pipeline import KannadaSentimentPipeline
>>> pipeline = KannadaSentimentPipeline(
...     translation_backend="fallback",
...     use_transliteration_model=False,
... )

>>> # Single review
>>> result = pipeline.process("ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ")
>>> result["sentiment_label"]
'Positive'
>>> result["pipeline_steps"]
['language_detection', 'translation', 'sentiment_classification']

>>> # Batch
>>> df = pipeline.process_batch(
...     ["Great quality!", "tumba ketta", "ಚೆನ್ನಾಗಿದೆ"],
...     show_progress=False,
... )
>>> list(df["sentiment_label"])
['Positive', 'Negative', 'Positive']
"""

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.preprocessing.language_detector import DetectionResult, LanguageDetector
from src.preprocessing.transliterator import Transliterator
from src.preprocessing.translator import Translator
from src.models.sentiment_classifier import SentimentClassifier, SentimentResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies — gracefully absent at import time.
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PANDAS_AVAILABLE = False

try:
    from tqdm.auto import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TQDM_AVAILABLE = False


class _NoOpBar:
    """Silent progress-bar stand-in used when tqdm is unavailable.

    Implements the same interface as a ``tqdm`` instance so callers need
    not check for ``None`` before calling :meth:`update`, :meth:`set_postfix`,
    or :meth:`close`.  All methods are no-ops.
    """
    def __init__(self, *_, **__): pass
    def update(self, n=1): pass
    def set_postfix(self, **__): pass
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ---------------------------------------------------------------------------
# CPU-count heuristics
# ---------------------------------------------------------------------------
_CPU_COUNT: int = os.cpu_count() or 2

# Each worker process loads its own copy of every ML model, so more workers
# means more RAM.  Cap at 4 and never exceed half the logical CPUs so the OS
# and other processes retain breathing room.
_DEFAULT_WORKERS: int = max(1, min(4, _CPU_COUNT // 2))


# ---------------------------------------------------------------------------
# Module-level worker helpers (must be top-level for pickling under "spawn")
# ---------------------------------------------------------------------------
# These are intentionally NOT methods — ProcessPoolExecutor pickles the
# function by its qualified name, which requires module-level placement.

_worker_pipeline: Optional[Any] = None  # KannadaSentimentPipeline, set per worker


def _worker_initializer(pipeline_kwargs: Dict[str, Any]) -> None:
    """
    Recreate the pipeline once inside each worker process.

    Called by ProcessPoolExecutor at worker start-up.  Stores the result in
    the module-global ``_worker_pipeline`` so ``_worker_process_one`` can
    reach it without pickling the live object.
    """
    global _worker_pipeline
    # Suppress verbose per-item logging in workers; the main process reports
    # batch-level progress through tqdm.
    logging.getLogger("src").setLevel(logging.WARNING)
    _worker_pipeline = KannadaSentimentPipeline(**pipeline_kwargs)


def _worker_process_one(indexed_text: tuple) -> tuple:
    """
    Process a single ``(original_index, text)`` pair in a worker process.

    Returns
    -------
    tuple
        ``(original_index, result_dict_or_None, error_str_or_None)``
    """
    idx, text = indexed_text
    try:
        result = _worker_pipeline.process(text)  # type: ignore[union-attr]
        return idx, result, None
    except Exception as exc:
        return idx, None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Result-flattening helpers
# ---------------------------------------------------------------------------

_SKIPPED_STATUS = "skipped"
_ERROR_STATUS = "error"
_WARN_STATUS = "ok_with_warnings"
_OK_STATUS = "ok"

_DF_COLUMNS = [
    "original_text",
    "detected_language",
    "detection_confidence",
    "transliterated_text",
    "translated_text",
    "sentiment_label",
    "confidence_score",
    "pipeline_steps",
    "errors",
    "processing_time_s",
    "status",
]

_EMPTY_ROW: Dict[str, Any] = {
    "original_text": None,
    "detected_language": "unknown",
    "detection_confidence": 0.0,
    "transliterated_text": None,
    "translated_text": None,
    "sentiment_label": "Neutral",
    "confidence_score": 0.0,
    "pipeline_steps": "",
    "errors": "",
    "processing_time_s": 0.0,
    "status": _ERROR_STATUS,
}


def _flatten(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a :meth:`~KannadaSentimentPipeline.process` result dict to a flat DataFrame row.

    Args:
        result: The dict returned by :meth:`KannadaSentimentPipeline.process`.

    Returns:
        A flat dict whose keys match :data:`_DF_COLUMNS`.  List / dict
        fields (``pipeline_steps``, ``errors``, ``timings``) are collapsed
        to comma-/semicolon-joined strings and a single float respectively.
    """
    errors = result.get("errors", [])
    return {
        "original_text": result.get("original_text"),
        "detected_language": result.get("detected_language", "unknown"),
        "detection_confidence": result.get("detection_confidence", 0.0),
        "transliterated_text": result.get("transliterated_text"),
        "translated_text": result.get("translated_text"),
        "sentiment_label": result.get("sentiment_label", "Neutral"),
        "confidence_score": result.get("confidence_score", 0.0),
        "pipeline_steps": ", ".join(result.get("pipeline_steps", [])),
        "errors": "; ".join(errors),
        "processing_time_s": result.get("timings", {}).get("total", 0.0),
        "status": _WARN_STATUS if errors else _OK_STATUS,
    }


def _error_row(text: Any, error: str) -> Dict[str, Any]:
    """Build a DataFrame row representing a failed processing attempt.

    Args:
        text: The original input value (may be non-string if validation failed).
        error: Human-readable error description.

    Returns:
        A flat dict with ``status='error'`` and default sentinel values for
        all analysis fields.
    """
    row = dict(_EMPTY_ROW)
    row["original_text"] = text
    row["errors"] = error
    row["status"] = _ERROR_STATUS
    return row


def _skipped_row(text: Any, reason: str) -> Dict[str, Any]:
    """Build a DataFrame row for an item that was skipped before processing.

    Items are skipped when they are not strings or are empty / whitespace-only.

    Args:
        text: The original input value.
        reason: Human-readable reason for skipping.

    Returns:
        A flat dict with ``status='skipped'`` and default sentinel values.
    """
    row = dict(_EMPTY_ROW)
    row["original_text"] = text
    row["errors"] = reason
    row["status"] = _SKIPPED_STATUS
    return row


def _make_pbar(total: int, show: bool) -> Any:
    """Create a tqdm progress bar or a silent :class:`_NoOpBar`.

    Args:
        total: Total number of items to track.
        show: If ``True`` and tqdm is installed, return a live progress bar.
            If ``False`` or tqdm is unavailable, return a :class:`_NoOpBar`.

    Returns:
        A tqdm bar (or compatible no-op object) that supports ``update()``,
        ``set_postfix()``, ``close()``, and context-manager protocol.
    """
    if show and _TQDM_AVAILABLE:
        return _tqdm(
            total=total,
            desc="Processing",
            unit="text",
            dynamic_ncols=True,
            colour="green",
        )
    return _NoOpBar()


def _in_interactive_session() -> bool:
    """
    Return True when running inside a REPL / Jupyter notebook.

    Under ``spawn`` (macOS default), ProcessPoolExecutor requires the
    ``__main__`` module to be importable — which is not guaranteed in
    interactive sessions.  We fall back to threads in that case.
    """
    try:
        import __main__
        return not hasattr(__main__, "__file__")
    except Exception:
        return True


# ===========================================================================
# Main pipeline class
# ===========================================================================

class PipelineError(Exception):
    """Raised when the pipeline encounters an unrecoverable error."""


class KannadaSentimentPipeline:
    """
    End-to-end pipeline for multilingual Kannada sentiment analysis.

    Detects the script / language of raw input text and routes it through
    the correct preprocessing steps before running sentiment classification.

    Supported routes
    ----------------
    * **Kannada script** (ಕನ್ನಡ)  → translate → classify
    * **Romanized Kannada**         → transliterate → translate → classify
    * **English**                   → classify directly
    * **Mixed / Unknown**           → classify directly (best-effort)

    Parameters
    ----------
    translation_backend:
        Which translation engine to use.
        One of ``'indictrans2'`` (offline), ``'google'`` (requires API key),
        or ``'fallback'`` (dictionary-based).
    use_transliteration_model:
        Whether to load the IndicXlit neural model for transliteration.
        When ``False`` (or the model is unavailable), falls back to the
        built-in dictionary.
    classifier_model_name:
        HuggingFace model identifier for the sentiment classifier.
        Defaults to ``distilbert-base-uncased-finetuned-sst-2-english``.
    classifier_model_path:
        Local directory containing a downloaded classifier model.
        Takes priority over ``classifier_model_name`` when both are given.
    google_api_key:
        Google Cloud Translate API key. Only required when
        ``translation_backend='google'``.
    auto_fallback:
        If ``True``, the translator will automatically fall back to simpler
        backends if the primary one is unavailable.

    Example
    -------
    >>> pipeline = KannadaSentimentPipeline()
    >>> result = pipeline.process("ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ")
    >>> print(result['sentiment_label'], result['confidence_score'])
    Positive 0.9512
    """

    def __init__(
        self,
        translation_backend: str = "indictrans2",
        use_transliteration_model: bool = True,
        classifier_model_name: Optional[str] = None,
        classifier_model_path: Optional[str] = None,
        google_api_key: Optional[str] = None,
        auto_fallback: bool = True,
    ) -> None:
        """Initialise all pipeline components.

        Each component is constructed eagerly; the sentiment classifier's
        underlying model is loaded lazily on the first :meth:`process` call.

        Args:
            translation_backend: Which translation engine to use.
                One of ``'indictrans2'`` (default, offline), ``'google'``
                (requires API key and internet), or ``'fallback'``
                (dictionary-based, always available).
            use_transliteration_model: Whether to attempt loading the
                IndicXlit neural transliteration model.  Set to ``False``
                to use only the built-in fallback dictionary (no heavy
                dependencies, suitable for testing).
            classifier_model_name: HuggingFace model identifier passed to
                :class:`~src.models.sentiment_classifier.SentimentClassifier`.
                Defaults to ``'distilbert-base-uncased-finetuned-sst-2-english'``.
            classifier_model_path: Absolute path to a locally cached model
                directory.  When supplied, the model is loaded from disk
                instead of being downloaded from HuggingFace.
            google_api_key: Google Cloud Translate API key.  Only required
                when ``translation_backend='google'``.  Can also be supplied
                via the ``GOOGLE_TRANSLATE_API_KEY`` environment variable.
            auto_fallback: If ``True`` (default), the :class:`Translator`
                will automatically fall back to simpler backends when the
                preferred one is unavailable.

        Example:
            >>> # Fully offline, no heavy model downloads
            >>> pipeline = KannadaSentimentPipeline(
            ...     translation_backend="fallback",
            ...     use_transliteration_model=False,
            ... )
            >>> pipeline.process("Hello world")["sentiment_label"]
            'Positive'
        """
        logger.info("Initializing KannadaSentimentPipeline…")

        t0 = time.perf_counter()
        self.detector = LanguageDetector()
        logger.info(f"LanguageDetector ready ({_ms(t0)})")

        t0 = time.perf_counter()
        self.transliterator = Transliterator(use_model=use_transliteration_model)
        logger.info(
            f"Transliterator ready "
            f"(model={'yes' if self.transliterator.model_available else 'no'}) "
            f"({_ms(t0)})"
        )

        t0 = time.perf_counter()
        self.translator = Translator(
            backend=translation_backend,
            google_api_key=google_api_key,
            auto_fallback=auto_fallback,
        )
        logger.info(
            f"Translator ready "
            f"(active_backend={self.translator.active_backend.value}) "
            f"({_ms(t0)})"
        )

        t0 = time.perf_counter()
        self.classifier = SentimentClassifier(
            model_name=classifier_model_name,
            model_path=classifier_model_path,
        )
        logger.info(f"SentimentClassifier ready ({_ms(t0)})")

        # Lightweight kwargs stored so process_batch can rebuild the pipeline
        # inside each worker process without pickling the live objects.
        self._init_kwargs: Dict[str, Any] = {
            "translation_backend": translation_backend,
            "use_transliteration_model": use_transliteration_model,
            "classifier_model_name": classifier_model_name,
            "classifier_model_path": classifier_model_path,
            "google_api_key": google_api_key,
            "auto_fallback": auto_fallback,
        }

        logger.info("KannadaSentimentPipeline initialized successfully.")

    # ------------------------------------------------------------------
    # Single-item API
    # ------------------------------------------------------------------

    def process(self, text: str) -> Dict[str, Any]:
        """
        Run the full sentiment-analysis pipeline on a single text.

        Parameters
        ----------
        text:
            Raw input string in any supported language / script.

        Returns
        -------
        dict with keys:

        ``original_text`` : str
            The unmodified input.
        ``detected_language`` : str
            One of ``'kannada_script'``, ``'romanized_kannada'``,
            ``'english'``, ``'mixed'``, ``'unknown'``.
        ``detection_confidence`` : float
            Confidence score for the language detection (0–1).
        ``transliterated_text`` : str or None
            Kannada-script form of romanized input; ``None`` if not applicable.
        ``translated_text`` : str or None
            English translation of Kannada text; ``None`` if not applicable.
        ``sentiment_label`` : str
            ``'Positive'``, ``'Negative'``, or ``'Neutral'``.
        ``confidence_score`` : float
            Classifier confidence for the predicted label (0–1).
        ``pipeline_steps`` : list[str]
            Ordered list of processing steps that were executed.
        ``errors`` : list[str]
            Non-fatal error messages (empty list if everything succeeded).
        ``timings`` : dict[str, float]
            Wall-clock time (seconds) for each step and the total run.
        """
        pipeline_start = time.perf_counter()

        result: Dict[str, Any] = {
            "original_text": text,
            "detected_language": "unknown",
            "detection_confidence": 0.0,
            "transliterated_text": None,
            "translated_text": None,
            "sentiment_label": "Neutral",
            "confidence_score": 0.0,
            "pipeline_steps": [],
            "errors": [],
            "timings": {},
        }

        logger.info("=" * 60)
        logger.info("Pipeline START")
        logger.info(f"Input ({len(text)} chars): {text!r}")

        # Step 1: Language detection
        detection = self._detect_language(text, result)

        # Steps 2–N: Route through the appropriate preprocessing chain
        text_for_sentiment = self._route(text, detection, result)

        # Final step: Sentiment classification
        if text_for_sentiment:
            self._classify_sentiment(text_for_sentiment, result)
        else:
            msg = "No text available for sentiment classification."
            result["errors"].append(msg)
            logger.error(msg)

        total = time.perf_counter() - pipeline_start
        result["timings"]["total"] = round(total, 4)

        logger.info(
            f"Pipeline DONE in {total*1000:.1f}ms | "
            f"lang={result['detected_language']} | "
            f"sentiment={result['sentiment_label']} "
            f"({result['confidence_score']:.2%})"
        )
        if result["errors"]:
            logger.warning(f"Non-fatal errors: {result['errors']}")
        logger.info("=" * 60)

        return result

    # ------------------------------------------------------------------
    # Batch API
    # ------------------------------------------------------------------

    def process_batch(
        self,
        inputs: Union[List[str], Any],  # Any covers pd.DataFrame
        text_column: str = "text",
        n_workers: Optional[int] = None,
        chunk_size: int = 10,
        show_progress: bool = True,
        use_multiprocessing: bool = True,
    ) -> Any:  # returns pd.DataFrame
        """
        Process a list of texts or a DataFrame in parallel.

        Parallelism strategy
        --------------------
        *  By default ``use_multiprocessing=True`` triggers a
           ``ProcessPoolExecutor``.  Each worker process gets its own
           pipeline instance (models are **not** pickled; only the lightweight
           constructor kwargs are).  True multi-core parallelism is achieved
           because worker processes are independent of the GIL.
        *  In interactive / Jupyter sessions (where ``spawn`` cannot import
           ``__main__``), or when ``use_multiprocessing=False``, the method
           falls back to a ``ThreadPoolExecutor``.  PyTorch releases the GIL
           during tensor operations, so threads still provide real speedup for
           the inference-heavy steps.
        *  Worker count is capped at ``min(4, cpu_count // 2)`` to prevent
           RAM exhaustion from multiple model copies and to leave headroom for
           the OS.

        Parameters
        ----------
        inputs:
            Either a plain ``list[str]`` or a ``pandas.DataFrame``.
            When a DataFrame is passed, *text_column* names the column to read.
        text_column:
            Column name to use when *inputs* is a DataFrame.  Default ``"text"``.
        n_workers:
            Number of parallel workers.  ``None`` (default) auto-selects
            ``min(4, cpu_count // 2)``.
        chunk_size:
            Number of items grouped into each worker task.  Larger values
            reduce scheduling overhead for big batches; smaller values give
            finer-grained progress updates.  Default ``10``.
        show_progress:
            Display a tqdm progress bar.  Default ``True``.
        use_multiprocessing:
            ``True`` (default) → attempt ``ProcessPoolExecutor`` first.
            ``False`` → always use ``ThreadPoolExecutor``.

        Returns
        -------
        pandas.DataFrame
            One row per input (including skipped / errored rows) with columns:

            ``original_text``, ``detected_language``, ``detection_confidence``,
            ``transliterated_text``, ``translated_text``, ``sentiment_label``,
            ``confidence_score``, ``pipeline_steps``, ``errors``,
            ``processing_time_s``, ``status``.

            *status* is one of ``"ok"``, ``"ok_with_warnings"``, ``"error"``,
            ``"skipped"``.

        Raises
        ------
        ImportError
            If pandas is not installed.

        Example
        -------
        >>> pipeline = KannadaSentimentPipeline()
        >>> df = pipeline.process_batch(
        ...     ["This is great!", "ಚೆನ್ನಾಗಿದೆ", "tumba olle"],
        ...     show_progress=False,
        ... )
        >>> print(df[["original_text", "sentiment_label", "confidence_score"]])
        """
        if not _PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for process_batch(). "
                "Install with: pip install pandas"
            )

        batch_start = time.perf_counter()

        # ── 1. Normalise inputs ─────────────────────────────────────────────
        raw_texts = self._extract_texts(inputs, text_column)
        total_n = len(raw_texts)
        logger.info(f"Batch of {total_n} items received.")

        # Separate valid texts from items that should be skipped immediately
        valid_items: List[tuple] = []   # (original_index, text)
        rows: Dict[int, Dict[str, Any]] = {}  # original_index → flat row

        for i, raw in enumerate(raw_texts):
            if not isinstance(raw, str):
                reason = f"Expected str, got {type(raw).__name__}"
                logger.warning(f"Skipping item {i}: {reason}")
                rows[i] = _skipped_row(raw, reason)
            elif not raw.strip():
                reason = "Empty or whitespace-only string"
                logger.warning(f"Skipping item {i}: {reason}")
                rows[i] = _skipped_row(raw, reason)
            else:
                valid_items.append((i, raw))

        skipped_n = total_n - len(valid_items)
        logger.info(
            f"{len(valid_items)} texts to process, {skipped_n} skipped."
        )

        if not valid_items:
            logger.warning("No valid texts to process; returning empty results.")
            return self._build_dataframe(rows, total_n)

        # ── 2. Choose execution strategy ────────────────────────────────────
        workers = n_workers if n_workers is not None else _DEFAULT_WORKERS
        # Never spin up more workers than there are items to process
        workers = min(workers, len(valid_items))

        use_proc = (
            use_multiprocessing
            and workers > 1
            and not _in_interactive_session()
        )

        logger.info(
            f"Execution: {'ProcessPool' if use_proc else 'ThreadPool'} "
            f"× {workers} worker(s), chunk_size={chunk_size}"
        )

        # ── 3. Run in parallel ──────────────────────────────────────────────
        if use_proc:
            self._run_process_pool(valid_items, rows, workers, chunk_size, show_progress)
        else:
            self._run_thread_pool(valid_items, rows, workers, chunk_size, show_progress)

        # ── 4. Build & return DataFrame ─────────────────────────────────────
        df = self._build_dataframe(rows, total_n)

        elapsed = time.perf_counter() - batch_start
        ok_n = (df["status"].isin([_OK_STATUS, _WARN_STATUS])).sum()
        err_n = (df["status"] == _ERROR_STATUS).sum()
        logger.info(
            f"Batch complete in {elapsed:.2f}s — "
            f"{ok_n} ok, {err_n} errors, {skipped_n} skipped."
        )
        return df

    # ------------------------------------------------------------------
    # Batch internals
    # ------------------------------------------------------------------

    def _extract_texts(
        self, inputs: Any, text_column: str
    ) -> List[Any]:
        """Normalise *inputs* to a plain list of raw cell values.

        Args:
            inputs: Either a ``list[str]`` or a ``pandas.DataFrame``.
            text_column: Column name to read when *inputs* is a DataFrame.

        Returns:
            Plain Python list of values (not necessarily strings; type
            checking happens in the caller).

        Raises:
            ValueError: If *inputs* is a DataFrame and *text_column* is absent.
            TypeError: If *inputs* is neither a list nor a DataFrame.
        """
        if _PANDAS_AVAILABLE and isinstance(inputs, pd.DataFrame):
            if text_column not in inputs.columns:
                raise ValueError(
                    f"Column '{text_column}' not found in DataFrame. "
                    f"Available columns: {list(inputs.columns)}"
                )
            return inputs[text_column].tolist()
        if isinstance(inputs, list):
            return inputs
        raise TypeError(
            f"inputs must be a list[str] or pandas.DataFrame, "
            f"got {type(inputs).__name__}"
        )

    def _run_process_pool(
        self,
        items: List[tuple],
        rows: Dict[int, Dict[str, Any]],
        workers: int,
        chunk_size: int,
        show_progress: bool,
    ) -> None:
        """Dispatch *items* via :class:`~concurrent.futures.ProcessPoolExecutor`.

        Falls back to :meth:`_run_thread_pool` transparently if the process
        pool raises (e.g., in interactive sessions where ``spawn`` cannot
        import ``__main__``).

        Args:
            items: List of ``(original_index, text)`` pairs to process.
            rows: Mutable dict mapping original indices to flat result rows;
                updated in-place as futures complete.
            workers: Maximum number of worker processes.
            chunk_size: Number of futures to keep in-flight at once.
            show_progress: Whether to display a tqdm progress bar.
        """
        try:
            self._execute_parallel(
                items, rows, workers, chunk_size, show_progress,
                use_processes=True,
            )
        except Exception as exc:
            # Most common cause: interactive session / spawn restriction.
            # Fall back to threads transparently.
            logger.warning(
                f"ProcessPoolExecutor failed ({exc}); "
                "falling back to ThreadPoolExecutor."
            )
            self._execute_parallel(
                items, rows, workers, chunk_size, show_progress,
                use_processes=False,
            )

    def _run_thread_pool(
        self,
        items: List[tuple],
        rows: Dict[int, Dict[str, Any]],
        workers: int,
        chunk_size: int,
        show_progress: bool,
    ) -> None:
        """Dispatch *items* via :class:`~concurrent.futures.ThreadPoolExecutor`.

        Used in interactive sessions or when ``use_multiprocessing=False``.
        PyTorch releases the GIL during tensor operations, so threads still
        achieve meaningful parallelism for the inference-heavy steps.

        Args:
            items: List of ``(original_index, text)`` pairs to process.
            rows: Mutable dict updated in-place as futures complete.
            workers: Maximum number of worker threads.
            chunk_size: Sliding-window submission batch size.
            show_progress: Whether to display a tqdm progress bar.
        """
        self._execute_parallel(
            items, rows, workers, chunk_size, show_progress,
            use_processes=False,
        )

    def _execute_parallel(
        self,
        items: List[tuple],
        rows: Dict[int, Dict[str, Any]],
        workers: int,
        chunk_size: int,
        show_progress: bool,
        *,
        use_processes: bool,
    ) -> None:
        """Core parallel executor shared by both pool strategies.

        Uses a **sliding-window submission** pattern: instead of creating
        ``len(items)`` futures at once (which would exhaust memory for huge
        batches), only ``chunk_size`` futures are live at any moment.
        As each future completes, a new one is submitted from the pending
        queue.

        The :func:`as_completed` iterator is re-entered on each loop
        iteration so that newly submitted futures are visible immediately.

        Args:
            items: ``(original_index, text)`` pairs to process.
            rows: Mutable result dict; updated in-place.
            workers: Executor concurrency limit.
            chunk_size: Sliding-window size (futures in flight at once).
            show_progress: Whether to tick the progress bar.
            use_processes: ``True`` → ``ProcessPoolExecutor`` with
                :func:`_worker_initializer` / :func:`_worker_process_one`;
                ``False`` → ``ThreadPoolExecutor`` with
                :meth:`_process_one_in_thread`.
        """
        executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        executor_kwargs: Dict[str, Any] = {"max_workers": workers}
        if use_processes:
            executor_kwargs["initializer"] = _worker_initializer
            executor_kwargs["initargs"] = (self._init_kwargs,)

        with executor_cls(**executor_kwargs) as executor:
            # --- submit in chunks so we don't create O(N) futures up-front
            #     for very large batches while still getting fine-grained
            #     progress updates.
            future_to_orig_idx: Dict[Any, int] = {}
            pending: List[tuple] = list(items)

            # Submit first wave
            wave, pending = pending[:chunk_size], pending[chunk_size:]
            for item in wave:
                f = self._submit_one(executor, item, use_processes)
                future_to_orig_idx[f] = item[0]

            pbar = _make_pbar(len(items), show_progress)

            try:
                while future_to_orig_idx:
                    # Collect one completed future at a time
                    done_iter = as_completed(future_to_orig_idx, timeout=None)
                    future = next(done_iter)
                    orig_idx = future_to_orig_idx.pop(future)

                    # Record result
                    try:
                        idx, result, error = future.result()
                        if error:
                            text = items[
                                next(
                                    j for j, (ii, _) in enumerate(items)
                                    if ii == orig_idx
                                )
                            ][1]
                            rows[orig_idx] = _error_row(text, error)
                            logger.warning(f"Item {orig_idx} failed: {error}")
                        else:
                            rows[orig_idx] = _flatten(result)
                    except Exception as exc:
                        text = next(
                            t for ii, t in items if ii == orig_idx
                        )
                        rows[orig_idx] = _error_row(text, str(exc))
                        logger.error(
                            f"Future for item {orig_idx} raised: {exc}",
                            exc_info=True,
                        )

                    pbar.update(1)
                    pbar.set_postfix(
                        ok=sum(
                            1 for r in rows.values()
                            if r["status"] in (_OK_STATUS, _WARN_STATUS)
                        ),
                        err=sum(
                            1 for r in rows.values()
                            if r["status"] == _ERROR_STATUS
                        ),
                    )

                    # Refill the executor with the next chunk
                    if pending:
                        next_wave, pending = pending[:chunk_size], pending[chunk_size:]
                        for item in next_wave:
                            f = self._submit_one(executor, item, use_processes)
                            future_to_orig_idx[f] = item[0]

            finally:
                pbar.close()

    def _submit_one(self, executor: Any, item: tuple, use_processes: bool) -> Any:
        """Submit a single ``(original_index, text)`` item to *executor*.

        Chooses the correct worker function depending on the executor type:
        * Process pool → module-level :func:`_worker_process_one` (picklable).
        * Thread pool  → :meth:`_process_one_in_thread` (closure, not pickled).

        Args:
            executor: An active :class:`~concurrent.futures.Executor` instance.
            item: ``(original_index, text)`` tuple to submit.
            use_processes: Selects which worker function to use.

        Returns:
            A :class:`~concurrent.futures.Future` representing the pending result.
        """
        if use_processes:
            return executor.submit(_worker_process_one, item)
        # Thread pool: use a closure so self is captured without pickling.
        return executor.submit(self._process_one_in_thread, item)

    def _process_one_in_thread(self, indexed_text: tuple) -> tuple:
        """Thread-pool worker — runs in the main process, no pickling required.

        Calls :meth:`process` on *self* directly.  Unlike the process-pool
        path, this method can use the already-loaded ML models without
        re-initialisation overhead.

        Args:
            indexed_text: ``(original_index, text)`` pair.

        Returns:
            ``(original_index, result_dict, None)`` on success or
            ``(original_index, None, error_str)`` on failure.
        """
        idx, text = indexed_text
        try:
            return idx, self.process(text), None
        except Exception as exc:
            return idx, None, f"{type(exc).__name__}: {exc}"

    def _build_dataframe(
        self, rows: Dict[int, Dict[str, Any]], total_n: int
    ) -> Any:  # pd.DataFrame
        """Assemble the final DataFrame from the collected result rows.

        Rows are sorted by their original index so the output order matches
        the input order regardless of completion order.

        Args:
            rows: Dict mapping original input indices to flat result dicts.
            total_n: Total number of inputs (some may have been skipped
                before reaching the executor and will already be in *rows*).

        Returns:
            :class:`pandas.DataFrame` with columns :data:`_DF_COLUMNS`,
            one row per input, index reset to 0-based integers.
        """
        ordered = [rows[i] for i in range(total_n) if i in rows]
        df = pd.DataFrame(ordered, columns=_DF_COLUMNS)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Single-item internal helpers
    # ------------------------------------------------------------------

    def _detect_language(
        self, text: str, result: Dict[str, Any]
    ) -> DetectionResult:
        """Run language detection and populate *result* with detection fields.

        Times the detection step and records the elapsed time under
        ``result['timings']['language_detection']``.  On failure, appends
        an error message to ``result['errors']`` and returns a safe default
        :class:`DetectionResult` with ``language='unknown'`` so the pipeline
        can continue.

        Args:
            text: Raw input string.
            result: The in-progress result dict (mutated in-place).

        Returns:
            A :class:`~src.preprocessing.language_detector.DetectionResult`
            instance.  Never raises.
        """
        step = "language_detection"
        t0 = time.perf_counter()
        result["pipeline_steps"].append(step)

        try:
            detection = self.detector.detect(text)
            elapsed = time.perf_counter() - t0
            result["timings"][step] = round(elapsed, 4)
            result["detected_language"] = detection.language
            result["detection_confidence"] = round(detection.confidence, 4)

            logger.info(
                f"[{step}] language={detection.language!r}  "
                f"confidence={detection.confidence:.2%}  ({elapsed*1000:.1f}ms)"
            )
            logger.debug(f"[{step}] details: {detection.details}")
            logger.debug(f"[{step}] script_proportions: {detection.script_proportions}")
            return detection

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            result["timings"][step] = round(elapsed, 4)
            msg = f"Language detection failed: {exc}"
            result["errors"].append(msg)
            logger.error(f"[{step}] {msg}", exc_info=True)

            return DetectionResult(
                language="unknown",
                confidence=0.0,
                script_proportions={},
                details="Detection failed due to exception.",
            )

    def _route(
        self,
        text: str,
        detection: DetectionResult,
        result: Dict[str, Any],
    ) -> Optional[str]:
        """Select and execute the preprocessing chain for *text*.

        Dispatches to the correct sequence of transliteration / translation
        stages based on ``detection.language`` and returns the English-ready
        text to feed into the sentiment classifier.

        Args:
            text: Original input text.
            detection: Output of :meth:`_detect_language`.
            result: In-progress result dict (mutated in-place via the
                helper methods it delegates to).

        Returns:
            English text ready for sentiment classification, or ``None`` if
            a fatal error prevents any output.
        """
        lang = detection.language

        if lang == "kannada_script":
            logger.info("Route: Kannada script → translate → sentiment")
            return self._translate(text, result)

        if lang == "romanized_kannada":
            logger.info("Route: Romanized Kannada → transliterate → translate → sentiment")
            kannada_text = self._transliterate(text, result)
            return self._translate(kannada_text, result)

        if lang == "english":
            logger.info("Route: English → sentiment directly")
            return text

        # mixed / unknown — best-effort
        logger.info(f"Route: {lang!r} text → sentiment directly (best-effort)")
        result["errors"].append(
            f"Language is '{lang}'; attempting direct sentiment classification."
        )
        return text

    def _transliterate(self, text: str, result: Dict[str, Any]) -> str:
        """Transliterate Romanized Kannada to native Kannada script.

        Times the transliteration step and records the result in
        ``result['transliterated_text']``.  On failure, logs an error,
        appends to ``result['errors']``, and returns the original *text*
        unchanged so translation can still be attempted.

        Args:
            text: Romanized Kannada text (Latin script).
            result: In-progress result dict (mutated in-place).

        Returns:
            Native Kannada script text on success; *text* unchanged on failure.
        """
        step = "transliteration"
        t0 = time.perf_counter()
        result["pipeline_steps"].append(step)

        try:
            tlit = self.transliterator.transliterate(text)
            elapsed = time.perf_counter() - t0
            result["timings"][step] = round(elapsed, 4)
            result["transliterated_text"] = tlit.transliterated

            logger.info(
                f"[{step}] method={tlit.method}  "
                f"output={tlit.transliterated!r}  ({elapsed*1000:.1f}ms)"
            )
            return tlit.transliterated

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            result["timings"][step] = round(elapsed, 4)
            msg = f"Transliteration failed: {exc}"
            result["errors"].append(msg)
            logger.error(f"[{step}] {msg}", exc_info=True)
            logger.warning(f"[{step}] Continuing with original romanized text.")
            result["transliterated_text"] = text
            return text

    def _translate(self, text: str, result: Dict[str, Any]) -> str:
        """Translate Kannada (native script) to English.

        Times the translation step and records the output in
        ``result['translated_text']``.  On failure, logs an error, appends
        to ``result['errors']``, and returns *text* unchanged so sentiment
        classification can still be attempted on the Kannada input.

        Args:
            text: Kannada Unicode text to translate.
            result: In-progress result dict (mutated in-place).

        Returns:
            English translation on success; *text* unchanged on failure.
        """
        step = "translation"
        t0 = time.perf_counter()
        result["pipeline_steps"].append(step)

        try:
            tran = self.translator.translate(text)
            elapsed = time.perf_counter() - t0
            result["timings"][step] = round(elapsed, 4)
            result["translated_text"] = tran.translated

            if tran.success:
                logger.info(
                    f"[{step}] backend={tran.backend.value}  "
                    f"output={tran.translated!r}  ({elapsed*1000:.1f}ms)"
                )
            else:
                msg = f"Translation finished with error: {tran.error}"
                result["errors"].append(msg)
                logger.warning(f"[{step}] {msg}")

            return tran.translated

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            result["timings"][step] = round(elapsed, 4)
            msg = f"Translation failed: {exc}"
            result["errors"].append(msg)
            logger.error(f"[{step}] {msg}", exc_info=True)
            logger.warning(f"[{step}] Continuing with pre-translation text.")
            return text

    def _classify_sentiment(self, text: str, result: Dict[str, Any]) -> None:
        """Run sentiment classification and populate *result* with the output.

        Times the classification step and records ``sentiment_label`` and
        ``confidence_score`` in *result*.  On failure, logs the error,
        appends to ``result['errors']``, and leaves the sentinel defaults
        (``'Neutral'``, ``0.0``) in place.

        Args:
            text: English text to classify (output of :meth:`_translate` or
                the original text for English / mixed inputs).
            result: In-progress result dict (mutated in-place).

        Returns:
            ``None``.  All output is written directly into *result*.
        """
        step = "sentiment_classification"
        t0 = time.perf_counter()
        result["pipeline_steps"].append(step)

        try:
            sentiment: SentimentResult = self.classifier.classify(text)
            elapsed = time.perf_counter() - t0
            result["timings"][step] = round(elapsed, 4)
            result["sentiment_label"] = sentiment.label.value
            result["confidence_score"] = round(sentiment.confidence, 4)

            logger.info(
                f"[{step}] label={sentiment.label.value}  "
                f"confidence={sentiment.confidence:.2%}  ({elapsed*1000:.1f}ms)"
            )
            logger.debug(f"[{step}] probabilities: {sentiment.probabilities}")

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            result["timings"][step] = round(elapsed, 4)
            msg = f"Sentiment classification failed: {exc}"
            result["errors"].append(msg)
            logger.error(f"[{step}] {msg}", exc_info=True)


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _ms(t0: float) -> str:
    """Format the elapsed time since *t0* as a human-readable string.

    Args:
        t0: Start timestamp from :func:`time.perf_counter`.

    Returns:
        String like ``'12.3ms'`` showing elapsed milliseconds.

    Example:
        >>> import time
        >>> t = time.perf_counter()
        >>> _ms(t)   # near-instant call
        '0.0ms'
    """
    return f"{(time.perf_counter() - t0) * 1000:.1f}ms"


# ---------------------------------------------------------------------------
# Factory: build a pipeline from config.yaml
# ---------------------------------------------------------------------------

def load_pipeline_from_config() -> KannadaSentimentPipeline:
    """
    Instantiate a :class:`KannadaSentimentPipeline` using settings from
    ``config.yaml`` at the project root.

    Falls back to all-default construction if the file is missing or
    cannot be parsed.

    Returns
    -------
    KannadaSentimentPipeline

    Example:
        >>> pipeline = load_pipeline_from_config()
        >>> result = pipeline.process("Hello world")
        >>> result["sentiment_label"] in ("Positive", "Negative", "Neutral")
        True
    """
    import os

    try:
        import yaml

        config_path = Path(__file__).parent.parent / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml not found at {config_path}")

        with open(config_path, "r") as fh:
            config = yaml.safe_load(fh)

        translation_cfg = config.get("models", {}).get("translation", {})
        sentiment_cfg = config.get("models", {}).get("sentiment", {})

        backend = translation_cfg.get("backend", "indictrans2")
        api_key = (
            os.environ.get("GOOGLE_TRANSLATE_API_KEY")
            if backend == "google"
            else None
        )
        auto_fallback = translation_cfg.get("auto_fallback", True)

        # Prefer a locally cached classifier model when available
        model_path: Optional[str] = None
        local_model = (
            Path(__file__).parent.parent / "data" / "models" / "distilbert-sst2"
        )
        if local_model.exists():
            model_path = str(local_model)
            logger.info(f"Using local classifier model: {model_path}")

        return KannadaSentimentPipeline(
            translation_backend=backend,
            classifier_model_name=sentiment_cfg.get("name"),
            classifier_model_path=model_path,
            google_api_key=api_key,
            auto_fallback=auto_fallback,
        )

    except Exception as exc:
        logger.warning(
            f"Could not load pipeline config ({exc}). Using defaults."
        )
        return KannadaSentimentPipeline()
