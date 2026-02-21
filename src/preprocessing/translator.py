"""
Translation Module for Kannada Sentiment Analysis.

Translates Kannada text to English using one of three interchangeable
backends, selected at construction time with automatic fallback:

Backends
--------
``indictrans2``
    AI4Bharat's IndicTrans2 (``ai4bharat/indictrans2-indic-en-1B``).
    Offline, high quality, ~3 GB download on first use.  Requires
    ``transformers`` and sufficient RAM (~4 GB).

``google``
    Google Cloud Translate REST API.  Requires an API key stored in the
    ``GOOGLE_TRANSLATE_API_KEY`` environment variable (or passed directly).
    Needs internet access; billed per character.

``fallback``
    Built-in word-by-word dictionary (~40 entries).  Always available,
    no network or heavy models required.  Accuracy is limited to words
    present in the dictionary.

Auto-fallback chain
-------------------
When ``auto_fallback=True`` (default) and the preferred backend fails, the
:class:`Translator` tries the next available backend:

    indictrans2 → google → fallback

Long-text segmentation
----------------------
Texts longer than ``max_segment_length`` (default 500 chars) are split at
sentence boundaries first, then at word boundaries.  Each segment is
translated independently and the results are rejoined with a single space.

Public symbols
--------------
* :class:`TranslationBackend`  — enum of the three backend identifiers.
* :class:`TranslationResult`   — dataclass returned by :meth:`Translator.translate`.
* :class:`TranslationError`    — raised when a backend call fails.
* :class:`BaseTranslator`      — abstract base class for backend implementations.
* :class:`IndicTrans2Translator`, :class:`GoogleTranslator`,
  :class:`FallbackTranslator`  — concrete backends.
* :class:`Translator`          — unified interface managing all backends.
* :func:`translate_kannada_to_english` — module-level convenience wrapper.
* :func:`load_config_translator`       — factory using ``config.yaml``.

Example
-------
>>> from src.preprocessing.translator import Translator
>>> tr = Translator(backend="fallback")
>>> result = tr.translate("ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ")
>>> result.translated
'very is good'
>>> result.success
True
>>> result.backend.value
'fallback'
"""

import re
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)


class TranslationBackend(Enum):
    """Enum for available translation backends."""
    INDICTRANS2 = "indictrans2"
    GOOGLE = "google"
    FALLBACK = "fallback"


@dataclass
class TranslationResult:
    """
    Data class to hold translation results.

    Attributes:
        original: The original input text.
        translated: The translated output text.
        source_lang: Detected or specified source language.
        target_lang: Target language for translation.
        backend: The backend used for translation.
        segments: List of (original_segment, translated_segment) tuples.
        success: Whether translation was successful.
        error: Error message if translation failed.
    """
    original: str
    translated: str
    source_lang: str
    target_lang: str
    backend: TranslationBackend
    segments: List[Tuple[str, str]]
    success: bool
    error: Optional[str] = None


class TranslationError(Exception):
    """Custom exception for translation failures."""
    pass


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator for retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds between retries.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed.")
            raise last_exception
        return wrapper
    return decorator


class BaseTranslator(ABC):
    """Abstract base class defining the interface for translation backends.

    All concrete translation backends must subclass this and implement
    :meth:`translate` and :meth:`is_available`.  The :class:`Translator`
    facade delegates to whichever concrete backend is currently active.

    Subclasses
    ----------
    * :class:`IndicTrans2Translator`
    * :class:`GoogleTranslator`
    * :class:`FallbackTranslator`
    """

    @abstractmethod
    def translate(self, text: str) -> str:
        """Translate *text* from Kannada to English.

        Args:
            text: Kannada-language text to translate.

        Returns:
            English translation of *text*.

        Raises:
            TranslationError: If the backend call fails.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if this backend is ready to accept translation requests.

        Called by :class:`Translator` to determine whether the backend was
        successfully initialized and can handle requests right now.

        Returns:
            ``True`` when the backend is operational, ``False`` otherwise.
        """
        pass


class IndicTrans2Translator(BaseTranslator):
    """
    Translator using AI4Bharat's IndicTrans2 model.

    This is an offline translator that doesn't require an API key.
    Note: Requires the model to be downloaded and may have compatibility
    issues with Python 3.11+.
    """

    def __init__(self):
        """Initialize the IndicTrans2 translator."""
        self._model = None
        self._tokenizer = None
        self._available = False
        self._initialize()

    def _initialize(self):
        """Download and load the IndicTrans2 model and tokenizer.

        Attempts to load ``ai4bharat/indictrans2-indic-en-1B`` from the
        HuggingFace Hub (downloaded to the local cache on first call).
        Sets ``self._available = True`` on success; logs a warning and
        sets ``False`` on any failure so :class:`Translator` can fall back
        gracefully.
        """
        try:
            # Try to import and initialize the model
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            model_name = "ai4bharat/indictrans2-indic-en-1B"

            logger.info(f"Loading IndicTrans2 model: {model_name}")
            logger.info("This may take a while on first run...")

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            self._available = True
            logger.info("IndicTrans2 model loaded successfully.")

        except ImportError as e:
            logger.warning(f"transformers package not available: {e}")
            self._available = False

        except Exception as e:
            logger.warning(f"Failed to load IndicTrans2 model: {e}")
            self._available = False

    def is_available(self) -> bool:
        """Check if IndicTrans2 is available."""
        return self._available

    @retry_with_backoff(max_retries=2, base_delay=0.5)
    def translate(self, text: str) -> str:
        """
        Translate Kannada text to English using IndicTrans2.

        Args:
            text: Kannada text to translate.

        Returns:
            English translation.

        Raises:
            TranslationError: If translation fails.
        """
        if not self._available:
            raise TranslationError("IndicTrans2 model not available.")

        try:
            # Prepare input with language tags
            # IndicTrans2 uses specific format: <2en> for target language
            input_text = f"<2en> {text}"

            inputs = self._tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            outputs = self._model.generate(
                **inputs,
                max_length=512,
                num_beams=5,
                early_stopping=True
            )

            translated = self._tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            return translated.strip()

        except Exception as e:
            raise TranslationError(f"IndicTrans2 translation failed: {e}")


class GoogleTranslator(BaseTranslator):
    """
    Translator using Google Translate API.

    This requires a Google Cloud API key and internet connection.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Google Translator.

        Args:
            api_key: Google Cloud API key. If None, attempts to use
                environment variable GOOGLE_TRANSLATE_API_KEY.
        """
        import os
        self._api_key = api_key or os.environ.get("GOOGLE_TRANSLATE_API_KEY")
        self._available = False
        self._client = None
        self._initialize()

    def _initialize(self):
        """Validate the API key and initialize the Google Translate client.

        Tries the ``google-cloud-translate`` library first; if unavailable,
        falls back to making raw REST calls via ``requests``.  Sets
        ``self._available = False`` when no API key is configured.
        """
        if not self._api_key:
            logger.warning(
                "Google Translate API key not provided. "
                "Set GOOGLE_TRANSLATE_API_KEY environment variable or pass api_key."
            )
            self._available = False
            return

        try:
            # Try using google-cloud-translate library
            from google.cloud import translate_v2 as translate

            self._client = translate.Client()
            self._available = True
            logger.info("Google Translate client initialized.")

        except ImportError:
            # Fall back to requests-based approach
            try:
                import requests
                self._available = True
                self._client = None  # Will use requests
                logger.info("Google Translate initialized (REST API mode).")
            except ImportError:
                logger.warning("Neither google-cloud-translate nor requests available.")
                self._available = False

        except Exception as e:
            logger.warning(f"Failed to initialize Google Translate: {e}")
            self._available = False

    def is_available(self) -> bool:
        """Check if Google Translate is available."""
        return self._available and self._api_key is not None

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def translate(self, text: str) -> str:
        """
        Translate Kannada text to English using Google Translate.

        Args:
            text: Kannada text to translate.

        Returns:
            English translation.

        Raises:
            TranslationError: If translation fails.
        """
        if not self.is_available():
            raise TranslationError("Google Translate not available.")

        try:
            if self._client is not None:
                # Use google-cloud-translate library
                result = self._client.translate(
                    text,
                    source_language="kn",
                    target_language="en"
                )
                return result["translatedText"]
            else:
                # Use REST API directly
                return self._translate_with_rest_api(text)

        except Exception as e:
            raise TranslationError(f"Google Translate failed: {e}")

    def _translate_with_rest_api(self, text: str) -> str:
        """Translate *text* using the Google Cloud Translation REST API v2.

        This path is used when the ``google-cloud-translate`` library is not
        installed.  It calls the JSON/REST endpoint directly with the
        configured API key.

        Args:
            text: The Kannada text to translate.

        Returns:
            The English translation from the API response.

        Raises:
            TranslationError: If the HTTP request fails or the response
                format is unexpected.
        """
        import requests

        url = "https://translation.googleapis.com/language/translate/v2"

        params = {
            "key": self._api_key,
            "q": text,
            "source": "kn",
            "target": "en",
            "format": "text"
        }

        response = requests.post(url, data=params, timeout=30)
        response.raise_for_status()

        result = response.json()

        if "data" in result and "translations" in result["data"]:
            return result["data"]["translations"][0]["translatedText"]
        else:
            raise TranslationError("Unexpected API response format.")


class FallbackTranslator(BaseTranslator):
    """Dictionary-based Kannada→English translator.

    This backend performs a simple word-by-word lookup against a built-in
    dictionary of ~40 common Kannada words.  Words not found in the
    dictionary are returned unchanged.

    Unlike the neural backends, this translator:
    * Requires no external dependencies or network access.
    * Is always available (``is_available()`` always returns ``True``).
    * Produces partial translations — unknown words remain in Kannada.

    It is used as the final fallback in the auto-fallback chain.
    """

    # Basic Kannada to English dictionary for common words
    BASIC_DICTIONARY: Dict[str, str] = {
        # Sentiment words
        "ಚೆನ್ನಾಗಿದೆ": "is good",
        "ಚೆನ್ನಾಗಿ": "good",
        "ತುಂಬ": "very",
        "ತುಂಬಾ": "very",
        "ಒಳ್ಳೆ": "good",
        "ಒಳ್ಳೆಯ": "good",
        "ಕೆಟ್ಟ": "bad",
        "ಕೆಟ್ಟದು": "is bad",
        "ಸುಂದರ": "beautiful",
        "ಸಂತೋಷ": "happy",
        "ದುಃಖ": "sad",
        "ಕೋಪ": "angry",

        # Common verbs
        "ಇದೆ": "is",
        "ಇಲ್ಲ": "is not",
        "ಆಗಿದೆ": "has become",
        "ಬೇಕು": "want",
        "ಬೇಕಿಲ್ಲ": "don't want",
        "ಬೇಡ": "don't want",
        "ಗೊತ್ತು": "know",
        "ಗೊತ್ತಿಲ್ಲ": "don't know",

        # Pronouns
        "ನಾನು": "I",
        "ನೀನು": "you",
        "ನೀವು": "you",
        "ಅವನು": "he",
        "ಅವಳು": "she",
        "ಅವರು": "they",
        "ಇದು": "this",
        "ಅದು": "that",

        # Question words
        "ಯೇನು": "what",
        "ಯಾರು": "who",
        "ಎಲ್ಲಿ": "where",
        "ಯಾವಾಗ": "when",
        "ಹೇಗೆ": "how",
        "ಯಾಕೆ": "why",

        # Common nouns
        "ಮನೆ": "house",
        "ಕೆಲಸ": "work",
        "ಹಣ": "money",
        "ಸಮಯ": "time",
        "ದಿನ": "day",

        # Connectors
        "ಮತ್ತು": "and",
        "ಆದರೆ": "but",
        "ಅಥವಾ": "or",

        # Product review words
        "ಪ್ರಾಡಕ್ಟ್": "product",
        "ಕ್ವಾಲಿಟಿ": "quality",
        "ಬೆಲೆ": "price",
        "ಡೆಲಿವರಿ": "delivery",
        "ಸರ್ವಿಸ್": "service",
    }

    def __init__(self):
        """Initialize the fallback translator."""
        self._dictionary = dict(self.BASIC_DICTIONARY)

    def is_available(self) -> bool:
        """Fallback translator is always available."""
        return True

    def translate(self, text: str) -> str:
        """
        Translate using basic dictionary lookup.

        Args:
            text: Kannada text to translate.

        Returns:
            Partially translated text (words not in dictionary remain as-is).
        """
        words = text.split()
        translated_words = []

        for word in words:
            # Remove punctuation for lookup
            clean_word = word.strip(".,!?;:")
            punctuation = word[len(clean_word):] if len(clean_word) < len(word) else ""

            if clean_word in self._dictionary:
                translated_words.append(self._dictionary[clean_word] + punctuation)
            else:
                # Keep original if not in dictionary
                translated_words.append(word)

        return " ".join(translated_words)

    def add_translation(self, kannada: str, english: str) -> None:
        """Add a single Kannada→English entry to the runtime dictionary.

        Args:
            kannada: The Kannada word or phrase to add.
            english: The corresponding English translation.

        Example:
            >>> ft = FallbackTranslator()
            >>> ft.add_translation("ಹೊಸ", "new")
            >>> ft.translate("ಹೊಸ product")
            'new product'
        """
        self._dictionary[kannada] = english


class Translator:
    """
    Main Translator class that manages multiple translation backends.

    This class provides a unified interface for translating Kannada text
    to English using either AI4Bharat's IndicTrans2 model or Google Translate API.

    Attributes:
        backend: The translation backend to use.
        max_segment_length: Maximum characters per segment for long texts.

    Example:
        >>> translator = Translator(backend=TranslationBackend.INDICTRANS2)
        >>> result = translator.translate("ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ")
        >>> print(result.translated)
        'This product is very good'
    """

    # Kannada sentence ending patterns
    SENTENCE_ENDINGS = re.compile(r'[.!?।॥]\s*')

    def __init__(
        self,
        backend: Union[TranslationBackend, str] = TranslationBackend.INDICTRANS2,
        google_api_key: Optional[str] = None,
        max_segment_length: int = 500,
        auto_fallback: bool = True
    ):
        """
        Initialize the Translator.

        Args:
            backend: The translation backend to use.
                Can be TranslationBackend enum or string ('indictrans2', 'google', 'fallback').
            google_api_key: API key for Google Translate (required if using Google backend).
            max_segment_length: Maximum characters per segment when splitting long texts.
            auto_fallback: If True, automatically fall back to simpler backends on failure.
        """
        # Handle string backend specification
        if isinstance(backend, str):
            backend = TranslationBackend(backend.lower())

        self.backend = backend
        self.max_segment_length = max_segment_length
        self.auto_fallback = auto_fallback

        # Initialize translators
        self._translators: Dict[TranslationBackend, BaseTranslator] = {}
        self._initialize_translators(google_api_key)

        # Determine active backend
        self._active_backend = self._select_backend()
        logger.info(f"Active translation backend: {self._active_backend.value}")

    def _initialize_translators(self, google_api_key: Optional[str]) -> None:
        """Instantiate all backend objects that may be needed.

        The fallback translator is always created.  IndicTrans2 and Google
        backends are only created when they are the preferred choice or when
        ``auto_fallback=True`` is set, to avoid unnecessary model downloads.

        Args:
            google_api_key: API key to pass to :class:`GoogleTranslator`.
                ``None`` causes that backend to self-report as unavailable.
        """
        # Always initialize fallback
        self._translators[TranslationBackend.FALLBACK] = FallbackTranslator()

        # Initialize IndicTrans2 if requested
        if self.backend == TranslationBackend.INDICTRANS2 or self.auto_fallback:
            try:
                self._translators[TranslationBackend.INDICTRANS2] = IndicTrans2Translator()
            except Exception as e:
                logger.warning(f"Failed to initialize IndicTrans2: {e}")

        # Initialize Google Translate if requested
        if self.backend == TranslationBackend.GOOGLE or self.auto_fallback:
            try:
                self._translators[TranslationBackend.GOOGLE] = GoogleTranslator(
                    api_key=google_api_key
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Google Translate: {e}")

    def _select_backend(self) -> TranslationBackend:
        """Select the best currently-available backend.

        Probes the preferred backend first; if it reports ``is_available()
        == False``, and ``auto_fallback`` is enabled, tries IndicTrans2 then
        Google then the dictionary fallback.

        Returns:
            The :class:`TranslationBackend` enum value for the backend that
            will be used for subsequent :meth:`translate` calls.
        """
        # Try preferred backend first
        if self.backend in self._translators:
            translator = self._translators[self.backend]
            if translator.is_available():
                return self.backend

        # Auto fallback to other backends
        if self.auto_fallback:
            # Try IndicTrans2
            if TranslationBackend.INDICTRANS2 in self._translators:
                if self._translators[TranslationBackend.INDICTRANS2].is_available():
                    logger.info("Falling back to IndicTrans2")
                    return TranslationBackend.INDICTRANS2

            # Try Google
            if TranslationBackend.GOOGLE in self._translators:
                if self._translators[TranslationBackend.GOOGLE].is_available():
                    logger.info("Falling back to Google Translate")
                    return TranslationBackend.GOOGLE

        # Final fallback to dictionary-based translator
        logger.warning("Using fallback dictionary translator")
        return TranslationBackend.FALLBACK

    def _segment_text(self, text: str) -> List[str]:
        """
        Split long text into manageable segments.

        Attempts to split at sentence boundaries first, then falls back
        to splitting at word boundaries if sentences are too long.

        Args:
            text: The text to segment.

        Returns:
            List of text segments.
        """
        if len(text) <= self.max_segment_length:
            return [text]

        # Split by sentence endings
        sentences = self.SENTENCE_ENDINGS.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        segments = []
        current_segment = ""

        for sentence in sentences:
            # If single sentence is too long, split by words
            if len(sentence) > self.max_segment_length:
                if current_segment:
                    segments.append(current_segment.strip())
                    current_segment = ""

                # Split long sentence by words
                words = sentence.split()
                word_segment = ""
                for word in words:
                    if len(word_segment) + len(word) + 1 > self.max_segment_length:
                        if word_segment:
                            segments.append(word_segment.strip())
                        word_segment = word
                    else:
                        word_segment = f"{word_segment} {word}".strip()

                if word_segment:
                    segments.append(word_segment.strip())

            elif len(current_segment) + len(sentence) + 1 > self.max_segment_length:
                # Start new segment
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = sentence
            else:
                # Add to current segment
                current_segment = f"{current_segment} {sentence}".strip()

        # Add remaining segment
        if current_segment:
            segments.append(current_segment.strip())

        return segments

    def translate(self, text: str) -> TranslationResult:
        """
        Translate Kannada text to English.

        Handles long texts by segmenting them and translating each segment.
        Includes retry logic and automatic fallback on failure.

        Args:
            text: The Kannada text to translate.

        Returns:
            A TranslationResult object containing the translation and metadata.

        Example:
            >>> translator = Translator()
            >>> result = translator.translate("ನಮಸ್ಕಾರ")
            >>> print(result.translated)
            'Hello'
        """
        if not text or not text.strip():
            return TranslationResult(
                original=text,
                translated=text,
                source_lang="kn",
                target_lang="en",
                backend=self._active_backend,
                segments=[],
                success=True
            )

        text = text.strip()

        # Segment long text
        segments = self._segment_text(text)
        translated_segments = []
        segment_pairs = []
        used_backend = self._active_backend

        try:
            translator = self._translators[self._active_backend]

            for segment in segments:
                try:
                    translated = translator.translate(segment)
                    translated_segments.append(translated)
                    segment_pairs.append((segment, translated))

                except TranslationError as e:
                    logger.warning(f"Segment translation failed: {e}")

                    # Try fallback if enabled
                    if self.auto_fallback and used_backend != TranslationBackend.FALLBACK:
                        fallback = self._translators[TranslationBackend.FALLBACK]
                        translated = fallback.translate(segment)
                        translated_segments.append(translated)
                        segment_pairs.append((segment, translated))
                        used_backend = TranslationBackend.FALLBACK
                    else:
                        raise

            # Join translated segments
            full_translation = " ".join(translated_segments)

            return TranslationResult(
                original=text,
                translated=full_translation,
                source_lang="kn",
                target_lang="en",
                backend=used_backend,
                segments=segment_pairs,
                success=True
            )

        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            logger.error(error_msg)

            return TranslationResult(
                original=text,
                translated=text,  # Return original on failure
                source_lang="kn",
                target_lang="en",
                backend=used_backend,
                segments=[],
                success=False,
                error=error_msg
            )

    def translate_simple(self, text: str) -> str:
        """
        Simple translation that returns only the translated text.

        Args:
            text: The Kannada text to translate.

        Returns:
            The English translation (or original text on failure).
        """
        return self.translate(text).translated

    def translate_batch(self, texts: List[str]) -> List[TranslationResult]:
        """
        Translate multiple texts.

        Args:
            texts: List of Kannada texts to translate.

        Returns:
            List of TranslationResult objects.
        """
        return [self.translate(text) for text in texts]

    def get_status(self) -> Dict:
        """
        Get the current status of the translator.

        Returns:
            Dictionary with status information.
        """
        status = {
            "preferred_backend": self.backend.value,
            "active_backend": self._active_backend.value,
            "max_segment_length": self.max_segment_length,
            "auto_fallback": self.auto_fallback,
            "backends": {}
        }

        for backend, translator in self._translators.items():
            status["backends"][backend.value] = {
                "available": translator.is_available()
            }

        return status

    @property
    def active_backend(self) -> TranslationBackend:
        """Get the currently active backend."""
        return self._active_backend


# Module-level convenience function
def translate_kannada_to_english(
    text: str,
    backend: str = "indictrans2"
) -> str:
    """
    Convenience function to translate Kannada to English.

    Args:
        text: The Kannada text to translate.
        backend: Translation backend ('indictrans2', 'google', 'fallback').

    Returns:
        The English translation.

    Example:
        >>> from src.preprocessing.translator import translate_kannada_to_english
        >>> translate_kannada_to_english("ಚೆನ್ನಾಗಿದೆ")
        'is good'
    """
    translator = Translator(backend=backend)
    return translator.translate_simple(text)


def load_config_translator() -> Translator:
    """
    Load translator with settings from config.yaml.

    Returns:
        Configured Translator instance.

    Example:
        >>> translator = load_config_translator()
        >>> result = translator.translate("ಚೆನ್ನಾಗಿದೆ")
        >>> result.success
        True
    """
    import os
    from pathlib import Path

    try:
        import yaml

        # Find config file
        config_path = Path(__file__).parent.parent.parent / "config.yaml"

        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Get translation settings
            translation_config = config.get("models", {}).get("translation", {})
            backend_name = translation_config.get("backend", "indictrans2")

            # Get API key from environment if using Google
            api_key = None
            if backend_name == "google":
                api_key = os.environ.get("GOOGLE_TRANSLATE_API_KEY")

            return Translator(
                backend=backend_name,
                google_api_key=api_key
            )

    except Exception as e:
        logger.warning(f"Failed to load config: {e}. Using defaults.")

    return Translator()
