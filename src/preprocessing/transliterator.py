"""
Transliteration Module for Kannada Sentiment Analysis.

This module provides functionality to transliterate Romanized Kannada
text to native Kannada script using AI4Bharat's IndicXlit model with
fallback to a phonetic mapping dictionary.
"""

import re
import logging
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TransliterationResult:
    """
    Data class to hold transliteration results.

    Attributes:
        original: The original input text.
        transliterated: The transliterated output text.
        word_mappings: List of (original_word, transliterated_word) tuples.
        method: The method used ('model', 'fallback', or 'mixed').
        success: Whether transliteration was successful.
    """
    original: str
    transliterated: str
    word_mappings: List[Tuple[str, str]]
    method: str
    success: bool


class TransliterationError(Exception):
    """Custom exception for transliteration failures."""
    pass


class Transliterator:
    """
    Transliterates Romanized Kannada text to native Kannada script.

    This class uses AI4Bharat's IndicXlit model as the primary transliteration
    engine, with a fallback phonetic mapping dictionary for common words and
    cases where the model is unavailable.

    Attributes:
        use_model: Whether to attempt using the IndicXlit model.
        fallback_dict: Dictionary of common Romanized Kannada to Kannada script mappings.

    Example:
        >>> transliterator = Transliterator()
        >>> result = transliterator.transliterate("namaskara hegidira")
        >>> print(result.transliterated)
        'ನಮಸ್ಕಾರ ಹೇಗಿದೀರ'

        >>> # Word-by-word transliteration
        >>> words = transliterator.transliterate_words(["tumba", "chennagide"])
        >>> print(words)
        ['ತುಂಬ', 'ಚೆನ್ನಾಗಿದೆ']
    """

    # Comprehensive fallback dictionary for common Romanized Kannada words
    DEFAULT_FALLBACK_DICT: Dict[str, str] = {
        # Greetings and common phrases
        "namaskara": "ನಮಸ್ಕಾರ",
        "namaste": "ನಮಸ್ತೆ",
        "dhanyavada": "ಧನ್ಯವಾದ",
        "dhanyavadagalu": "ಧನ್ಯವಾದಗಳು",
        "shubhodaya": "ಶುಭೋದಯ",
        "shubharatri": "ಶುಭರಾತ್ರಿ",

        # Sentiment words - Positive
        "chennagide": "ಚೆನ್ನಾಗಿದೆ",
        "chennagi": "ಚೆನ್ನಾಗಿ",
        "tumba": "ತುಂಬ",
        "tumbaa": "ತುಂಬಾ",
        "olle": "ಒಳ್ಳೆ",
        "olleya": "ಒಳ್ಳೆಯ",
        "olleyadu": "ಒಳ್ಳೆಯದು",
        "sundar": "ಸುಂದರ",
        "sundara": "ಸುಂದರ",
        "khushi": "ಖುಷಿ",
        "santhosha": "ಸಂತೋಷ",
        "santhoshavagide": "ಸಂತೋಷವಾಗಿದೆ",
        "adbhuta": "ಅದ್ಭುತ",
        "uttama": "ಉತ್ತಮ",
        "meccuge": "ಮೆಚ್ಚುಗೆ",
        "ishta": "ಇಷ್ಟ",
        "ishtavagide": "ಇಷ್ಟವಾಗಿದೆ",
        "saku": "ಸಾಕು",
        "super": "ಸೂಪರ್",

        # Sentiment words - Negative
        "ketta": "ಕೆಟ್ಟ",
        "kettadu": "ಕೆಟ್ಟದು",
        "bekilla": "ಬೇಕಿಲ್ಲ",
        "beda": "ಬೇಡ",
        "illa": "ಇಲ್ಲ",
        "illaandu": "ಇಲ್ಲಾಂದು",
        "kashtavagide": "ಕಷ್ಟವಾಗಿದೆ",
        "kashta": "ಕಷ್ಟ",
        "dukhavagide": "ದುಃಖವಾಗಿದೆ",
        "kopa": "ಕೋಪ",
        "sari": "ಸರಿ",
        "sariyilla": "ಸರಿಯಿಲ್ಲ",
        "waste": "ವೇಸ್ಟ್",
        "mosakavagide": "ಮೋಸಕವಾಗಿದೆ",

        # Common verbs and auxiliaries
        "ide": "ಇದೆ",
        "ide.": "ಇದೆ.",
        "agide": "ಆಗಿದೆ",
        "aagide": "ಆಗಿದೆ",
        "aagitta": "ಆಗಿತ್ತ",
        "aagutta": "ಆಗುತ್ತ",
        "aagutte": "ಆಗುತ್ತೆ",
        "madide": "ಮಾಡಿದೆ",
        "madidini": "ಮಾಡಿದಿನಿ",
        "madtini": "ಮಾಡ್ತೀನಿ",
        "madtene": "ಮಾಡ್ತೇನೆ",
        "baralla": "ಬರಲ್ಲ",
        "baratte": "ಬರತ್ತೆ",
        "barutte": "ಬರುತ್ತೆ",
        "beku": "ಬೇಕು",
        "bekagide": "ಬೇಕಾಗಿದೆ",
        "gottu": "ಗೊತ್ತು",
        "gottilla": "ಗೊತ್ತಿಲ್ಲ",
        "gottide": "ಗೊತ್ತಿದೆ",

        # Pronouns
        "naanu": "ನಾನು",
        "nanu": "ನಾನು",
        "ninna": "ನಿನ್ನ",
        "neenu": "ನೀನು",
        "ninu": "ನೀನು",
        "neevu": "ನೀವು",
        "nivu": "ನೀವು",
        "avanu": "ಅವನು",
        "avalu": "ಅವಳು",
        "avaru": "ಅವರು",
        "ivanu": "ಇವನು",
        "ivalu": "ಇವಳು",
        "ivaru": "ಇವರು",
        "adu": "ಅದು",
        "idu": "ಇದು",
        "yenu": "ಯೇನು",
        "yenu?": "ಯೇನು?",
        "yaaru": "ಯಾರು",
        "yaaru?": "ಯಾರು?",
        "yavaga": "ಯಾವಾಗ",
        "yavaga?": "ಯಾವಾಗ?",
        "elli": "ಎಲ್ಲಿ",
        "elli?": "ಎಲ್ಲಿ?",
        "hege": "ಹೇಗೆ",
        "hege?": "ಹೇಗೆ?",
        "hegidira": "ಹೇಗಿದೀರ",
        "hegidira?": "ಹೇಗಿದೀರ?",
        "hegidiya": "ಹೇಗಿದ್ದೀಯ",
        "hegidiya?": "ಹೇಗಿದ್ದೀಯ?",

        # Question words
        "yaake": "ಯಾಕೆ",
        "yaake?": "ಯಾಕೆ?",
        "yeshtu": "ಎಷ್ಟು",
        "yeshtu?": "ಎಷ್ಟು?",
        "yaava": "ಯಾವ",

        # Common nouns
        "mane": "ಮನೆ",
        "oota": "ಊಟ",
        "neeru": "ನೀರು",
        "haalu": "ಹಾಲು",
        "anna": "ಅನ್ನ",
        "akki": "ಅಕ್ಕಿ",
        "tarkari": "ತರಕಾರಿ",
        "haNNu": "ಹಣ್ಣು",
        "hannu": "ಹಣ್ಣು",
        "product": "ಪ್ರಾಡಕ್ಟ್",
        "service": "ಸರ್ವಿಸ್",
        "quality": "ಕ್ವಾಲಿಟಿ",
        "price": "ಪ್ರೈಸ್",
        "bele": "ಬೆಲೆ",
        "review": "ರಿವ್ಯೂ",

        # Family relations
        "amma": "ಅಮ್ಮ",
        "appa": "ಅಪ್ಪ",
        "anna": "ಅಣ್ಣ",
        "akka": "ಅಕ್ಕ",
        "thambi": "ತಮ್ಮ",
        "tangi": "ತಂಗಿ",
        "maga": "ಮಗ",
        "magalu": "ಮಗಳು",
        "guru": "ಗುರು",
        "friend": "ಫ್ರೆಂಡ್",
        "snehita": "ಸ್ನೇಹಿತ",

        # Places
        "bengaluru": "ಬೆಂಗಳೂರು",
        "bangalore": "ಬೆಂಗಳೂರು",
        "mysuru": "ಮೈಸೂರು",
        "mysore": "ಮೈಸೂರು",
        "mangaluru": "ಮಂಗಳೂರು",
        "mangalore": "ಮಂಗಳೂರು",
        "karnataka": "ಕರ್ನಾಟಕ",
        "kannada": "ಕನ್ನಡ",

        # Time words
        "ivattu": "ಇವತ್ತು",
        "ivaaga": "ಈವಾಗ",
        "naale": "ನಾಳೆ",
        "nenne": "ನೆನ್ನೆ",
        "mundhe": "ಮುಂದೆ",
        "hinde": "ಹಿಂದೆ",

        # Connectors and particles
        "mattu": "ಮತ್ತು",
        "athava": "ಅಥವಾ",
        "aadare": "ಆದರೆ",
        "aadru": "ಆದ್ರೂ",
        "yaakaandre": "ಯಾಕಾಂದ್ರೆ",
        "andre": "ಅಂದ್ರೆ",
        "antaa": "ಅಂತಾ",
        "antha": "ಅಂಥ",
        "ashte": "ಅಷ್ಟೆ",
        "only": "ಓನ್ಲಿ",

        # Numbers (as words)
        "ondu": "ಒಂದು",
        "yeradu": "ಎರಡು",
        "eradu": "ಎರಡು",
        "mooru": "ಮೂರು",
        "nalku": "ನಾಲ್ಕು",
        "aidu": "ಐದು",

        # Adjectives
        "dodda": "ದೊಡ್ಡ",
        "chikka": "ಚಿಕ್ಕ",
        "hosa": "ಹೊಸ",
        "haladu": "ಹಳೆಯ",
        "bisi": "ಬಿಸಿ",
        "thanda": "ತಂಡ",

        # Review-specific words
        "recommend": "ರೆಕಮೆಂಡ್",
        "buy": "ಬೈ",
        "use": "ಯೂಸ್",
        "best": "ಬೆಸ್ಟ್",
        "worst": "ವರ್ಸ್ಟ್",
        "value": "ವ್ಯಾಲ್ಯೂ",
        "money": "ಮನಿ",
        "worth": "ವರ್ತ್",
        "fast": "ಫಾಸ್ಟ್",
        "slow": "ಸ್ಲೋ",
        "delivery": "ಡೆಲಿವರಿ",
        "ok": "ಓಕೆ",
        "okay": "ಓಕೆ",
    }

    def __init__(
        self,
        use_model: bool = True,
        fallback_dict: Optional[Dict[str, str]] = None,
        beam_width: int = 4,
    ) -> None:
        """
        Initialize the Transliterator.

        Args:
            use_model: Whether to attempt using the IndicXlit model.
                If False or model unavailable, uses fallback dictionary only.
            fallback_dict: Custom fallback dictionary. If None, uses default.
            beam_width: Beam width for model inference. Higher = better quality
                but slower. Default is 4.
        """
        self.use_model = use_model
        self.beam_width = beam_width
        self._engine = None
        self._model_available = False

        # Initialize fallback dictionary
        self.fallback_dict = dict(self.DEFAULT_FALLBACK_DICT)
        if fallback_dict:
            self.fallback_dict.update(fallback_dict)

        # Create lowercase version for case-insensitive matching
        self._fallback_lower = {k.lower(): v for k, v in self.fallback_dict.items()}

        # Try to load model if requested
        if self.use_model:
            self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Initialize the IndicXlit model.

        Handles errors gracefully and falls back to dictionary-based
        transliteration if model is unavailable.
        """
        try:
            from ai4bharat.transliteration import XlitEngine

            logger.info("Initializing IndicXlit transliteration model...")
            self._engine = XlitEngine(
                src_script_type="roman",
                beam_width=self.beam_width,
                rescore=False
            )
            self._model_available = True
            logger.info("IndicXlit model loaded successfully.")

        except ImportError:
            logger.warning(
                "ai4bharat-transliteration package not installed. "
                "Using fallback dictionary only."
            )
            self._model_available = False

        except Exception as e:
            # Handle fairseq compatibility issues with Python 3.11+
            if "mutable default" in str(e) or "default_factory" in str(e):
                logger.warning(
                    "IndicXlit model has compatibility issues with Python 3.11+. "
                    "Using fallback dictionary only."
                )
            else:
                logger.warning(f"Failed to load IndicXlit model: {e}. Using fallback.")
            self._model_available = False

    def _transliterate_with_model(self, word: str) -> Optional[str]:
        """
        Transliterate a single word using the IndicXlit model.

        Args:
            word: The Romanized Kannada word to transliterate.

        Returns:
            The transliterated Kannada word, or None if failed.
        """
        if not self._model_available or not self._engine:
            return None

        try:
            result = self._engine.translit_word(word, lang_code="kn", topk=1)
            if result and "kn" in result and result["kn"]:
                return result["kn"][0]
        except Exception as e:
            logger.debug(f"Model transliteration failed for '{word}': {e}")

        return None

    def _transliterate_with_fallback(self, word: str) -> Optional[str]:
        """
        Transliterate a word using the fallback dictionary.

        Args:
            word: The Romanized Kannada word to transliterate.

        Returns:
            The transliterated Kannada word, or None if not in dictionary.
        """
        # Try exact match first (case-insensitive)
        word_lower = word.lower()
        if word_lower in self._fallback_lower:
            return self._fallback_lower[word_lower]

        # Try without trailing punctuation
        word_stripped = word_lower.rstrip(".,!?;:")
        if word_stripped in self._fallback_lower:
            # Preserve punctuation
            punctuation = word_lower[len(word_stripped):]
            return self._fallback_lower[word_stripped] + punctuation

        return None

    def transliterate_word(self, word: str) -> Tuple[str, str]:
        """
        Transliterate a single word to Kannada script.

        Attempts model-based transliteration first, then falls back
        to dictionary lookup if model fails or is unavailable.

        Args:
            word: The Romanized Kannada word to transliterate.

        Returns:
            A tuple of (transliterated_word, method_used).
            method_used is one of: 'model', 'fallback', 'unchanged'.
        """
        if not word or not word.strip():
            return word, "unchanged"

        # Skip if already in Kannada script
        if self._is_kannada_script(word):
            return word, "unchanged"

        # Skip pure numbers and special characters
        if not any(c.isalpha() for c in word):
            return word, "unchanged"

        # Try model first if available
        if self._model_available:
            result = self._transliterate_with_model(word)
            if result:
                return result, "model"

        # Try fallback dictionary
        result = self._transliterate_with_fallback(word)
        if result:
            return result, "fallback"

        # Return original if no transliteration found
        logger.debug(f"No transliteration found for: '{word}'")
        return word, "unchanged"

    def transliterate_words(self, words: List[str]) -> List[str]:
        """
        Transliterate a list of words to Kannada script.

        Args:
            words: List of Romanized Kannada words.

        Returns:
            List of transliterated Kannada words.
        """
        return [self.transliterate_word(word)[0] for word in words]

    def transliterate(self, text: str) -> TransliterationResult:
        """
        Transliterate a full sentence or text to Kannada script.

        Performs word-by-word transliteration while preserving
        whitespace and punctuation patterns.

        Args:
            text: The Romanized Kannada text to transliterate.

        Returns:
            A TransliterationResult object containing the transliterated text
            and metadata about the transliteration process.

        Example:
            >>> t = Transliterator()
            >>> result = t.transliterate("tumba chennagide")
            >>> print(result.transliterated)
            'ತುಂಬ ಚೆನ್ನಾಗಿದೆ'
        """
        if not text or not text.strip():
            return TransliterationResult(
                original=text,
                transliterated=text,
                word_mappings=[],
                method="unchanged",
                success=True
            )

        # Tokenize while preserving whitespace
        tokens = self._tokenize(text)
        word_mappings = []
        methods_used = set()
        transliterated_tokens = []

        for token in tokens:
            if token.isspace():
                # Preserve whitespace
                transliterated_tokens.append(token)
            else:
                # Transliterate the word
                result, method = self.transliterate_word(token)
                transliterated_tokens.append(result)
                word_mappings.append((token, result))
                methods_used.add(method)

        # Determine overall method
        if "model" in methods_used and "fallback" in methods_used:
            overall_method = "mixed"
        elif "model" in methods_used:
            overall_method = "model"
        elif "fallback" in methods_used:
            overall_method = "fallback"
        else:
            overall_method = "unchanged"

        transliterated_text = "".join(transliterated_tokens)

        return TransliterationResult(
            original=text,
            transliterated=transliterated_text,
            word_mappings=word_mappings,
            method=overall_method,
            success=True
        )

    def transliterate_simple(self, text: str) -> str:
        """
        Simple transliteration that returns only the transliterated text.

        Convenience method for cases where only the output text is needed.

        Args:
            text: The Romanized Kannada text to transliterate.

        Returns:
            The transliterated Kannada text.
        """
        return self.transliterate(text).transliterated

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words and whitespace, preserving structure.

        Args:
            text: The text to tokenize.

        Returns:
            List of tokens (words and whitespace).
        """
        # Split on whitespace while keeping the whitespace
        tokens = re.split(r'(\s+)', text)
        return [t for t in tokens if t]  # Remove empty strings

    def _is_kannada_script(self, text: str) -> bool:
        """
        Check if text contains Kannada Unicode characters.

        Args:
            text: The text to check.

        Returns:
            True if text contains Kannada characters.
        """
        for char in text:
            code_point = ord(char)
            if 0x0C80 <= code_point <= 0x0CFF:
                return True
        return False

    def add_to_fallback(self, romanized: str, kannada: str) -> None:
        """
        Add a new word mapping to the fallback dictionary.

        Args:
            romanized: The Romanized Kannada word.
            kannada: The Kannada script equivalent.
        """
        self.fallback_dict[romanized] = kannada
        self._fallback_lower[romanized.lower()] = kannada
        logger.info(f"Added fallback mapping: '{romanized}' -> '{kannada}'")

    def get_fallback_dict(self) -> Dict[str, str]:
        """
        Get a copy of the current fallback dictionary.

        Returns:
            Dictionary of Romanized to Kannada mappings.
        """
        return dict(self.fallback_dict)

    @property
    def model_available(self) -> bool:
        """Check if the IndicXlit model is available."""
        return self._model_available

    def get_status(self) -> Dict[str, any]:
        """
        Get the current status of the transliterator.

        Returns:
            Dictionary with status information.
        """
        return {
            "model_available": self._model_available,
            "fallback_dict_size": len(self.fallback_dict),
            "beam_width": self.beam_width,
            "use_model": self.use_model,
        }


# Module-level convenience function
def transliterate_to_kannada(text: str) -> str:
    """
    Convenience function to transliterate Romanized Kannada to Kannada script.

    Creates a Transliterator instance and performs transliteration.

    Args:
        text: The Romanized Kannada text.

    Returns:
        The transliterated Kannada text.

    Example:
        >>> from src.preprocessing.transliterator import transliterate_to_kannada
        >>> transliterate_to_kannada("tumba chennagide")
        'ತುಂಬ ಚೆನ್ನಾಗಿದೆ'
    """
    transliterator = Transliterator()
    return transliterator.transliterate_simple(text)
