
"""
Language Detection Module for Kannada Sentiment Analysis.

This module detects whether a piece of text is written in native Kannada
Unicode script (ಕನ್ನಡ), Romanized Kannada (Kanglish), plain English, a
mixture of the two, or an unknown language.

Detection strategy
------------------
1. **Unicode-block ratio** — count characters in the Kannada Unicode block
   (U+0C80–U+0CFF) vs. ASCII alpha characters.  A high Kannada ratio
   immediately resolves the case; a high ASCII ratio triggers step 2.
2. **Regex heuristics** — match common Romanized Kannada word patterns
   (e.g. "tumba", "chennagi", "bengaluru") against the Latin text.
3. **langdetect** — Google's language-detection library provides a
   supplementary probability signal for disambiguation between Romanized
   Kannada and ordinary English.

Thresholds (configurable via ``LanguageDetector.__init__``)
-----------------------------------------------------------
* ``kannada_threshold``  (default 0.60) — min Kannada-char proportion for
  ``'kannada_script'`` classification.
* ``english_threshold``  (default 0.80) — min ASCII-alpha proportion for
  high-confidence ``'english'`` classification.
* ``mixed_threshold``    (default 0.40) — below this, neither script is
  dominant enough → ``'mixed'``.

Public symbols
--------------
* :class:`DetectionResult` — dataclass holding the full detection output.
* :class:`LanguageDetector` — main detection class.
* :func:`detect_language`   — module-level convenience wrapper.

Example
-------
>>> from src.preprocessing.language_detector import LanguageDetector
>>> detector = LanguageDetector()

>>> # Native Kannada script
>>> r = detector.detect("ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ")
>>> r.language, round(r.confidence, 2)
('kannada_script', 1.0)

>>> # Romanized Kannada (Kanglish)
>>> r = detector.detect("tumba chennagide, olle product")
>>> r.language
'romanized_kannada'

>>> # Plain English
>>> r = detector.detect("This product has excellent quality.")
>>> r.language
'english'
"""

import re
from typing import Literal, Tuple, Dict
from dataclasses import dataclass

from langdetect import detect, detect_langs, LangDetectException


# Type alias for language detection results
LanguageType = Literal["kannada_script", "romanized_kannada", "english", "mixed", "unknown"]


@dataclass
class DetectionResult:
    """
    Data class to hold language detection results.

    Attributes:
        language: The detected language type.
        confidence: Confidence score between 0 and 1.
        script_proportions: Dictionary with proportions of different scripts.
        details: Additional details about the detection.

    Example:
        >>> from src.preprocessing.language_detector import LanguageDetector
        >>> r = LanguageDetector().detect("ಕನ್ನಡ")
        >>> r.language
        'kannada_script'
        >>> r.confidence
        1.0
    """
    language: LanguageType
    confidence: float
    script_proportions: Dict[str, float]
    details: str


class LanguageDetector:
    """
    A class for detecting language and script type in text input.

    This detector is specifically designed for Kannada sentiment analysis
    and can distinguish between:
    - Native Kannada script (ಕನ್ನಡ)
    - Romanized Kannada (Kannada written in Latin script)
    - English text
    - Mixed language text

    Attributes:
        kannada_threshold: Minimum proportion of Kannada characters to classify as Kannada script.
        english_threshold: Minimum proportion of English characters to classify as English.
        mixed_threshold: Threshold below which text is considered mixed.

    Example:
        >>> detector = LanguageDetector()
        >>> result = detector.detect("ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ")
        >>> print(result.language)
        'kannada_script'

        >>> result = detector.detect("Ee product tumba chennagide")
        >>> print(result.language)
        'romanized_kannada'
    """

    # Unicode range for Kannada script: U+0C80 to U+0CFF
    KANNADA_UNICODE_START = 0x0C80
    KANNADA_UNICODE_END = 0x0CFF

    # Common Romanized Kannada patterns and words.
    # Each pattern group covers a different semantic category so that even a
    # single match from any group is a reliable indicator of Romanized Kannada.
    ROMANIZED_KANNADA_PATTERNS = [
        # Intensifiers / quality descriptors
        r'\b(tumba|chenag|olle|ketta|chennagi|chennagide|chennagilla|jaasti|jaasthi|kammi)\b',
        # Verb endings and copulas — the most diagnostic markers
        r'\b(agide|agilla|agidhe|aagide|aagilla|madidini|madidhe|madidu|hogbekilla|baralla|gottilla)\b',
        # Emotional / experiential words
        r'\b(ishta|ishtavagide|ishtailla|khushi|santosha|kashtavagide|kasta|bejaar|hodedru)\b',
        # Place names and cultural markers
        r'\b(kannada|bengaluru|karnataka|mysuru|mangaluru|hubli|dharwad|belgaum)\b',
        # Kinship / address terms
        r'\b(guru|anna|akka|amma|appa|maga|magalu|avru|avara|avrige)\b',
        # Common verbs / auxiliary words
        r'\b(saku|beku|illa|ide|agi|idu|yenu|hegidira|mattu|aadare|modalige)\b',
        # Demonstratives, pronouns, question words (excluding single-char 'ee'/'aa' to avoid false positives)
        r'\b(adu|idu|nanna|ninna|ivara|yavaga|elli|hege|yaake|yaru)\b',
        # Transliteration artifacts common in reviews
        r'\b(namask[aā]ra|dhanyavada|dhanyavaad|gottu|barli|hogbeku|madli)\b',
    ]

    def __init__(
        self,
        kannada_threshold: float = 0.6,
        english_threshold: float = 0.8,
        mixed_threshold: float = 0.4
    ) -> None:
        """
        Initialize the LanguageDetector with configurable thresholds.

        Args:
            kannada_threshold: Minimum proportion of Kannada Unicode characters
                required to classify text as 'kannada_script'. Default is 0.6.
            english_threshold: Minimum proportion of ASCII alphabetic characters
                required to classify text as 'english'. Default is 0.8.
            mixed_threshold: If the dominant script proportion falls below this
                threshold, text is classified as 'mixed'. Default is 0.4.
        """
        self.kannada_threshold = kannada_threshold
        self.english_threshold = english_threshold
        self.mixed_threshold = mixed_threshold

        # Compile romanized Kannada patterns for efficiency
        self._romanized_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.ROMANIZED_KANNADA_PATTERNS
        ]

    def _is_kannada_char(self, char: str) -> bool:
        """
        Check if a character belongs to the Kannada Unicode block.

        Args:
            char: A single character to check.

        Returns:
            True if the character is in the Kannada Unicode range (U+0C80-U+0CFF).
        """
        code_point = ord(char)
        return self.KANNADA_UNICODE_START <= code_point <= self.KANNADA_UNICODE_END

    def _is_english_char(self, char: str) -> bool:
        """
        Check if a character is an English/ASCII alphabetic character.

        Args:
            char: A single character to check.

        Returns:
            True if the character is an ASCII letter (a-z, A-Z).
        """
        return char.isascii() and char.isalpha()

    def _calculate_script_proportions(self, text: str) -> Dict[str, float]:
        """
        Calculate the proportion of different scripts in the text.

        Args:
            text: The input text to analyze.

        Returns:
            A dictionary containing proportions for:
            - 'kannada': Proportion of Kannada Unicode characters
            - 'english': Proportion of ASCII alphabetic characters
            - 'numeric': Proportion of numeric characters
            - 'other': Proportion of other characters (punctuation, spaces, etc.)
        """
        if not text:
            return {'kannada': 0.0, 'english': 0.0, 'numeric': 0.0, 'other': 0.0}

        kannada_count = 0
        english_count = 0
        numeric_count = 0
        other_count = 0

        # Only count non-whitespace characters for proportions
        text_no_space = ''.join(text.split())
        total_chars = len(text_no_space)

        if total_chars == 0:
            return {'kannada': 0.0, 'english': 0.0, 'numeric': 0.0, 'other': 0.0}

        for char in text_no_space:
            if self._is_kannada_char(char):
                kannada_count += 1
            elif self._is_english_char(char):
                english_count += 1
            elif char.isdigit():
                numeric_count += 1
            else:
                other_count += 1

        return {
            'kannada': kannada_count / total_chars,
            'english': english_count / total_chars,
            'numeric': numeric_count / total_chars,
            'other': other_count / total_chars
        }

    def _detect_romanized_kannada(self, text: str) -> Tuple[bool, float]:
        """
        Detect if text is Romanized Kannada using pattern matching and langdetect.

        This method uses two approaches:
        1. Pattern matching for common Kannada words written in Latin script
        2. The langdetect library to identify the language

        Args:
            text: The input text to analyze.

        Returns:
            A tuple of (is_romanized_kannada, confidence_score).
        """
        # Count absolute matches across all pattern groups.
        # Any genuine Kannada word in a review is a strong indicator — even a
        # single match suffices to mark the text as Romanized Kannada.
        pattern_matches = 0
        for pattern in self._romanized_patterns:
            matches = pattern.findall(text.lower())
            pattern_matches += len(matches)

        # Absolute-count confidence: saturates at 1.0 after 3 matches.
        # Using absolute counts rather than a word-count ratio prevents
        # code-switched reviews (many English loanwords + few Kannada words)
        # from being mis-classified as English.
        pattern_confidence = min(pattern_matches / 3.0, 1.0)

        # Use langdetect as a supporting signal.
        # langdetect never returns 'kn' for Latin-script Romanized Kannada;
        # it typically returns 'ro', 'id', 'af', or other non-English codes.
        # Treat any non-English top detection as a positive signal.
        langdetect_boost = 0.0
        try:
            detected_langs = detect_langs(text)
            if detected_langs:
                top = detected_langs[0]
                if top.lang != 'en':
                    # Non-English detection supports the Romanized Kannada hypothesis
                    langdetect_boost = top.prob * 0.3
        except LangDetectException:
            pass

        combined_confidence = min((pattern_confidence * 0.7) + langdetect_boost, 1.0)

        # Positive if even one Kannada indicator word was found (absolute count ≥ 1)
        is_romanized = pattern_matches >= 1

        return is_romanized, combined_confidence

    def _detect_with_langdetect(self, text: str) -> Tuple[str, float]:
        """
        Use the langdetect library to detect language.

        Args:
            text: The input text to analyze.

        Returns:
            A tuple of (language_code, confidence_score).
            Returns ('unknown', 0.0) if detection fails.
        """
        try:
            detected_langs = detect_langs(text)
            if detected_langs:
                top_lang = detected_langs[0]
                return top_lang.lang, top_lang.prob
        except LangDetectException:
            pass

        return 'unknown', 0.0

    def detect(self, text: str) -> DetectionResult:
        """
        Detect the language and script type of the input text.

        This is the main method for language detection. It analyzes the input
        text and determines whether it is:
        - 'kannada_script': Text written in native Kannada Unicode script
        - 'romanized_kannada': Kannada language written in Latin/Roman script
        - 'english': English language text
        - 'mixed': Text containing significant portions of multiple languages
        - 'unknown': Unable to determine the language

        Args:
            text: The input text to analyze. Can be any Unicode string.

        Returns:
            A DetectionResult object containing:
            - language: The detected language type
            - confidence: Confidence score (0.0 to 1.0)
            - script_proportions: Dictionary with character type proportions
            - details: Human-readable explanation of the detection

        Example:
            >>> detector = LanguageDetector()
            >>> result = detector.detect("ಕನ್ನಡ ಭಾಷೆ")
            >>> print(f"{result.language}: {result.confidence:.2f}")
            'kannada_script: 1.00'
        """
        # Handle empty or whitespace-only input
        if not text or not text.strip():
            return DetectionResult(
                language="unknown",
                confidence=0.0,
                script_proportions={'kannada': 0.0, 'english': 0.0, 'numeric': 0.0, 'other': 0.0},
                details="Empty or whitespace-only input"
            )

        text = text.strip()

        # Calculate script proportions
        proportions = self._calculate_script_proportions(text)

        # Case 1: Predominantly Kannada script
        if proportions['kannada'] >= self.kannada_threshold:
            return DetectionResult(
                language="kannada_script",
                confidence=proportions['kannada'],
                script_proportions=proportions,
                details=f"Detected {proportions['kannada']*100:.1f}% Kannada Unicode characters"
            )

        # Case 2: Predominantly English/Latin characters
        if proportions['english'] >= self.mixed_threshold:
            # Check if it might be Romanized Kannada
            is_romanized, romanized_confidence = self._detect_romanized_kannada(text)

            if is_romanized:
                return DetectionResult(
                    language="romanized_kannada",
                    confidence=romanized_confidence,
                    script_proportions=proportions,
                    details="Detected Kannada language patterns in Latin script"
                )

            # Use langdetect to verify English
            lang_code, lang_confidence = self._detect_with_langdetect(text)

            if lang_code == 'en' and proportions['english'] >= self.english_threshold:
                return DetectionResult(
                    language="english",
                    confidence=lang_confidence,
                    script_proportions=proportions,
                    details=f"Detected English with {lang_confidence*100:.1f}% confidence"
                )
            elif lang_code == 'en':
                return DetectionResult(
                    language="english",
                    confidence=lang_confidence * proportions['english'],
                    script_proportions=proportions,
                    details=f"Likely English ({proportions['english']*100:.1f}% Latin characters)"
                )

            # Fallback: If text is predominantly Latin characters with no Kannada,
            # and not detected as Romanized Kannada, classify as English
            if proportions['english'] >= self.english_threshold and proportions['kannada'] == 0:
                return DetectionResult(
                    language="english",
                    confidence=proportions['english'],
                    script_proportions=proportions,
                    details=f"Classified as English based on {proportions['english']*100:.1f}% Latin characters"
                )

        # Case 3: Mixed language text
        if proportions['kannada'] >= self.mixed_threshold or proportions['english'] >= self.mixed_threshold:
            dominant_script = 'kannada' if proportions['kannada'] > proportions['english'] else 'english'
            return DetectionResult(
                language="mixed",
                confidence=max(proportions['kannada'], proportions['english']),
                script_proportions=proportions,
                details=f"Mixed text with {dominant_script} as dominant script"
            )

        # Case 4: Unable to determine
        return DetectionResult(
            language="unknown",
            confidence=0.0,
            script_proportions=proportions,
            details="Unable to determine language with sufficient confidence"
        )

    def detect_simple(self, text: str) -> LanguageType:
        """
        Simplified detection that returns only the language type.

        This is a convenience method for cases where only the language
        classification is needed without detailed metrics.

        Args:
            text: The input text to analyze.

        Returns:
            One of: 'kannada_script', 'romanized_kannada', 'english', 'mixed', 'unknown'

        Example:
            >>> detector = LanguageDetector()
            >>> detector.detect_simple("Hello world")
            'english'
        """
        return self.detect(text).language

    def is_kannada(self, text: str) -> bool:
        """
        Check if the text is in Kannada (either script or romanized).

        Args:
            text: The input text to analyze.

        Returns:
            True if text is either Kannada script or Romanized Kannada.

        Example:
            >>> detector = LanguageDetector()
            >>> detector.is_kannada("ಕನ್ನಡ")
            True
            >>> detector.is_kannada("kannada")
            True
        """
        language = self.detect_simple(text)
        return language in ("kannada_script", "romanized_kannada")

    def get_kannada_char_count(self, text: str) -> int:
        """
        Count the number of Kannada Unicode characters in the text.

        Args:
            text: The input text to analyze.

        Returns:
            The count of characters in the Kannada Unicode range.
        """
        return sum(1 for char in text if self._is_kannada_char(char))


# Module-level convenience function
def detect_language(text: str) -> LanguageType:
    """
    Convenience function for quick language detection.

    Creates a LanguageDetector instance with default settings and
    returns the detected language type.

    Args:
        text: The input text to analyze.

    Returns:
        One of: 'kannada_script', 'romanized_kannada', 'english', 'mixed', 'unknown'

    Example:
        >>> from src.preprocessing.language_detector import detect_language
        >>> detect_language("ನಮಸ್ಕಾರ")
        'kannada_script'
    """
    detector = LanguageDetector()
    return detector.detect_simple(text)
