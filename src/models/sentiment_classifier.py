"""
Sentiment Classification Module for Kannada Sentiment Analysis.

This module provides a SentimentClassifier class that uses pre-trained
transformer models (DistilBERT) to classify text sentiment as Positive,
Negative, or Neutral with confidence scores.

The classifier is optimized for CPU inference with:
- Small batch sizes (1-2)
- Optional model quantization for faster inference
- Efficient preprocessing with truncation and padding

Pipeline Overview:
    Input Text → Tokenization → Model Inference → Softmax → Label + Confidence
"""

import logging
from typing import List, Dict, Union, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
import torch.nn.functional as F

# Configure logging
logger = logging.getLogger(__name__)


class SentimentLabel(Enum):
    """
    Enumeration of possible sentiment labels.

    The classifier maps model outputs to these standardized labels,
    regardless of the underlying model's label scheme.
    """
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"


@dataclass
class SentimentResult:
    """
    Data class containing sentiment classification results.

    Attributes:
        text: The original input text that was classified.
        label: The predicted sentiment label (Positive/Negative/Neutral).
        confidence: Confidence score for the prediction (0.0 to 1.0).
        probabilities: Dictionary of probabilities for each class.
        raw_scores: Raw logits from the model (before softmax).
    """
    text: str
    label: SentimentLabel
    confidence: float
    probabilities: Dict[str, float]
    raw_scores: Optional[List[float]] = None


class SentimentClassifier:
    """
    Sentiment classifier using pre-trained DistilBERT models.

    This class provides an easy-to-use interface for sentiment analysis
    of English text. It handles:
    - Model loading and initialization
    - Text preprocessing (tokenization, padding, truncation)
    - Batch processing for efficiency
    - Label mapping from model-specific to standardized labels

    The classifier is optimized for CPU inference with options for
    quantization to reduce memory usage and improve speed.

    Attributes:
        model_name: Name or path of the pre-trained model.
        max_length: Maximum sequence length for tokenization.
        batch_size: Batch size for processing multiple texts.
        device: Device to run inference on (cpu/cuda).

    Example:
        >>> classifier = SentimentClassifier()
        >>> result = classifier.classify("This product is amazing!")
        >>> print(f"{result.label.value}: {result.confidence:.2f}")
        Positive: 0.95

        >>> # Batch processing
        >>> texts = ["Great!", "Terrible!", "It's okay"]
        >>> results = classifier.classify_batch(texts)
    """

    # =========================================================================
    # MODEL CONFIGURATION
    # =========================================================================

    # Default model: DistilBERT fine-tuned on SST-2 (binary sentiment)
    DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

    # Alternative models and their label mappings
    # Each model may have different output labels that we map to our standard labels
    MODEL_LABEL_MAPPINGS = {
        # SST-2 models (2 classes: NEGATIVE=0, POSITIVE=1)
        "distilbert-base-uncased-finetuned-sst-2-english": {
            0: SentimentLabel.NEGATIVE,
            1: SentimentLabel.POSITIVE,
        },
        "distilbert-sst2": {
            0: SentimentLabel.NEGATIVE,
            1: SentimentLabel.POSITIVE,
        },
        "textattack/bert-base-uncased-SST-2": {
            0: SentimentLabel.NEGATIVE,
            1: SentimentLabel.POSITIVE,
        },

        # Twitter RoBERTa (3 classes: negative=0, neutral=1, positive=2)
        "cardiffnlp/twitter-roberta-base-sentiment-latest": {
            0: SentimentLabel.NEGATIVE,
            1: SentimentLabel.NEUTRAL,
            2: SentimentLabel.POSITIVE,
        },
        "roberta-sentiment": {
            0: SentimentLabel.NEGATIVE,
            1: SentimentLabel.NEUTRAL,
            2: SentimentLabel.POSITIVE,
        },

        # NLPTown (5 classes: 1-5 stars)
        # We map: 1-2 stars = Negative, 3 = Neutral, 4-5 = Positive
        "nlptown/bert-base-multilingual-uncased-sentiment": {
            0: SentimentLabel.NEGATIVE,   # 1 star
            1: SentimentLabel.NEGATIVE,   # 2 stars
            2: SentimentLabel.NEUTRAL,    # 3 stars
            3: SentimentLabel.POSITIVE,   # 4 stars
            4: SentimentLabel.POSITIVE,   # 5 stars
        },
        "nlptown-sentiment": {
            0: SentimentLabel.NEGATIVE,
            1: SentimentLabel.NEGATIVE,
            2: SentimentLabel.NEUTRAL,
            3: SentimentLabel.POSITIVE,
            4: SentimentLabel.POSITIVE,
        },
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 2,
        use_quantization: bool = False,
        device: Optional[str] = None,
    ):
        """
        Initialize the SentimentClassifier.

        Args:
            model_name: HuggingFace model name or key from MODEL_LABEL_MAPPINGS.
                If None, uses DEFAULT_MODEL.
            model_path: Local path to a downloaded model directory.
                If provided, model_name is used only for label mapping.
            max_length: Maximum sequence length for tokenization.
                Longer texts are truncated. Default is 512 (BERT's limit).
            batch_size: Number of texts to process together.
                Smaller batches use less memory. Default is 2 for CPU.
            use_quantization: If True, apply dynamic quantization to reduce
                model size and speed up CPU inference.
            device: Device for inference ('cpu' or 'cuda').
                If None, automatically detects GPU availability.

        Raises:
            ImportError: If transformers library is not installed.
        """
        # =====================================================================
        # STEP 1: Import required libraries
        # We import here to provide clear error messages if not installed
        # =====================================================================
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
            self._AutoModel = AutoModelForSequenceClassification
            self._AutoTokenizer = AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers library is required. "
                "Install with: pip install transformers"
            )

        # =====================================================================
        # STEP 2: Set configuration parameters
        # =====================================================================
        self.model_name = model_name or self.DEFAULT_MODEL
        self.model_path = model_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_quantization = use_quantization

        # Determine device (CPU/GPU)
        # For this project, we optimize for CPU as GPU may not be available
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using device: {self.device}")

        # =====================================================================
        # STEP 3: Initialize model and tokenizer
        # These are loaded lazily on first use to save memory
        # =====================================================================
        self._model = None
        self._tokenizer = None
        self._label_mapping = None
        self._is_loaded = False

        # =====================================================================
        # STEP 4: Set up label mapping for this model
        # Different models have different output schemes
        # =====================================================================
        self._setup_label_mapping()

    def _setup_label_mapping(self) -> None:
        """
        Set up label mapping based on the model name.

        This maps the model's numeric output indices to our standardized
        SentimentLabel enum (Positive, Negative, Neutral).
        """
        # Try exact match first
        if self.model_name in self.MODEL_LABEL_MAPPINGS:
            self._label_mapping = self.MODEL_LABEL_MAPPINGS[self.model_name]
        # Try partial match (for local model paths)
        else:
            for key in self.MODEL_LABEL_MAPPINGS:
                if key in self.model_name or self.model_name in key:
                    self._label_mapping = self.MODEL_LABEL_MAPPINGS[key]
                    break

        # Default to binary classification if unknown model
        if self._label_mapping is None:
            logger.warning(
                f"Unknown model '{self.model_name}'. "
                "Using default binary label mapping (0=Negative, 1=Positive)."
            )
            self._label_mapping = {
                0: SentimentLabel.NEGATIVE,
                1: SentimentLabel.POSITIVE,
            }

    def _load_model(self) -> None:
        """
        Load the model and tokenizer.

        This is called lazily on first use to avoid loading the model
        if it's never needed. Loading typically takes 2-5 seconds.

        The model is loaded in evaluation mode and optionally quantized
        for faster CPU inference.
        """
        if self._is_loaded:
            return

        logger.info(f"Loading sentiment model: {self.model_name}")

        # =====================================================================
        # STEP 1: Determine model source (local path or HuggingFace)
        # =====================================================================
        if self.model_path:
            model_source = self.model_path
            logger.info(f"Loading from local path: {model_source}")
        else:
            model_source = self.model_name
            logger.info(f"Loading from HuggingFace: {model_source}")

        # =====================================================================
        # STEP 2: Load tokenizer
        # The tokenizer converts text to token IDs that the model understands
        # =====================================================================
        logger.info("Loading tokenizer...")
        self._tokenizer = self._AutoTokenizer.from_pretrained(model_source)

        # =====================================================================
        # STEP 3: Load model
        # We use AutoModelForSequenceClassification which includes the
        # classification head on top of the base transformer
        # =====================================================================
        logger.info("Loading model...")
        self._model = self._AutoModel.from_pretrained(model_source)

        # =====================================================================
        # STEP 4: Move model to device and set to evaluation mode
        # Evaluation mode disables dropout for consistent predictions
        # =====================================================================
        self._model.to(self.device)
        self._model.eval()

        # =====================================================================
        # STEP 5: Apply quantization if requested
        # Dynamic quantization reduces model size and speeds up CPU inference
        # by converting weights from float32 to int8
        # =====================================================================
        if self.use_quantization and self.device.type == "cpu":
            logger.info("Applying dynamic quantization for CPU optimization...")
            self._model = torch.quantization.quantize_dynamic(
                self._model,
                {torch.nn.Linear},  # Quantize linear layers
                dtype=torch.qint8   # Use 8-bit integers
            )
            logger.info("Quantization applied successfully.")

        self._is_loaded = True
        logger.info("Model loaded successfully.")

        # Log model info
        num_labels = self._model.config.num_labels
        logger.info(f"Model has {num_labels} output labels")

    def _preprocess(
        self,
        texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess texts for model input.

        This method tokenizes the input texts, applies padding and truncation,
        and converts them to PyTorch tensors.

        Args:
            texts: List of text strings to preprocess.

        Returns:
            Dictionary containing:
            - input_ids: Token ID tensors [batch_size, seq_length]
            - attention_mask: Attention mask tensors [batch_size, seq_length]

        The attention_mask indicates which tokens are real (1) vs padding (0).
        """
        # =====================================================================
        # Tokenize with padding and truncation
        # - padding='longest': Pad to longest sequence in batch (efficient)
        # - truncation=True: Cut sequences longer than max_length
        # - return_tensors='pt': Return PyTorch tensors
        # =====================================================================
        encoded = self._tokenizer(
            texts,
            padding='longest',           # Pad to longest in batch
            truncation=True,             # Truncate to max_length
            max_length=self.max_length,  # Maximum sequence length
            return_tensors='pt'          # Return PyTorch tensors
        )

        # Move tensors to the appropriate device (CPU/GPU)
        return {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device)
        }

    def _postprocess(
        self,
        logits: torch.Tensor,
        texts: List[str],
        return_raw_scores: bool = False
    ) -> List[SentimentResult]:
        """
        Convert model outputs to SentimentResult objects.

        This method applies softmax to convert logits to probabilities,
        determines the predicted class, and maps it to our label scheme.

        Args:
            logits: Raw model output [batch_size, num_classes]
            texts: Original input texts (for including in results)
            return_raw_scores: Whether to include raw logits in results

        Returns:
            List of SentimentResult objects with labels and confidence scores.
        """
        # =====================================================================
        # STEP 1: Apply softmax to convert logits to probabilities
        # Softmax ensures all probabilities sum to 1
        # =====================================================================
        probabilities = F.softmax(logits, dim=-1)

        # =====================================================================
        # STEP 2: Get predicted class and confidence for each sample
        # =====================================================================
        predictions = torch.argmax(probabilities, dim=-1)
        confidences = torch.max(probabilities, dim=-1).values

        # =====================================================================
        # STEP 3: Convert to SentimentResult objects
        # =====================================================================
        results = []
        for i, text in enumerate(texts):
            pred_idx = predictions[i].item()
            confidence = confidences[i].item()
            probs = probabilities[i].tolist()

            # Map numeric prediction to SentimentLabel
            label = self._label_mapping.get(pred_idx, SentimentLabel.NEUTRAL)

            # Build probability dictionary with label names
            prob_dict = {}
            for idx, prob in enumerate(probs):
                label_name = self._label_mapping.get(idx, SentimentLabel.NEUTRAL)
                if label_name.value in prob_dict:
                    # Aggregate probabilities for same label (e.g., NLPTown)
                    prob_dict[label_name.value] += prob
                else:
                    prob_dict[label_name.value] = prob

            result = SentimentResult(
                text=text,
                label=label,
                confidence=confidence,
                probabilities=prob_dict,
                raw_scores=logits[i].tolist() if return_raw_scores else None
            )
            results.append(result)

        return results

    def classify(
        self,
        text: str,
        return_raw_scores: bool = False
    ) -> SentimentResult:
        """
        Classify the sentiment of a single text.

        This is a convenience method for classifying one text at a time.
        For multiple texts, use classify_batch() for better efficiency.

        Args:
            text: The text to classify.
            return_raw_scores: If True, include raw model logits in result.

        Returns:
            SentimentResult with label, confidence, and probabilities.

        Example:
            >>> classifier = SentimentClassifier()
            >>> result = classifier.classify("I love this product!")
            >>> print(f"{result.label.value}: {result.confidence:.2%}")
            Positive: 98.50%
        """
        results = self.classify_batch([text], return_raw_scores=return_raw_scores)
        return results[0]

    def classify_batch(
        self,
        texts: List[str],
        return_raw_scores: bool = False
    ) -> List[SentimentResult]:
        """
        Classify the sentiment of multiple texts efficiently.

        Texts are processed in batches to optimize memory usage and
        take advantage of parallel computation.

        Args:
            texts: List of texts to classify.
            return_raw_scores: If True, include raw model logits in results.

        Returns:
            List of SentimentResult objects, one per input text.

        Example:
            >>> classifier = SentimentClassifier()
            >>> texts = ["Great product!", "Terrible service", "It's okay"]
            >>> results = classifier.classify_batch(texts)
            >>> for r in results:
            ...     print(f"{r.text[:20]}: {r.label.value}")
            Great product!: Positive
            Terrible service: Negative
            It's okay: Neutral
        """
        # Load model if not already loaded
        self._load_model()

        all_results = []

        # =====================================================================
        # Process texts in batches for memory efficiency
        # Smaller batches use less memory but may be slower
        # =====================================================================
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            # Preprocess: tokenize and convert to tensors
            inputs = self._preprocess(batch_texts)

            # =====================================================================
            # Run inference with no gradient computation
            # torch.no_grad() disables gradient tracking for faster inference
            # and lower memory usage
            # =====================================================================
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits

            # Postprocess: convert logits to labels and confidence
            batch_results = self._postprocess(
                logits, batch_texts, return_raw_scores
            )
            all_results.extend(batch_results)

        return all_results

    def classify_with_neutral_threshold(
        self,
        text: str,
        neutral_threshold: float = 0.6
    ) -> SentimentResult:
        """
        Classify sentiment with a confidence threshold for neutral.

        If the model's confidence is below the threshold, the result
        is labeled as Neutral regardless of the predicted class.
        This is useful for binary models that don't have a neutral class.

        Args:
            text: The text to classify.
            neutral_threshold: Minimum confidence to accept prediction.
                If confidence < threshold, returns Neutral.

        Returns:
            SentimentResult with potentially adjusted label.

        Example:
            >>> classifier = SentimentClassifier()
            >>> result = classifier.classify_with_neutral_threshold(
            ...     "The product is okay I guess",
            ...     neutral_threshold=0.7
            ... )
            >>> # If confidence < 0.7, will return Neutral
        """
        result = self.classify(text)

        # If confidence is below threshold, change to Neutral
        if result.confidence < neutral_threshold:
            return SentimentResult(
                text=result.text,
                label=SentimentLabel.NEUTRAL,
                confidence=result.confidence,
                probabilities=result.probabilities,
                raw_scores=result.raw_scores
            )

        return result

    def get_sentiment_scores(
        self,
        text: str
    ) -> Dict[str, float]:
        """
        Get sentiment probabilities as a simple dictionary.

        This is a convenience method that returns just the probability
        distribution without the full SentimentResult object.

        Args:
            text: The text to analyze.

        Returns:
            Dictionary mapping sentiment labels to probabilities.

        Example:
            >>> classifier = SentimentClassifier()
            >>> scores = classifier.get_sentiment_scores("Great product!")
            >>> print(scores)
            {'Positive': 0.98, 'Negative': 0.02}
        """
        result = self.classify(text)
        return result.probabilities

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model configuration and status.
        """
        info = {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "is_loaded": self._is_loaded,
            "device": str(self.device),
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "quantized": self.use_quantization,
            "num_labels": len(self._label_mapping),
            "label_mapping": {
                k: v.value for k, v in self._label_mapping.items()
            }
        }

        if self._is_loaded:
            info["vocab_size"] = self._tokenizer.vocab_size
            info["model_type"] = self._model.config.model_type

        return info

    def warmup(self) -> None:
        """
        Warm up the model by running a dummy inference.

        This pre-loads the model and runs a test inference to ensure
        everything is initialized. Useful before timing benchmarks
        or in production to avoid cold-start latency.
        """
        logger.info("Warming up model...")
        _ = self.classify("This is a warmup sentence.")
        logger.info("Model warmup complete.")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def classify_sentiment(
    text: str,
    model_name: Optional[str] = None
) -> Tuple[str, float]:
    """
    Convenience function for quick sentiment classification.

    Creates a classifier instance and classifies the text.
    For multiple texts, instantiate SentimentClassifier directly.

    Args:
        text: The text to classify.
        model_name: Optional model name to use.

    Returns:
        Tuple of (sentiment_label, confidence_score).

    Example:
        >>> label, confidence = classify_sentiment("I love this!")
        >>> print(f"{label}: {confidence:.2%}")
        Positive: 96.00%
    """
    classifier = SentimentClassifier(model_name=model_name)
    result = classifier.classify(text)
    return result.label.value, result.confidence


def load_classifier_from_config() -> SentimentClassifier:
    """
    Load a SentimentClassifier with settings from config.yaml.

    Returns:
        Configured SentimentClassifier instance.
    """
    try:
        import yaml

        config_path = Path(__file__).parent.parent.parent / "config.yaml"

        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            sentiment_config = config.get("models", {}).get("sentiment", {})
            processing_config = config.get("processing", {})

            # Check for local model path
            model_path = None
            models_dir = Path(__file__).parent.parent.parent / "data" / "models"
            local_model = models_dir / "distilbert-sst2"
            if local_model.exists():
                model_path = str(local_model)

            return SentimentClassifier(
                model_name=sentiment_config.get("name", "distilbert-sst2"),
                model_path=model_path,
                max_length=processing_config.get("max_text_length", 512),
                batch_size=processing_config.get("batch_size", 2),
            )

    except Exception as e:
        logger.warning(f"Failed to load config: {e}. Using defaults.")

    return SentimentClassifier()
