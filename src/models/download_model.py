#!/usr/bin/env python3
"""
Model Download Script for Kannada Sentiment Analysis.

This script downloads pre-trained models from HuggingFace and saves them
locally for offline use. It downloads:
1. distilbert-base-uncased - Base model for fine-tuning
2. A pre-trained sentiment classifier (SST-2 fine-tuned)

Usage:
    python src/models/download_model.py [--models MODEL1,MODEL2] [--cache-dir DIR]

Options:
    --models      Comma-separated list of models to download
    --cache-dir   Directory to save models (default: data/models/)
    --verify      Verify models after download
    --list        List available models
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enum for model types."""
    BASE = "base"
    SENTIMENT = "sentiment"
    MULTILINGUAL = "multilingual"


@dataclass
class ModelInfo:
    """Information about a downloadable model."""
    name: str
    huggingface_id: str
    model_type: ModelType
    description: str
    size_mb: int  # Approximate size in MB
    task: str


# Available models for download
AVAILABLE_MODELS: Dict[str, ModelInfo] = {
    # Base models
    "distilbert-base": ModelInfo(
        name="distilbert-base",
        huggingface_id="distilbert-base-uncased",
        model_type=ModelType.BASE,
        description="DistilBERT base model (uncased) - Good for fine-tuning",
        size_mb=250,
        task="feature-extraction"
    ),
    "bert-base": ModelInfo(
        name="bert-base",
        huggingface_id="bert-base-uncased",
        model_type=ModelType.BASE,
        description="BERT base model (uncased) - Standard transformer",
        size_mb=420,
        task="feature-extraction"
    ),

    # Sentiment models (pre-trained on SST-2 or similar)
    "distilbert-sst2": ModelInfo(
        name="distilbert-sst2",
        huggingface_id="distilbert-base-uncased-finetuned-sst-2-english",
        model_type=ModelType.SENTIMENT,
        description="DistilBERT fine-tuned on SST-2 for sentiment analysis",
        size_mb=250,
        task="sentiment-analysis"
    ),
    "bert-sst2": ModelInfo(
        name="bert-sst2",
        huggingface_id="textattack/bert-base-uncased-SST-2",
        model_type=ModelType.SENTIMENT,
        description="BERT fine-tuned on SST-2 dataset",
        size_mb=420,
        task="sentiment-analysis"
    ),
    "roberta-sentiment": ModelInfo(
        name="roberta-sentiment",
        huggingface_id="cardiffnlp/twitter-roberta-base-sentiment-latest",
        model_type=ModelType.SENTIMENT,
        description="RoBERTa fine-tuned on tweets for sentiment (3 classes)",
        size_mb=500,
        task="sentiment-analysis"
    ),

    # Multilingual models (support Kannada)
    "xlm-roberta-base": ModelInfo(
        name="xlm-roberta-base",
        huggingface_id="xlm-roberta-base",
        model_type=ModelType.MULTILINGUAL,
        description="XLM-RoBERTa base - Supports 100 languages including Kannada",
        size_mb=1100,
        task="feature-extraction"
    ),
    "mbert": ModelInfo(
        name="mbert",
        huggingface_id="bert-base-multilingual-cased",
        model_type=ModelType.MULTILINGUAL,
        description="Multilingual BERT - Supports 104 languages",
        size_mb=680,
        task="feature-extraction"
    ),
    "muril": ModelInfo(
        name="muril",
        huggingface_id="google/muril-base-cased",
        model_type=ModelType.MULTILINGUAL,
        description="MuRIL - Specifically trained on Indian languages",
        size_mb=890,
        task="feature-extraction"
    ),

    # Multilingual sentiment
    "nlptown-sentiment": ModelInfo(
        name="nlptown-sentiment",
        huggingface_id="nlptown/bert-base-multilingual-uncased-sentiment",
        model_type=ModelType.SENTIMENT,
        description="Multilingual BERT for sentiment (5 stars rating)",
        size_mb=680,
        task="sentiment-analysis"
    ),
}

# Default models to download
DEFAULT_MODELS = ["distilbert-base", "distilbert-sst2"]


class ModelDownloader:
    """
    Downloads and manages HuggingFace models for sentiment analysis.

    This class handles downloading models and tokenizers from HuggingFace,
    saving them locally, and verifying they work correctly.

    Attributes:
        cache_dir: Directory to save downloaded models.
        models: Dictionary of available model information.

    Example:
        >>> downloader = ModelDownloader()
        >>> downloader.download_model("distilbert-sst2")
        >>> model, tokenizer = downloader.load_model("distilbert-sst2")
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the ModelDownloader.

        Args:
            cache_dir: Directory to save models. Defaults to 'data/models/'.
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Default to project's data/models directory
            self.cache_dir = Path(__file__).parent.parent.parent / "data" / "models"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models = AVAILABLE_MODELS

    def list_available_models(self) -> None:
        """Print list of available models."""
        print("\n" + "=" * 80)
        print(" AVAILABLE MODELS")
        print("=" * 80)

        # Group by type
        by_type: Dict[ModelType, List[ModelInfo]] = {}
        for model_info in self.models.values():
            if model_info.model_type not in by_type:
                by_type[model_info.model_type] = []
            by_type[model_info.model_type].append(model_info)

        for model_type in ModelType:
            if model_type in by_type:
                print(f"\n{model_type.value.upper()} MODELS:")
                print("-" * 60)
                for info in by_type[model_type]:
                    status = "✓" if self._is_downloaded(info.name) else " "
                    print(f"  [{status}] {info.name:<20} (~{info.size_mb}MB)")
                    print(f"       {info.description}")
                    print(f"       HuggingFace: {info.huggingface_id}")

        print("\n" + "=" * 80)
        print("  [✓] = Already downloaded")
        print(f"  Cache directory: {self.cache_dir}")
        print("=" * 80 + "\n")

    def _is_downloaded(self, model_name: str) -> bool:
        """Check if a model is already downloaded."""
        model_path = self.cache_dir / model_name
        return model_path.exists() and any(model_path.iterdir())

    def _get_model_path(self, model_name: str) -> Path:
        """Get the local path for a model."""
        return self.cache_dir / model_name

    def download_model(
        self,
        model_name: str,
        force: bool = False
    ) -> Tuple[bool, str]:
        """
        Download a model from HuggingFace.

        Args:
            model_name: Name of the model to download (from AVAILABLE_MODELS).
            force: If True, re-download even if model exists.

        Returns:
            Tuple of (success: bool, message: str).
        """
        if model_name not in self.models:
            return False, f"Unknown model: {model_name}. Use --list to see available models."

        model_info = self.models[model_name]
        model_path = self._get_model_path(model_name)

        # Check if already downloaded
        if self._is_downloaded(model_name) and not force:
            return True, f"Model '{model_name}' already downloaded at {model_path}"

        logger.info(f"Downloading {model_name} from HuggingFace...")
        logger.info(f"  HuggingFace ID: {model_info.huggingface_id}")
        logger.info(f"  Estimated size: ~{model_info.size_mb}MB")
        logger.info(f"  Destination: {model_path}")

        try:
            from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

            # Create directory
            model_path.mkdir(parents=True, exist_ok=True)

            # Download tokenizer
            logger.info("Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_info.huggingface_id,
                cache_dir=str(self.cache_dir / ".cache")
            )
            tokenizer.save_pretrained(str(model_path))
            logger.info("Tokenizer saved.")

            # Download model (use appropriate class based on task)
            logger.info("Downloading model...")
            if model_info.task == "sentiment-analysis":
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_info.huggingface_id,
                    cache_dir=str(self.cache_dir / ".cache")
                )
            else:
                model = AutoModel.from_pretrained(
                    model_info.huggingface_id,
                    cache_dir=str(self.cache_dir / ".cache")
                )
            model.save_pretrained(str(model_path))
            logger.info("Model saved.")

            # Create info file
            info_file = model_path / "model_info.txt"
            with open(info_file, 'w') as f:
                f.write(f"Model Name: {model_name}\n")
                f.write(f"HuggingFace ID: {model_info.huggingface_id}\n")
                f.write(f"Type: {model_info.model_type.value}\n")
                f.write(f"Task: {model_info.task}\n")
                f.write(f"Description: {model_info.description}\n")

            return True, f"Successfully downloaded '{model_name}' to {model_path}"

        except ImportError:
            return False, "transformers library not installed. Run: pip install transformers"

        except Exception as e:
            # Clean up partial download
            if model_path.exists():
                import shutil
                shutil.rmtree(model_path)
            return False, f"Download failed: {str(e)}"

    def download_multiple(
        self,
        model_names: List[str],
        force: bool = False
    ) -> Dict[str, Tuple[bool, str]]:
        """
        Download multiple models.

        Args:
            model_names: List of model names to download.
            force: If True, re-download existing models.

        Returns:
            Dictionary mapping model names to (success, message) tuples.
        """
        results = {}

        print(f"\nDownloading {len(model_names)} model(s)...\n")

        for i, name in enumerate(model_names, 1):
            print(f"[{i}/{len(model_names)}] {name}")
            print("-" * 50)

            success, message = self.download_model(name, force=force)
            results[name] = (success, message)

            if success:
                print(f"✓ {message}")
            else:
                print(f"✗ {message}")

            print()

        return results

    def verify_model(self, model_name: str) -> Tuple[bool, str]:
        """
        Verify that a downloaded model loads correctly.

        Args:
            model_name: Name of the model to verify.

        Returns:
            Tuple of (success: bool, message: str).
        """
        if not self._is_downloaded(model_name):
            return False, f"Model '{model_name}' is not downloaded."

        model_path = self._get_model_path(model_name)
        model_info = self.models.get(model_name)

        logger.info(f"Verifying model: {model_name}")

        try:
            from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

            # Load tokenizer
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))

            # Load model
            logger.info("Loading model...")
            if model_info and model_info.task == "sentiment-analysis":
                model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            else:
                model = AutoModel.from_pretrained(str(model_path))

            # Test tokenization
            logger.info("Testing tokenization...")
            test_text = "This is a test sentence for verification."
            inputs = tokenizer(test_text, return_tensors="pt")

            # Test forward pass
            logger.info("Testing forward pass...")
            outputs = model(**inputs)

            # Check outputs
            if hasattr(outputs, 'last_hidden_state'):
                output_shape = outputs.last_hidden_state.shape
            elif hasattr(outputs, 'logits'):
                output_shape = outputs.logits.shape
            else:
                output_shape = "unknown"

            message = (
                f"Model '{model_name}' verified successfully!\n"
                f"  Path: {model_path}\n"
                f"  Tokenizer vocab size: {tokenizer.vocab_size}\n"
                f"  Output shape: {output_shape}"
            )

            return True, message

        except Exception as e:
            return False, f"Verification failed: {str(e)}"

    def load_model(self, model_name: str):
        """
        Load a downloaded model and tokenizer.

        Args:
            model_name: Name of the model to load.

        Returns:
            Tuple of (model, tokenizer).

        Raises:
            ValueError: If model is not downloaded.
        """
        if not self._is_downloaded(model_name):
            raise ValueError(f"Model '{model_name}' is not downloaded. Run download first.")

        model_path = self._get_model_path(model_name)
        model_info = self.models.get(model_name)

        from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        if model_info and model_info.task == "sentiment-analysis":
            model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        else:
            model = AutoModel.from_pretrained(str(model_path))

        return model, tokenizer

    def get_downloaded_models(self) -> List[str]:
        """Get list of downloaded model names."""
        downloaded = []
        for name in self.models:
            if self._is_downloaded(name):
                downloaded.append(name)
        return downloaded

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a model."""
        return self.models.get(model_name)


def print_header():
    """Print script header."""
    print("\n" + "=" * 60)
    print(" SENTIMENT ANALYSIS MODEL DOWNLOADER")
    print(" Downloads pre-trained models from HuggingFace")
    print("=" * 60)


def print_summary(results: Dict[str, Tuple[bool, str]]):
    """Print download summary."""
    print("\n" + "=" * 60)
    print(" DOWNLOAD SUMMARY")
    print("=" * 60)

    success_count = sum(1 for s, _ in results.values() if s)
    total = len(results)

    print(f"\n  Total: {total}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {total - success_count}")

    if total - success_count > 0:
        print("\n  Failed models:")
        for name, (success, message) in results.items():
            if not success:
                print(f"    - {name}: {message}")

    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download pre-trained models for sentiment analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/models/download_model.py --list
  python src/models/download_model.py --models distilbert-base,distilbert-sst2
  python src/models/download_model.py --models xlm-roberta-base --verify
  python src/models/download_model.py --all
        """
    )

    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to download"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to save models (default: data/models/)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify models after download"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download existing models"
    )
    parser.add_argument(
        "--default",
        action="store_true",
        help="Download default models (distilbert-base, distilbert-sst2)"
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = ModelDownloader(cache_dir=args.cache_dir)

    # Handle --list
    if args.list:
        downloader.list_available_models()
        return

    print_header()
    print(f"\nCache directory: {downloader.cache_dir}\n")

    # Determine which models to download
    if args.all:
        model_names = list(AVAILABLE_MODELS.keys())
    elif args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    elif args.default:
        model_names = DEFAULT_MODELS
    else:
        # Interactive mode - show list and ask
        downloader.list_available_models()
        print("Use --models to specify models, --default for defaults, or --all for everything.")
        print(f"Default models: {', '.join(DEFAULT_MODELS)}")

        response = input("\nDownload default models? (y/n): ").strip().lower()
        if response == 'y':
            model_names = DEFAULT_MODELS
        else:
            return

    # Download models
    results = downloader.download_multiple(model_names, force=args.force)

    # Verify if requested
    if args.verify:
        print("\n" + "=" * 60)
        print(" VERIFYING MODELS")
        print("=" * 60 + "\n")

        for name in model_names:
            if results[name][0]:  # Only verify successful downloads
                print(f"Verifying {name}...")
                success, message = downloader.verify_model(name)
                if success:
                    print(f"✓ {message}\n")
                else:
                    print(f"✗ {message}\n")

    # Print summary
    print_summary(results)

    # Show downloaded models
    downloaded = downloader.get_downloaded_models()
    if downloaded:
        print(f"\nDownloaded models: {', '.join(downloaded)}")
        print(f"Location: {downloader.cache_dir}")


if __name__ == "__main__":
    main()
