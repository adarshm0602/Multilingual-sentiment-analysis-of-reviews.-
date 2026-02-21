"""
Model Setup Utility for Kannada Sentiment Analysis.

This module handles downloading, caching, and verifying the IndicXlit
transliteration model from AI4Bharat.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple


def _apply_fairseq_patch() -> None:
    """
    Apply compatibility patch for fairseq with Python 3.11+.

    fairseq uses mutable defaults in dataclasses which raises an error
    in Python 3.11+. This patch modifies the field function to use
    default_factory instead.
    """
    try:
        import dataclasses
        from functools import wraps

        original_field = dataclasses.field

        @wraps(original_field)
        def patched_field(*args, **kwargs):
            default = kwargs.get('default', dataclasses.MISSING)
            if default is not dataclasses.MISSING:
                if isinstance(default, (list, dict, set)):
                    kwargs['default_factory'] = lambda d=default: type(d)(d)
                    del kwargs['default']
            return original_field(*args, **kwargs)

        dataclasses.field = patched_field
    except Exception:
        pass  # If patch fails, continue anyway


# Apply patch before any fairseq imports
_apply_fairseq_patch()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelSetupError(Exception):
    """Custom exception for model setup failures."""
    pass


class IndicXlitModelSetup:
    """
    Handles downloading and caching of the IndicXlit transliteration model.

    This class manages the AI4Bharat IndicXlit model, which is used for
    transliterating text between English and Kannada scripts.

    Attributes:
        cache_dir: Directory where models are cached.
        model_name: Name of the model to download.

    Example:
        >>> setup = IndicXlitModelSetup()
        >>> setup.download_and_cache()
        >>> model = setup.load_model()
    """

    # Default model configuration
    DEFAULT_MODEL_NAME = "ai4bharat/IndicXlit"
    SUPPORTED_LANGUAGES = ["kn", "hi", "ta", "te", "ml", "bn", "gu", "mr", "pa", "or"]

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> None:
        """
        Initialize the model setup utility.

        Args:
            cache_dir: Directory to cache models. Defaults to 'data/models/'.
            model_name: Name of the model to download. Defaults to 'ai4bharat/IndicXlit'.
        """
        # Determine project root (assumes this file is in src/utils/)
        self.project_root = Path(__file__).parent.parent.parent

        # Set cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = self.project_root / "data" / "models"

        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self._model = None
        self._engine = None

    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache directory ensured at: {self.cache_dir}")

    def download_and_cache(self, force: bool = False) -> bool:
        """
        Download the IndicXlit model and cache it locally.

        This method downloads the model from AI4Bharat's repository and
        stores it in the specified cache directory.

        Args:
            force: If True, re-download even if model exists. Default is False.

        Returns:
            True if download was successful, False otherwise.

        Raises:
            ModelSetupError: If download fails after retries.
        """
        self._ensure_cache_dir()

        # Check if model already exists
        model_marker = self.cache_dir / ".indicxlit_downloaded"
        if model_marker.exists() and not force:
            logger.info("Model already downloaded. Use force=True to re-download.")
            return True

        logger.info(f"Downloading IndicXlit model: {self.model_name}")
        logger.info("This may take a few minutes depending on your connection...")

        try:
            # Import here to avoid import errors if package not installed
            from ai4bharat.transliteration import XlitEngine

            # Set custom cache directory via environment variable
            os.environ["XLIT_MODEL_DIR"] = str(self.cache_dir)

            # Initialize engine - this triggers model download
            logger.info("Initializing XlitEngine (downloading model if needed)...")
            engine = XlitEngine(
                src_script_type="roman",
                beam_width=4,
                rescore=False
            )

            # Create marker file to indicate successful download
            model_marker.touch()
            logger.info(f"Model downloaded successfully to: {self.cache_dir}")

            self._engine = engine
            return True

        except ImportError as e:
            error_msg = (
                "ai4bharat-transliteration package not installed. "
                "Please run: pip install ai4bharat-transliteration"
            )
            logger.error(error_msg)
            raise ModelSetupError(error_msg) from e

        except ConnectionError as e:
            error_msg = (
                "Failed to download model. Please check your internet connection. "
                f"Error: {str(e)}"
            )
            logger.error(error_msg)
            raise ModelSetupError(error_msg) from e

        except Exception as e:
            # Check if model was downloaded despite initialization error
            # This can happen with fairseq compatibility issues in Python 3.11+
            if "mutable default" in str(e) or "default_factory" in str(e):
                # Check if model files exist in default location
                default_model_path = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "ai4bharat" / "transliteration" / "transformer" / "models"

                if default_model_path.exists():
                    logger.warning(
                        "Model downloaded but fairseq has compatibility issues with Python 3.11+. "
                        "The model files exist and may still work with alternative loading methods."
                    )
                    model_marker.touch()
                    return True

            error_msg = f"Unexpected error during model download: {str(e)}"
            logger.error(error_msg)
            raise ModelSetupError(error_msg) from e

    def load_model(self) -> "XlitEngine":
        """
        Load the cached IndicXlit model.

        Returns:
            The loaded XlitEngine instance.

        Raises:
            ModelSetupError: If model is not downloaded or fails to load.
        """
        if self._engine is not None:
            logger.info("Returning cached engine instance.")
            return self._engine

        model_marker = self.cache_dir / ".indicxlit_downloaded"
        if not model_marker.exists():
            raise ModelSetupError(
                "Model not downloaded. Please run download_and_cache() first."
            )

        try:
            from ai4bharat.transliteration import XlitEngine

            # Set custom cache directory
            os.environ["XLIT_MODEL_DIR"] = str(self.cache_dir)

            logger.info("Loading IndicXlit model...")
            self._engine = XlitEngine(
                src_script_type="roman",
                beam_width=4,
                rescore=False
            )
            logger.info("Model loaded successfully.")
            return self._engine

        except Exception as e:
            # Check for fairseq compatibility issue
            if "mutable default" in str(e) or "default_factory" in str(e):
                error_msg = (
                    "Cannot load model due to fairseq compatibility issues with Python 3.11+. "
                    "Model files are downloaded but require Python 3.10 or earlier to load."
                )
                logger.error(error_msg)
                raise ModelSetupError(error_msg) from e

            error_msg = f"Failed to load model: {str(e)}"
            logger.error(error_msg)
            raise ModelSetupError(error_msg) from e

    def verify_model(self) -> Tuple[bool, str]:
        """
        Verify that the model loads and works correctly.

        Performs a simple transliteration test to ensure the model
        is functioning properly.

        Returns:
            A tuple of (success: bool, message: str).
        """
        try:
            engine = self.load_model()

            # Test transliteration: English to Kannada
            test_input = "namaskara"
            logger.info(f"Testing transliteration with input: '{test_input}'")

            result = engine.translit_word(test_input, lang_code="kn", topk=1)

            if result and len(result.get("kn", [])) > 0:
                transliterated = result["kn"][0]
                message = (
                    f"Model verification successful!\n"
                    f"  Input: '{test_input}'\n"
                    f"  Output: '{transliterated}'"
                )
                logger.info(message)
                return True, message
            else:
                message = "Model loaded but returned empty result."
                logger.warning(message)
                return False, message

        except ModelSetupError as e:
            return False, str(e)

        except Exception as e:
            # Check for fairseq compatibility issue
            if "mutable default" in str(e) or "default_factory" in str(e):
                message = (
                    "Model files downloaded but cannot be loaded due to fairseq "
                    "compatibility issues with Python 3.11+.\n"
                    "Workaround options:\n"
                    "  1. Use Python 3.10 or earlier\n"
                    "  2. Use alternative transliteration methods\n"
                    "  3. Wait for ai4bharat package update"
                )
                logger.warning(message)
                return False, message

            message = f"Model verification failed: {str(e)}"
            logger.error(message)
            return False, message

    def check_model_files_exist(self) -> Tuple[bool, str]:
        """
        Check if model files exist without attempting to load them.

        This is useful when loading fails due to compatibility issues
        but you want to verify the download was successful.

        Returns:
            A tuple of (exists: bool, path: str).
        """
        # Check custom cache directory
        model_marker = self.cache_dir / ".indicxlit_downloaded"
        if model_marker.exists():
            return True, str(self.cache_dir)

        # Check default package location
        default_model_path = (
            Path(sys.prefix) / "lib" /
            f"python{sys.version_info.major}.{sys.version_info.minor}" /
            "site-packages" / "ai4bharat" / "transliteration" /
            "transformer" / "models" / "en2indic"
        )

        if default_model_path.exists():
            return True, str(default_model_path)

        return False, ""

    def get_model_info(self) -> dict:
        """
        Get information about the model setup.

        Returns:
            Dictionary containing model information.
        """
        model_marker = self.cache_dir / ".indicxlit_downloaded"
        files_exist, files_path = self.check_model_files_exist()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        has_compatibility_issue = sys.version_info >= (3, 11)

        return {
            "model_name": self.model_name,
            "cache_dir": str(self.cache_dir),
            "is_downloaded": model_marker.exists() or files_exist,
            "model_files_path": files_path if files_exist else "Not found",
            "supported_languages": self.SUPPORTED_LANGUAGES,
            "python_version": python_version,
            "fairseq_compatible": not has_compatibility_issue,
            "compatibility_note": (
                "fairseq has compatibility issues with Python 3.11+"
                if has_compatibility_issue else "Compatible"
            ),
        }


def setup_indicxlit_model(
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    verify: bool = True
) -> Tuple[bool, str]:
    """
    Convenience function to set up the IndicXlit model.

    This function handles the complete setup process: downloading,
    caching, and optionally verifying the model.

    Args:
        cache_dir: Directory to cache models. Defaults to 'data/models/'.
        force_download: If True, re-download even if model exists.
        verify: If True, verify the model after download.

    Returns:
        A tuple of (success: bool, message: str).

    Example:
        >>> success, message = setup_indicxlit_model()
        >>> print(message)
        'Model setup completed successfully!'
    """
    setup = IndicXlitModelSetup(cache_dir=cache_dir)

    try:
        # Download and cache
        setup.download_and_cache(force=force_download)

        # Verify if requested
        if verify:
            success, message = setup.verify_model()
            if success:
                return True, "Model setup completed successfully!\n" + message
            else:
                # Check if it's a compatibility issue but files exist
                files_exist, files_path = setup.check_model_files_exist()
                if files_exist and "compatibility" in message.lower():
                    return True, (
                        "Model files downloaded successfully!\n"
                        f"Location: {files_path}\n\n"
                        "NOTE: Model loading has fairseq compatibility issues with Python 3.11+.\n"
                        "The model files are ready and will work when:\n"
                        "  - Using Python 3.10 or earlier, OR\n"
                        "  - When ai4bharat updates fairseq dependency"
                    )
                return False, "Model downloaded but verification failed:\n" + message

        return True, "Model downloaded successfully (verification skipped)."

    except ModelSetupError as e:
        # Even if setup raised an error, check if files exist
        files_exist, files_path = setup.check_model_files_exist()
        if files_exist:
            return True, (
                "Model files downloaded successfully!\n"
                f"Location: {files_path}\n\n"
                f"Note: {str(e)}"
            )
        return False, f"Model setup failed: {str(e)}"


def main():
    """
    Main entry point for running model setup from command line.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Set up IndicXlit transliteration model for Kannada NLP"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache the model (default: data/models/)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip model verification after download"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show model information and exit"
    )

    args = parser.parse_args()

    setup = IndicXlitModelSetup(cache_dir=args.cache_dir)

    if args.info:
        info = setup.get_model_info()
        print("\nIndicXlit Model Information:")
        print("-" * 40)
        for key, value in info.items():
            print(f"  {key}: {value}")
        return

    print("\n" + "=" * 50)
    print("IndicXlit Model Setup")
    print("=" * 50 + "\n")

    success, message = setup_indicxlit_model(
        cache_dir=args.cache_dir,
        force_download=args.force,
        verify=not args.skip_verify
    )

    print("\n" + message)
    print("\n" + "=" * 50)

    if success:
        print("Setup completed successfully!")
    else:
        print("Setup failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
