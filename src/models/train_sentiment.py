#!/usr/bin/env python3
"""
Sentiment Model Training Script for Kannada Sentiment Analysis.

This script fine-tunes a DistilBERT model on sentiment classification datasets.
It uses the HuggingFace Trainer API with configurations optimized for CPU training.

NOTE: This is OPTIONAL. The pre-trained distilbert-sst2 model works well out of
the box. Use this script only if you want to:
- Fine-tune on domain-specific data (e.g., product reviews)
- Train on Kannada-English mixed data
- Improve performance on your specific use case

Supported Datasets:
- SST-2 (Stanford Sentiment Treebank) - Binary classification
- Amazon Reviews - Multi-class (1-5 stars)
- Custom CSV files with 'text' and 'label' columns

Usage:
    # Train on SST-2 dataset
    python src/models/train_sentiment.py --dataset sst2

    # Train on custom CSV
    python src/models/train_sentiment.py --dataset custom --train-file data/train.csv

    # Resume from checkpoint
    python src/models/train_sentiment.py --resume-from checkpoints/checkpoint-500

CPU Optimization:
- Small batch sizes (4-8)
- Gradient accumulation for effective larger batches
- Mixed precision disabled (CPU doesn't benefit)
- Reduced number of epochs
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for data loading."""
    dataset_name: str = "sst2"  # sst2, amazon, custom
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    text_column: str = "text"
    label_column: str = "label"
    max_samples: Optional[int] = None  # Limit samples for quick testing
    val_split: float = 0.1


@dataclass
class ModelConfig:
    """Configuration for the model."""
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2
    max_length: int = 256  # Shorter for faster training
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """
    Configuration for training, optimized for CPU.

    Key CPU optimizations:
    - Small batch size (4-8) to fit in memory
    - Gradient accumulation to simulate larger batches
    - Fewer epochs (3-5 is usually enough for fine-tuning)
    - No mixed precision (fp16 doesn't help on CPU)
    """
    output_dir: str = "data/models/sentiment-finetuned"
    num_epochs: int = 3
    batch_size: int = 8  # Small for CPU
    gradient_accumulation_steps: int = 4  # Effective batch = 8 * 4 = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 500
    save_total_limit: int = 2  # Keep only 2 best checkpoints
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_accuracy"
    seed: int = 42


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_sst2_dataset(max_samples: Optional[int] = None) -> Tuple[Any, Any]:
    """
    Load the SST-2 dataset from HuggingFace datasets.

    SST-2 (Stanford Sentiment Treebank) is a binary sentiment classification
    dataset with movie reviews labeled as positive (1) or negative (0).

    Args:
        max_samples: Optional limit on number of samples.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    from datasets import load_dataset

    logger.info("Loading SST-2 dataset from HuggingFace...")

    # Load dataset
    dataset = load_dataset("glue", "sst2")

    train_data = dataset["train"]
    val_data = dataset["validation"]

    # Limit samples if requested (useful for testing)
    if max_samples:
        train_data = train_data.select(range(min(max_samples, len(train_data))))
        val_data = val_data.select(range(min(max_samples // 5, len(val_data))))

    logger.info(f"Loaded {len(train_data)} training samples")
    logger.info(f"Loaded {len(val_data)} validation samples")

    return train_data, val_data


def load_amazon_reviews(max_samples: Optional[int] = None) -> Tuple[Any, Any]:
    """
    Load Amazon product reviews dataset.

    This dataset has 5-star ratings which we convert to:
    - Negative: 1-2 stars
    - Neutral: 3 stars
    - Positive: 4-5 stars

    Args:
        max_samples: Optional limit on number of samples.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    from datasets import load_dataset

    logger.info("Loading Amazon Reviews dataset...")

    # Load a subset of Amazon reviews
    dataset = load_dataset("amazon_polarity", split="train")

    # Amazon polarity has binary labels (0=negative, 1=positive)
    # Rename columns to match our expected format
    dataset = dataset.rename_column("content", "text")

    # Split into train/val
    split = dataset.train_test_split(test_size=0.1, seed=42)

    train_data = split["train"]
    val_data = split["test"]

    if max_samples:
        train_data = train_data.select(range(min(max_samples, len(train_data))))
        val_data = val_data.select(range(min(max_samples // 5, len(val_data))))

    logger.info(f"Loaded {len(train_data)} training samples")
    logger.info(f"Loaded {len(val_data)} validation samples")

    return train_data, val_data


def load_custom_dataset(
    train_file: str,
    val_file: Optional[str] = None,
    text_column: str = "text",
    label_column: str = "label",
    val_split: float = 0.1,
    max_samples: Optional[int] = None
) -> Tuple[Any, Any]:
    """
    Load a custom CSV dataset.

    Expected CSV format:
        text,label
        "This is great!",1
        "This is bad!",0

    Args:
        train_file: Path to training CSV file.
        val_file: Optional path to validation CSV file.
        text_column: Name of the text column.
        label_column: Name of the label column.
        val_split: Validation split ratio if val_file not provided.
        max_samples: Optional limit on number of samples.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    from datasets import load_dataset, Dataset
    import pandas as pd

    logger.info(f"Loading custom dataset from {train_file}...")

    # Load training data
    train_df = pd.read_csv(train_file)

    # Rename columns if needed
    if text_column != "text":
        train_df = train_df.rename(columns={text_column: "text"})
    if label_column != "label":
        train_df = train_df.rename(columns={label_column: "label"})

    # Load or split validation data
    if val_file:
        val_df = pd.read_csv(val_file)
        if text_column != "text":
            val_df = val_df.rename(columns={text_column: "text"})
        if label_column != "label":
            val_df = val_df.rename(columns={label_column: "label"})
    else:
        # Split training data
        val_df = train_df.sample(frac=val_split, random_state=42)
        train_df = train_df.drop(val_df.index)

    # Limit samples if requested
    if max_samples:
        train_df = train_df.head(max_samples)
        val_df = val_df.head(max_samples // 5)

    # Convert to HuggingFace datasets
    train_data = Dataset.from_pandas(train_df)
    val_data = Dataset.from_pandas(val_df)

    logger.info(f"Loaded {len(train_data)} training samples")
    logger.info(f"Loaded {len(val_data)} validation samples")

    return train_data, val_data


# =============================================================================
# TOKENIZATION
# =============================================================================

def get_tokenize_function(tokenizer, max_length: int, text_column: str = "sentence"):
    """
    Create a tokenization function for the dataset.

    This function will be applied to each sample in the dataset to convert
    text into token IDs that the model can process.

    Args:
        tokenizer: The tokenizer to use.
        max_length: Maximum sequence length.
        text_column: Name of the text column in the dataset.

    Returns:
        Tokenization function.
    """
    def tokenize_function(examples):
        # Handle different column names (SST-2 uses 'sentence', others use 'text')
        texts = examples.get(text_column) or examples.get("text") or examples.get("sentence")

        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    return tokenize_function


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    This function is called by the Trainer to compute metrics during evaluation.

    Args:
        eval_pred: Tuple of (predictions, labels).

    Returns:
        Dictionary of metric names to values.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# =============================================================================
# TRAINER SETUP
# =============================================================================

def setup_trainer(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    training_config: TrainingConfig
):
    """
    Set up the HuggingFace Trainer with CPU-optimized settings.

    The Trainer API handles:
    - Training loop
    - Gradient accumulation
    - Checkpointing
    - Evaluation
    - Logging

    Args:
        model: The model to train.
        tokenizer: The tokenizer.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        training_config: Training configuration.

    Returns:
        Configured Trainer instance.
    """
    from transformers import Trainer, TrainingArguments

    # =========================================================================
    # Configure Training Arguments
    # These are optimized for CPU training
    # =========================================================================
    training_args = TrainingArguments(
        # Output settings
        output_dir=training_config.output_dir,
        overwrite_output_dir=True,

        # Training hyperparameters
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size * 2,  # Can be larger for eval
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,

        # Optimizer settings
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_ratio=training_config.warmup_ratio,

        # Logging
        logging_dir=f"{training_config.output_dir}/logs",
        logging_steps=training_config.logging_steps,
        logging_first_step=True,

        # Evaluation
        eval_strategy="steps",
        eval_steps=training_config.eval_steps,

        # Checkpointing
        save_strategy="steps",
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        load_best_model_at_end=training_config.load_best_model_at_end,
        metric_for_best_model=training_config.metric_for_best_model,
        greater_is_better=True,

        # CPU optimizations
        fp16=False,  # No mixed precision on CPU
        bf16=False,
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        dataloader_pin_memory=False,

        # Reproducibility
        seed=training_config.seed,

        # Disable wandb/tensorboard if not needed
        report_to="none",

        # Show progress bar
        disable_tqdm=False,
    )

    # =========================================================================
    # Create Trainer
    # =========================================================================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    return trainer


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_sentiment_model(
    data_config: DataConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    resume_from: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main function to train the sentiment model.

    This function orchestrates the entire training pipeline:
    1. Load dataset
    2. Load model and tokenizer
    3. Tokenize dataset
    4. Set up trainer
    5. Train
    6. Evaluate
    7. Save

    Args:
        data_config: Data loading configuration.
        model_config: Model configuration.
        training_config: Training configuration.
        resume_from: Optional checkpoint path to resume from.

    Returns:
        Dictionary with training results.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    results = {}

    # =========================================================================
    # STEP 1: Load Dataset
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Loading Dataset")
    logger.info("=" * 60)

    if data_config.dataset_name == "sst2":
        train_dataset, val_dataset = load_sst2_dataset(data_config.max_samples)
        text_column = "sentence"  # SST-2 uses 'sentence'
    elif data_config.dataset_name == "amazon":
        train_dataset, val_dataset = load_amazon_reviews(data_config.max_samples)
        text_column = "text"
    elif data_config.dataset_name == "custom":
        if not data_config.train_file:
            raise ValueError("--train-file required for custom dataset")
        train_dataset, val_dataset = load_custom_dataset(
            train_file=data_config.train_file,
            val_file=data_config.val_file,
            text_column=data_config.text_column,
            label_column=data_config.label_column,
            val_split=data_config.val_split,
            max_samples=data_config.max_samples,
        )
        text_column = "text"
    else:
        raise ValueError(f"Unknown dataset: {data_config.dataset_name}")

    # Determine number of labels from dataset
    if hasattr(train_dataset.features.get("label", {}), "num_classes"):
        num_labels = train_dataset.features["label"].num_classes
    else:
        num_labels = len(set(train_dataset["label"]))

    logger.info(f"Number of labels: {num_labels}")
    model_config.num_labels = num_labels

    # =========================================================================
    # STEP 2: Load Model and Tokenizer
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 2: Loading Model and Tokenizer")
    logger.info("=" * 60)

    logger.info(f"Loading tokenizer: {model_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)

    logger.info(f"Loading model: {model_config.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name,
        num_labels=model_config.num_labels,
    )

    # Log model size
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # =========================================================================
    # STEP 3: Tokenize Dataset
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 3: Tokenizing Dataset")
    logger.info("=" * 60)

    tokenize_fn = get_tokenize_function(tokenizer, model_config.max_length, text_column)

    logger.info("Tokenizing training set...")
    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in train_dataset.column_names if c not in ["label"]],
        desc="Tokenizing train"
    )

    logger.info("Tokenizing validation set...")
    val_dataset = val_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in val_dataset.column_names if c not in ["label"]],
        desc="Tokenizing val"
    )

    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # =========================================================================
    # STEP 4: Set Up Trainer
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 4: Setting Up Trainer")
    logger.info("=" * 60)

    trainer = setup_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_config=training_config,
    )

    # =========================================================================
    # STEP 5: Train
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 5: Training")
    logger.info("=" * 60)

    logger.info("Starting training...")
    logger.info(f"  Epochs: {training_config.num_epochs}")
    logger.info(f"  Batch size: {training_config.batch_size}")
    logger.info(f"  Gradient accumulation: {training_config.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {training_config.batch_size * training_config.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {training_config.learning_rate}")

    # Train (resume from checkpoint if specified)
    train_result = trainer.train(resume_from_checkpoint=resume_from)

    results["train_runtime"] = train_result.metrics.get("train_runtime", 0)
    results["train_loss"] = train_result.metrics.get("train_loss", 0)

    logger.info(f"Training completed in {results['train_runtime']:.2f} seconds")

    # =========================================================================
    # STEP 6: Evaluate
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 6: Evaluating")
    logger.info("=" * 60)

    eval_results = trainer.evaluate()
    results["eval_accuracy"] = eval_results.get("eval_accuracy", 0)
    results["eval_f1"] = eval_results.get("eval_f1", 0)
    results["eval_loss"] = eval_results.get("eval_loss", 0)

    logger.info(f"Evaluation Results:")
    logger.info(f"  Accuracy: {results['eval_accuracy']:.4f}")
    logger.info(f"  F1 Score: {results['eval_f1']:.4f}")
    logger.info(f"  Loss: {results['eval_loss']:.4f}")

    # =========================================================================
    # STEP 7: Save Model
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 7: Saving Model")
    logger.info("=" * 60)

    # Save the best model
    final_model_path = Path(training_config.output_dir) / "final"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    logger.info(f"Model saved to: {final_model_path}")

    # Save training info
    info_file = final_model_path / "training_info.txt"
    with open(info_file, 'w') as f:
        f.write(f"Base Model: {model_config.model_name}\n")
        f.write(f"Dataset: {data_config.dataset_name}\n")
        f.write(f"Num Labels: {model_config.num_labels}\n")
        f.write(f"Epochs: {training_config.num_epochs}\n")
        f.write(f"Batch Size: {training_config.batch_size}\n")
        f.write(f"Learning Rate: {training_config.learning_rate}\n")
        f.write(f"\nResults:\n")
        for key, value in results.items():
            f.write(f"  {key}: {value}\n")

    results["model_path"] = str(final_model_path)

    return results


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main entry point for command line usage."""
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for sentiment classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on SST-2 (default)
  python src/models/train_sentiment.py --dataset sst2

  # Train on SST-2 with limited samples (for testing)
  python src/models/train_sentiment.py --dataset sst2 --max-samples 1000

  # Train on custom CSV
  python src/models/train_sentiment.py --dataset custom --train-file data/train.csv

  # Resume from checkpoint
  python src/models/train_sentiment.py --resume-from checkpoints/checkpoint-500
        """
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["sst2", "amazon", "custom"],
        default="sst2",
        help="Dataset to train on"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        help="Path to training CSV (for custom dataset)"
    )
    parser.add_argument(
        "--val-file",
        type=str,
        help="Path to validation CSV (for custom dataset)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit number of training samples (for testing)"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="distilbert-base-uncased",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length"
    )

    # Training arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models/sentiment-finetuned",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume training from checkpoint"
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "=" * 70)
    print(" SENTIMENT MODEL TRAINING")
    print(" Fine-tuning DistilBERT for Sentiment Classification")
    print("=" * 70 + "\n")

    # Create configurations
    data_config = DataConfig(
        dataset_name=args.dataset,
        train_file=args.train_file,
        val_file=args.val_file,
        max_samples=args.max_samples,
    )

    model_config = ModelConfig(
        model_name=args.model,
        max_length=args.max_length,
    )

    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Dataset: {data_config.dataset_name}")
    logger.info(f"  Model: {model_config.model_name}")
    logger.info(f"  Output: {training_config.output_dir}")
    logger.info(f"  Epochs: {training_config.num_epochs}")
    logger.info(f"  Batch size: {training_config.batch_size}")
    logger.info(f"  Learning rate: {training_config.learning_rate}")

    # Train
    try:
        results = train_sentiment_model(
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
            resume_from=args.resume_from,
        )

        # Print summary
        print("\n" + "=" * 70)
        print(" TRAINING COMPLETE")
        print("=" * 70)
        print(f"\n  Model saved to: {results['model_path']}")
        print(f"  Accuracy: {results['eval_accuracy']:.4f}")
        print(f"  F1 Score: {results['eval_f1']:.4f}")
        print(f"  Training time: {results['train_runtime']:.2f} seconds")
        print("\n" + "=" * 70 + "\n")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
