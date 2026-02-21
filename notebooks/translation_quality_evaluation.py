#!/usr/bin/env python3
"""
Translation Quality Evaluation Script

This script translates sample Kannada sentences and allows manual quality scoring
to evaluate translation performance.

Usage:
    python notebooks/translation_quality_evaluation.py [--auto] [--output FILE]

Options:
    --auto      Skip manual scoring (for automated testing)
    --output    Output CSV file path (default: data/processed/translation_quality.csv)
"""

import sys
import csv
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing import Translator, TranslationBackend


# Sample Kannada sentences for evaluation (20 sentences)
SAMPLE_SENTENCES: List[Dict[str, str]] = [
    # Positive sentiments (Product reviews)
    {
        "kannada": "ಈ ಉತ್ಪನ್ನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ",
        "expected": "This product is very good",
        "category": "positive"
    },
    {
        "kannada": "ಕ್ವಾಲಿಟಿ ಅತ್ಯುತ್ತಮವಾಗಿದೆ",
        "expected": "Quality is excellent",
        "category": "positive"
    },
    {
        "kannada": "ನಾನು ಈ ಪ್ರಾಡಕ್ಟ್ ಅನ್ನು ಶಿಫಾರಸು ಮಾಡುತ್ತೇನೆ",
        "expected": "I recommend this product",
        "category": "positive"
    },
    {
        "kannada": "ಬೆಲೆಗೆ ತಕ್ಕ ಮೌಲ್ಯ ಇದೆ",
        "expected": "Good value for the price",
        "category": "positive"
    },
    {
        "kannada": "ಡೆಲಿವರಿ ತುಂಬಾ ವೇಗವಾಗಿತ್ತು",
        "expected": "Delivery was very fast",
        "category": "positive"
    },

    # Negative sentiments
    {
        "kannada": "ಈ ಪ್ರಾಡಕ್ಟ್ ಕೆಟ್ಟದಾಗಿದೆ",
        "expected": "This product is bad",
        "category": "negative"
    },
    {
        "kannada": "ನನಗೆ ಇದು ಬೇಕಿಲ್ಲ",
        "expected": "I don't want this",
        "category": "negative"
    },
    {
        "kannada": "ಕ್ವಾಲಿಟಿ ತುಂಬಾ ಕಳಪೆಯಾಗಿದೆ",
        "expected": "Quality is very poor",
        "category": "negative"
    },
    {
        "kannada": "ಹಣ ವ್ಯರ್ಥವಾಯಿತು",
        "expected": "Money was wasted",
        "category": "negative"
    },
    {
        "kannada": "ಸರ್ವಿಸ್ ತುಂಬಾ ಕೆಟ್ಟದಾಗಿದೆ",
        "expected": "Service is very bad",
        "category": "negative"
    },

    # Neutral sentiments
    {
        "kannada": "ಈ ಉತ್ಪನ್ನ ಸರಿಯಾಗಿದೆ",
        "expected": "This product is okay",
        "category": "neutral"
    },
    {
        "kannada": "ನಿರೀಕ್ಷೆಯಂತೆ ಇದೆ",
        "expected": "It is as expected",
        "category": "neutral"
    },
    {
        "kannada": "ಬೆಲೆ ಸ್ವಲ್ಪ ಹೆಚ್ಚು",
        "expected": "Price is a bit high",
        "category": "neutral"
    },

    # Questions
    {
        "kannada": "ಈ ಉತ್ಪನ್ನ ಎಲ್ಲಿ ಸಿಗುತ್ತದೆ",
        "expected": "Where is this product available",
        "category": "question"
    },
    {
        "kannada": "ಬೆಲೆ ಎಷ್ಟು",
        "expected": "What is the price",
        "category": "question"
    },

    # Complex sentences
    {
        "kannada": "ಈ ಫೋನ್ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ ಆದರೆ ಬ್ಯಾಟರಿ ಬೇಗ ಖಾಲಿಯಾಗುತ್ತದೆ",
        "expected": "This phone is very good but battery drains quickly",
        "category": "mixed"
    },
    {
        "kannada": "ಗುಣಮಟ್ಟ ಒಳ್ಳೆಯದು ಮತ್ತು ಬೆಲೆ ಕಡಿಮೆ",
        "expected": "Quality is good and price is low",
        "category": "positive"
    },
    {
        "kannada": "ನಾನು ಮೂರು ವರ್ಷಗಳಿಂದ ಬಳಸುತ್ತಿದ್ದೇನೆ",
        "expected": "I have been using for three years",
        "category": "neutral"
    },

    # Greetings and common phrases
    {
        "kannada": "ನಮಸ್ಕಾರ ನೀವು ಹೇಗಿದ್ದೀರಿ",
        "expected": "Hello how are you",
        "category": "greeting"
    },
    {
        "kannada": "ಧನ್ಯವಾದಗಳು ನಿಮ್ಮ ಸಹಾಯಕ್ಕೆ",
        "expected": "Thank you for your help",
        "category": "greeting"
    },
]


def print_header():
    """Print script header."""
    print("\n" + "=" * 80)
    print(" TRANSLATION QUALITY EVALUATION")
    print(" Kannada → English Translation Assessment")
    print("=" * 80 + "\n")


def print_table_header():
    """Print the table header."""
    print("\n" + "-" * 100)
    print(f"{'#':<3} | {'Kannada':<35} | {'English Translation':<35} | {'Score':<6}")
    print("-" * 100)


def format_text(text: str, max_len: int = 35) -> str:
    """Truncate text if too long for display."""
    if len(text) > max_len:
        return text[:max_len - 3] + "..."
    return text


def get_quality_score(prompt: str = "Enter quality score (1-5): ") -> int:
    """Get quality score from user input."""
    while True:
        try:
            score = input(prompt).strip()
            if score.lower() == 'q':
                return -1  # Signal to quit
            if score.lower() == 's':
                return 0  # Signal to skip
            score = int(score)
            if 1 <= score <= 5:
                return score
            print("  Please enter a number between 1 and 5.")
        except ValueError:
            print("  Invalid input. Enter 1-5, 's' to skip, or 'q' to quit.")


def evaluate_translations(
    translator: Translator,
    sentences: List[Dict[str, str]],
    auto_mode: bool = False
) -> List[Dict]:
    """
    Translate sentences and optionally collect quality scores.

    Args:
        translator: Translator instance to use.
        sentences: List of sentence dictionaries with 'kannada' and 'expected' keys.
        auto_mode: If True, skip manual scoring.

    Returns:
        List of evaluation results.
    """
    results = []

    print_table_header()

    for idx, sentence in enumerate(sentences, 1):
        kannada = sentence["kannada"]
        expected = sentence["expected"]
        category = sentence.get("category", "unknown")

        # Translate
        result = translator.translate(kannada)
        translated = result.translated

        # Display
        print(f"{idx:<3} | {format_text(kannada):<35} | {format_text(translated):<35} |", end=" ")

        if auto_mode:
            # Auto-calculate basic similarity score
            score = calculate_similarity_score(translated, expected)
            print(f"{score:.1f}")
        else:
            print()
            print(f"    Expected: {expected}")
            print(f"    Got:      {translated}")
            print(f"    Category: {category}")
            print()

            # Get manual score
            print("    Quality: 1=Poor, 2=Bad, 3=OK, 4=Good, 5=Excellent")
            score = get_quality_score("    Score (1-5, 's'=skip, 'q'=quit): ")

            if score == -1:  # Quit
                print("\n  Evaluation stopped by user.")
                break
            elif score == 0:  # Skip
                score = None
                print("    Skipped.")
            else:
                print(f"    Recorded: {score}/5")

        # Store result
        results.append({
            "id": idx,
            "kannada": kannada,
            "expected_english": expected,
            "translated_english": translated,
            "category": category,
            "backend": result.backend.value,
            "quality_score": score,
            "success": result.success,
            "timestamp": datetime.now().isoformat()
        })

        print("-" * 100)

    return results


def calculate_similarity_score(translated: str, expected: str) -> float:
    """
    Calculate a basic similarity score between translated and expected text.

    This is a simple word overlap metric, not a proper translation quality metric.
    """
    translated_words = set(translated.lower().split())
    expected_words = set(expected.lower().split())

    if not expected_words:
        return 0.0

    overlap = len(translated_words & expected_words)
    score = (overlap / len(expected_words)) * 5  # Scale to 1-5

    return min(max(score, 1.0), 5.0)


def save_results_to_csv(results: List[Dict], output_path: Path) -> None:
    """Save evaluation results to CSV file."""
    if not results:
        print("No results to save.")
        return

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    fieldnames = [
        "id", "kannada", "expected_english", "translated_english",
        "category", "backend", "quality_score", "success", "timestamp"
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_path}")


def print_summary(results: List[Dict]) -> None:
    """Print evaluation summary statistics."""
    if not results:
        print("No results to summarize.")
        return

    scores = [r["quality_score"] for r in results if r["quality_score"] is not None]

    print("\n" + "=" * 60)
    print(" EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total sentences:    {len(results)}")
    print(f"  Scored sentences:   {len(scores)}")

    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"  Average score:      {avg_score:.2f} / 5.0")
        print(f"  Min score:          {min(scores)}")
        print(f"  Max score:          {max(scores)}")

        # Score distribution
        print("\n  Score Distribution:")
        for s in range(1, 6):
            count = scores.count(s)
            bar = "█" * count + "░" * (len(scores) - count)
            pct = (count / len(scores)) * 100 if scores else 0
            print(f"    {s}: {bar} ({count}, {pct:.1f}%)")

        # Category breakdown
        print("\n  By Category:")
        categories = {}
        for r in results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = []
            if r["quality_score"] is not None:
                categories[cat].append(r["quality_score"])

        for cat, cat_scores in sorted(categories.items()):
            if cat_scores:
                avg = sum(cat_scores) / len(cat_scores)
                print(f"    {cat:<12}: {avg:.2f}/5 ({len(cat_scores)} samples)")

    # Backend info
    backends = set(r["backend"] for r in results)
    print(f"\n  Backend(s) used:    {', '.join(backends)}")

    success_rate = sum(1 for r in results if r["success"]) / len(results) * 100
    print(f"  Success rate:       {success_rate:.1f}%")

    print("=" * 60 + "\n")


def display_all_translations(results: List[Dict]) -> None:
    """Display all translations in a formatted table."""
    print("\n" + "=" * 110)
    print(" ALL TRANSLATIONS")
    print("=" * 110)
    print(f"{'#':<3} | {'Kannada':<30} | {'Expected':<25} | {'Translated':<25} | {'Score':<5}")
    print("-" * 110)

    for r in results:
        score_str = str(r['quality_score']) if r['quality_score'] else '-'
        print(
            f"{r['id']:<3} | "
            f"{format_text(r['kannada'], 30):<30} | "
            f"{format_text(r['expected_english'], 25):<25} | "
            f"{format_text(r['translated_english'], 25):<25} | "
            f"{score_str:<5}"
        )

    print("-" * 110)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate Kannada to English translation quality"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Skip manual scoring (use automatic similarity)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/translation_quality.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["indictrans2", "google", "fallback"],
        default="fallback",
        help="Translation backend to use"
    )

    args = parser.parse_args()

    # Print header
    print_header()

    # Initialize translator
    print(f"Initializing translator (backend: {args.backend})...")
    translator = Translator(backend=args.backend, auto_fallback=True)

    status = translator.get_status()
    print(f"  Active backend: {status['active_backend']}")
    print(f"  Available backends: {[k for k, v in status['backends'].items() if v['available']]}")

    if not args.auto:
        print("\n" + "-" * 60)
        print("INSTRUCTIONS:")
        print("  - Review each translation and rate quality from 1-5")
        print("  - 1=Poor, 2=Bad, 3=OK, 4=Good, 5=Excellent")
        print("  - Enter 's' to skip a sentence")
        print("  - Enter 'q' to quit early")
        print("-" * 60)

        input("\nPress Enter to start evaluation...")

    # Evaluate translations
    results = evaluate_translations(
        translator=translator,
        sentences=SAMPLE_SENTENCES,
        auto_mode=args.auto
    )

    # Display all translations
    display_all_translations(results)

    # Print summary
    print_summary(results)

    # Save results
    output_path = project_root / args.output
    save_results_to_csv(results, output_path)

    print("Evaluation complete!")


if __name__ == "__main__":
    main()
