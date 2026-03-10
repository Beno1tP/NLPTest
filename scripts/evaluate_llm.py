#!/usr/bin/env python3
"""Evaluate LLM zero-shot NLU on PhoATIS test set.

Computes intent accuracy, slot F1, and other metrics for comparison
with trained SVM and JointBERT models.

Usage:
    python scripts/evaluate_llm.py --provider mock
    python scripts/evaluate_llm.py --provider anthropic --max-samples 100
    python scripts/evaluate_llm.py --provider openai --config configs/llm_config.yaml
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import LLMDataLoader
from src.nlu.llm_nlu import LLMNLUClassifier, create_llm_classifier


def compute_intent_metrics(
    y_true: list,
    y_pred: list,
) -> dict:
    """Compute intent classification metrics.

    Args:
        y_true: True intent labels
        y_pred: Predicted intent labels

    Returns:
        Dictionary with accuracy, precision, recall, f1
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        classification_report,
    )

    accuracy = accuracy_score(y_true, y_pred)

    # Macro-averaged metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Per-class report (for later analysis)
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    return {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4),
        "per_class": report,
    }


def compute_slot_metrics(
    y_true_slots: list,
    y_pred_slots: list,
) -> dict:
    """Compute slot filling metrics using seqeval.

    Args:
        y_true_slots: List of true BIO tag sequences
        y_pred_slots: List of predicted BIO tag sequences

    Returns:
        Dictionary with precision, recall, f1 for slot filling
    """
    try:
        from seqeval.metrics import (
            precision_score,
            recall_score,
            f1_score,
            classification_report,
        )

        precision = precision_score(y_true_slots, y_pred_slots, zero_division=0)
        recall = recall_score(y_true_slots, y_pred_slots, zero_division=0)
        f1 = f1_score(y_true_slots, y_pred_slots, zero_division=0)

        # Per-entity report
        report = classification_report(
            y_true_slots, y_pred_slots, output_dict=True, zero_division=0
        )

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "per_entity": report,
        }

    except ImportError:
        print("Warning: seqeval not installed. Computing simple token-level metrics.")

        # Fallback: token-level accuracy
        correct = 0
        total = 0
        for true_seq, pred_seq in zip(y_true_slots, y_pred_slots):
            for t, p in zip(true_seq, pred_seq):
                if t == p:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0
        return {
            "token_accuracy": round(accuracy, 4),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }


def compute_sentence_accuracy(
    intent_true: list,
    intent_pred: list,
    slots_true: list,
    slots_pred: list,
) -> float:
    """Compute sentence-level accuracy (both intent and all slots correct).

    Args:
        intent_true: True intent labels
        intent_pred: Predicted intent labels
        slots_true: True slot sequences
        slots_pred: Predicted slot sequences

    Returns:
        Sentence accuracy score
    """
    correct = 0
    for it, ip, st, sp in zip(intent_true, intent_pred, slots_true, slots_pred):
        if it == ip and st == sp:
            correct += 1

    return round(correct / len(intent_true), 4) if intent_true else 0.0


def evaluate_llm_nlu(
    classifier: LLMNLUClassifier,
    texts: list,
    true_intents: list,
    true_slots: list,
    show_progress: bool = True,
) -> dict:
    """Run full evaluation of LLM NLU.

    Args:
        classifier: LLM NLU classifier
        texts: Input utterances
        true_intents: Ground truth intent labels
        true_slots: Ground truth BIO slot sequences
        show_progress: Whether to show progress bar

    Returns:
        Dictionary with all metrics and predictions
    """
    print(f"\nEvaluating {len(texts)} samples...")
    start_time = time.time()

    # Get predictions with BIO format
    predictions = classifier.predict_batch_with_bio(texts, show_progress=show_progress)

    elapsed = time.time() - start_time
    print(f"Evaluation completed in {elapsed:.1f}s ({elapsed/len(texts):.2f}s/sample)")

    # Extract predictions
    pred_intents = [p[0] for p in predictions]
    pred_confidences = [p[1] for p in predictions]
    pred_slots = [p[2] for p in predictions]

    # Compute metrics
    intent_metrics = compute_intent_metrics(true_intents, pred_intents)
    slot_metrics = compute_slot_metrics(true_slots, pred_slots)
    sentence_acc = compute_sentence_accuracy(
        true_intents, pred_intents, true_slots, pred_slots
    )

    # Confidence statistics
    avg_confidence = sum(pred_confidences) / len(pred_confidences)
    high_conf_correct = sum(
        1 for i, c in enumerate(pred_confidences)
        if c >= 0.8 and pred_intents[i] == true_intents[i]
    )
    high_conf_total = sum(1 for c in pred_confidences if c >= 0.8)
    high_conf_acc = high_conf_correct / high_conf_total if high_conf_total > 0 else 0

    # Statistics from classifier
    stats = classifier.get_statistics()

    return {
        "intent": intent_metrics,
        "slot": slot_metrics,
        "sentence_accuracy": sentence_acc,
        "confidence": {
            "average": round(avg_confidence, 4),
            "high_confidence_accuracy": round(high_conf_acc, 4),
            "high_confidence_samples": high_conf_total,
        },
        "statistics": stats,
        "timing": {
            "total_seconds": round(elapsed, 2),
            "samples_per_second": round(len(texts) / elapsed, 2),
        },
        "predictions": {
            "intents": pred_intents,
            "slots": pred_slots,
            "confidences": pred_confidences,
        },
    }


def convert_to_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_results(results: dict, output_path: Path, provider: str):
    """Save evaluation results to JSON file.

    Args:
        results: Evaluation results dictionary
        output_path: Path to save results
        provider: Provider name for metadata
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata
    results["metadata"] = {
        "provider": provider,
        "timestamp": datetime.now().isoformat(),
        "model": results.get("model", "unknown"),
    }

    # Remove large prediction lists for summary file
    summary = {k: v for k, v in results.items() if k != "predictions"}

    # Convert numpy types to JSON-serializable
    summary = convert_to_serializable(summary)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")

    # Save full predictions separately
    pred_path = output_path.with_stem(output_path.stem + "_predictions")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(results["predictions"], f, indent=2, ensure_ascii=False)

    print(f"Predictions saved to: {pred_path}")


def print_results_summary(results: dict):
    """Print formatted results summary.

    Args:
        results: Evaluation results dictionary
    """
    print("\n" + "=" * 60)
    print("LLM Zero-Shot NLU Evaluation Results")
    print("=" * 60)

    intent = results["intent"]
    slot = results["slot"]

    print(f"\n--- Intent Classification ---")
    print(f"  Accuracy:  {intent['accuracy']:.4f}")
    print(f"  F1 Macro:  {intent['f1_macro']:.4f}")
    print(f"  Precision: {intent['precision_macro']:.4f}")
    print(f"  Recall:    {intent['recall_macro']:.4f}")

    print(f"\n--- Slot Filling ---")
    print(f"  F1:        {slot['f1']:.4f}")
    print(f"  Precision: {slot['precision']:.4f}")
    print(f"  Recall:    {slot['recall']:.4f}")

    print(f"\n--- Overall ---")
    print(f"  Sentence Accuracy: {results['sentence_accuracy']:.4f}")

    conf = results["confidence"]
    print(f"\n--- Confidence ---")
    print(f"  Average:      {conf['average']:.4f}")
    print(f"  High-conf (>=0.8) Accuracy: {conf['high_confidence_accuracy']:.4f}")
    print(f"  High-conf Samples: {conf['high_confidence_samples']}")

    timing = results["timing"]
    print(f"\n--- Timing ---")
    print(f"  Total time:  {timing['total_seconds']:.1f}s")
    print(f"  Speed:       {timing['samples_per_second']:.2f} samples/s")

    stats = results["statistics"]
    if stats["errors"] > 0:
        print(f"\n--- Errors ---")
        print(f"  Error count: {stats['errors']}")
        print(f"  Error rate:  {stats['error_rate']:.3f}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM zero-shot NLU on PhoATIS test set"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="mock",
        choices=["anthropic", "openai", "mock"],
        help="LLM provider (default: mock)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name override",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/llm_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (for cost control)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Data split to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: results/llm_evaluation_{provider}.json)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Delay between API calls (seconds)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )

    args = parser.parse_args()

    # Load config
    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        print(f"Loading config from: {config_path}")

    # Load data
    print(f"\nLoading {args.split} data...")
    loader = LLMDataLoader()

    # Determine sample size
    max_samples = args.max_samples
    if max_samples is None and config_path.exists():
        try:
            import yaml
            config = yaml.safe_load(config_path.read_text())
            max_samples = config.get("evaluation", {}).get("max_samples")
        except Exception:
            pass

    texts, intents, slots = loader.load(
        args.split,
        sample_size=max_samples,
        seed=args.seed,
    )

    print(f"Loaded {len(texts)} samples")
    print(f"Intents: {len(set(intents))} unique")

    # Create classifier
    print(f"\nInitializing {args.provider} classifier...")
    classifier = LLMNLUClassifier(
        provider=args.provider,
        model=args.model,
        rate_limit_delay=args.rate_limit,
    )

    # Run evaluation
    results = evaluate_llm_nlu(
        classifier=classifier,
        texts=texts,
        true_intents=intents,
        true_slots=slots,
        show_progress=not args.no_progress,
    )

    # Add model info
    results["model"] = args.model or classifier.client.model

    # Print summary
    print_results_summary(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PROJECT_ROOT / f"results/llm_evaluation_{args.provider}.json"

    save_results(results, output_path, args.provider)

    # Return metrics for programmatic use
    return results


if __name__ == "__main__":
    main()
