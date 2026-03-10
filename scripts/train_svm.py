#!/usr/bin/env python3
"""Training script for TF-IDF + SVM Baseline NLU.

Trains SVMIntentClassifier and CRFSlotFiller on PhoATIS dataset,
saves models to models/ directory, and prints evaluation metrics.

Usage:
    python scripts/train_svm.py
    python scripts/train_svm.py --model-dir models/custom
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import SVMDataLoader
from src.nlu.crf_slot import CRFSlotFiller
from src.nlu.svm_intent import SVMIntentClassifier
from src.nlu.svm_nlu import SVMNLU


def train_and_evaluate(
    model_dir: str = "models",
    verbose: bool = True,
) -> dict:
    """Train SVM NLU models and evaluate on dev/test sets.

    Args:
        model_dir: directory to save trained models
        verbose: print detailed progress

    Returns:
        dict with all evaluation metrics
    """
    model_path = PROJECT_ROOT / model_dir
    model_path.mkdir(parents=True, exist_ok=True)

    # Load data
    if verbose:
        print("=" * 60)
        print("Loading PhoATIS dataset...")
        print("=" * 60)

    loader = SVMDataLoader()

    train_texts, train_intent_ids, train_slot_labels = loader.load("train")
    dev_texts, dev_intent_ids, dev_slot_labels = loader.load("dev")
    test_texts, test_intent_ids, test_slot_labels = loader.load("test")

    if verbose:
        print(f"  Train samples: {len(train_texts):,}")
        print(f"  Dev samples:   {len(dev_texts):,}")
        print(f"  Test samples:  {len(test_texts):,}")
        print(f"  Intents:       {loader.num_intents}")
        print(f"  Slot labels:   {loader.num_slots}")
        print()

    # --- Train Intent Classifier ---
    if verbose:
        print("=" * 60)
        print("Training SVM Intent Classifier...")
        print("=" * 60)

    intent_start = time.time()

    intent_classifier = SVMIntentClassifier(
        ngram_range=(1, 2),
        max_features=10000,
        svm_c=1.0,
        class_weight="balanced",
    )

    intent_classifier.fit(
        train_texts,
        train_intent_ids,
        loader.id2intent,
    )

    intent_train_time = time.time() - intent_start

    if verbose:
        print(f"  Training time: {intent_train_time:.2f}s")
        print(f"  Vocabulary size: {len(intent_classifier.feature_names):,}")
        print()

    # Evaluate intent on dev
    if verbose:
        print("  Evaluating on dev set...")
    dev_intent_results = intent_classifier.evaluate(dev_texts, dev_intent_ids)
    if verbose:
        print(f"    Dev Intent Accuracy: {dev_intent_results['accuracy']:.4f}")
        print(f"    Dev Intent F1 Macro: {dev_intent_results['f1_macro']:.4f}")
        print()

    # Evaluate intent on test
    if verbose:
        print("  Evaluating on test set...")
    test_intent_results = intent_classifier.evaluate(test_texts, test_intent_ids)
    if verbose:
        print(f"    Test Intent Accuracy: {test_intent_results['accuracy']:.4f}")
        print(f"    Test Intent F1 Macro: {test_intent_results['f1_macro']:.4f}")
        print()

    # Save intent model
    intent_model_path = model_path / "svm_intent.joblib"
    intent_classifier.save(str(intent_model_path))
    if verbose:
        print(f"  Saved intent model: {intent_model_path}")
        print()

    # --- Train Slot Filler ---
    if verbose:
        print("=" * 60)
        print("Training CRF Slot Filler...")
        print("=" * 60)

    slot_start = time.time()

    slot_filler = CRFSlotFiller(
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        algorithm="lbfgs",
    )

    slot_filler.fit(train_texts, train_slot_labels)

    slot_train_time = time.time() - slot_start

    if verbose:
        print(f"  Training time: {slot_train_time:.2f}s")
        print(f"  Slot labels: {slot_filler.num_slots}")
        print()

    # Evaluate slots on dev
    if verbose:
        print("  Evaluating on dev set...")
    dev_slot_results = slot_filler.evaluate(dev_texts, dev_slot_labels)
    if verbose:
        print(f"    Dev Slot F1 Weighted: {dev_slot_results['f1_weighted']:.4f}")
        print(f"    Dev Slot F1 Macro:    {dev_slot_results['f1_macro']:.4f}")
        print(f"    Dev Token Accuracy:   {dev_slot_results['token_accuracy']:.4f}")
        print()

    # Evaluate slots on test
    if verbose:
        print("  Evaluating on test set...")
    test_slot_results = slot_filler.evaluate(test_texts, test_slot_labels)
    if verbose:
        print(f"    Test Slot F1 Weighted: {test_slot_results['f1_weighted']:.4f}")
        print(f"    Test Slot F1 Macro:    {test_slot_results['f1_macro']:.4f}")
        print(f"    Test Token Accuracy:   {test_slot_results['token_accuracy']:.4f}")
        print()

    # Save slot model
    slot_model_path = model_path / "crf_slot.joblib"
    slot_filler.save(str(slot_model_path))
    if verbose:
        print(f"  Saved slot model: {slot_model_path}")
        print()

    # --- Combined NLU Evaluation ---
    if verbose:
        print("=" * 60)
        print("Combined NLU Evaluation...")
        print("=" * 60)

    nlu = SVMNLU(
        intent_classifier=intent_classifier,
        slot_filler=slot_filler,
    )
    nlu._is_trained = True

    # Full evaluation on test set
    combined_results = nlu.evaluate(test_texts, test_intent_ids, test_slot_labels)

    if verbose:
        print(f"  Test Sentence Accuracy: {combined_results['sentence_accuracy']:.4f}")
        print()

    # --- Demo predictions ---
    if verbose:
        print("=" * 60)
        print("Sample Predictions")
        print("=" * 60)

        demo_texts = [
            "tôi muốn đặt vé máy bay đi đà nẵng",
            "giá vé từ hà nội đến hồ chí minh",
            "chuyến bay nào sớm nhất vào ngày mai",
            "cho tôi biết thông tin hãng vietnam airlines",
        ]

        for text in demo_texts:
            result = nlu.predict(text)
            print(f"\n  Input: {text}")
            print(f"  Intent: {result['intent']} ({result['confidence']:.2%})")
            print(f"  Slots: {result['slots']}")

    # --- Summary ---
    if verbose:
        print()
        print("=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"  Total training time: {intent_train_time + slot_train_time:.2f}s")
        print()
        print("  Test Set Results:")
        print(f"    Intent Accuracy:    {test_intent_results['accuracy']:.4f}")
        print(f"    Intent F1 Macro:    {test_intent_results['f1_macro']:.4f}")
        print(f"    Slot F1 Weighted:   {test_slot_results['f1_weighted']:.4f}")
        print(f"    Slot F1 Macro:      {test_slot_results['f1_macro']:.4f}")
        print(f"    Sentence Accuracy:  {combined_results['sentence_accuracy']:.4f}")
        print()
        print(f"  Models saved to: {model_path}")
        print("=" * 60)

    return {
        "intent": {
            "dev": dev_intent_results,
            "test": test_intent_results,
            "train_time": intent_train_time,
        },
        "slot": {
            "dev": dev_slot_results,
            "test": test_slot_results,
            "train_time": slot_train_time,
        },
        "combined": {
            "test_sentence_accuracy": combined_results["sentence_accuracy"],
        },
        "model_dir": str(model_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + SVM baseline NLU on PhoATIS"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory to save trained models (default: models)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    results = train_and_evaluate(
        model_dir=args.model_dir,
        verbose=not args.quiet,
    )

    # Exit with error if metrics are too low
    test_intent_acc = results["intent"]["test"]["accuracy"]
    test_slot_f1 = results["slot"]["test"]["f1_weighted"]

    if test_intent_acc < 0.85:
        print(f"\nWarning: Intent accuracy ({test_intent_acc:.4f}) below target (0.90)")
    if test_slot_f1 < 0.80:
        print(f"\nWarning: Slot F1 ({test_slot_f1:.4f}) below target (0.85)")


if __name__ == "__main__":
    main()
