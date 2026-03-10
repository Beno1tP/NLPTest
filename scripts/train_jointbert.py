#!/usr/bin/env python3
"""Training script for JointBERT model with PhoBERT encoder.

Usage:
    python scripts/train_jointbert.py
    python scripts/train_jointbert.py --config configs/joint_bert_config.yaml
    python scripts/train_jointbert.py --batch-size 8 --epochs 5  # For CPU/low memory
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nlu.jointbert_model import create_model
from src.nlu.jointbert_data import create_data_module
from src.nlu.jointbert_trainer import JointBERTTrainer, evaluate_model


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train JointBERT model for Vietnamese NLU"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/joint_bert_config.yaml",
        help="Path to configuration file",
    )

    # Override config options
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="HuggingFace model name (default: vinai/phobert-base-v2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: 32, use 8 for CPU/low memory)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: 15)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Maximum sequence length (default: 128)",
    )
    parser.add_argument(
        "--use-crf",
        action="store_true",
        help="Use CRF layer for slot prediction",
    )
    parser.add_argument(
        "--use-intent-slot-attention",
        action="store_true",
        help="Use intent-slot attention variant",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (cuda/cpu/mps)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only run evaluation on test set (requires trained model)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint for evaluation",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Load config
    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        config = load_config(str(config_path))
        print(f"Loaded config from: {config_path}")
    else:
        print(f"Config file not found: {config_path}")
        print("Using default configuration")
        config = {
            "model": {"name": "vinai/phobert-base-v2", "dropout": 0.1},
            "training": {
                "learning_rate": 5e-5,
                "batch_size": 32,
                "epochs": 15,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "max_seq_length": 128,
                "early_stopping_patience": 3,
            },
            "paths": {
                "checkpoint_dir": "models/checkpoints/jointbert",
                "best_model": "models/best_jointbert.pt",
            },
        }

    # Override config with command line args
    model_name = args.model_name or config["model"]["name"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    epochs = args.epochs or config["training"]["epochs"]
    learning_rate = args.learning_rate or config["training"]["learning_rate"]
    max_seq_length = args.max_seq_length or config["training"]["max_seq_length"]
    dropout = config["model"].get("dropout", 0.1)
    warmup_ratio = config["training"].get("warmup_ratio", 0.1)
    weight_decay = config["training"].get("weight_decay", 0.01)
    patience = config["training"].get("early_stopping_patience", 3)

    checkpoint_dir = args.output_dir or config["paths"]["checkpoint_dir"]
    best_model_path = config["paths"]["best_model"]

    # Determine device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\n{'='*60}")
    print("JointBERT Training Configuration")
    print(f"{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max seq length: {max_seq_length}")
    print(f"  Dropout: {dropout}")
    print(f"  Warmup ratio: {warmup_ratio}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Use CRF: {args.use_crf}")
    print(f"  Use intent-slot attention: {args.use_intent_slot_attention}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"{'='*60}\n")

    # Create data module
    print("Loading data...")
    data_module = create_data_module(
        model_name=model_name,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
    )

    print(f"  Train samples: {len(data_module.get_dataset('train'))}")
    print(f"  Dev samples: {len(data_module.get_dataset('dev'))}")
    print(f"  Test samples: {len(data_module.get_dataset('test'))}")
    print(f"  Intents: {data_module.num_intents}")
    print(f"  Slot labels: {data_module.num_slots}")

    # Evaluation only mode
    if args.evaluate_only:
        model_path = args.model_path or str(PROJECT_ROOT / best_model_path)
        print(f"\nEvaluation only mode. Loading model from: {model_path}")

        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint.get("model_config", {})

        model = create_model(
            model_name=model_name,
            num_intents=data_module.num_intents,
            num_slots=data_module.num_slots,
            dropout=dropout,
            use_crf=model_config.get("use_crf", args.use_crf),
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        # Evaluate on test set
        test_dataloader = data_module.get_test_dataloader()
        metrics, report = evaluate_model(
            model=model,
            dataloader=test_dataloader,
            id2intent=data_module.id2intent,
            id2slot=data_module.id2slot,
            device=device,
        )

        print(f"\n{'='*60}")
        print("Test Set Results")
        print(f"{'='*60}")
        print(f"  Intent Accuracy: {metrics['intent_accuracy']:.4f}")
        print(f"  Slot F1: {metrics['slot_f1']:.4f}")
        print(f"  Sentence Accuracy: {metrics['sentence_accuracy']:.4f}")
        print(f"\nSlot Classification Report:")
        print(report)

        return

    # Create model
    print("\nCreating model...")
    model = create_model(
        model_name=model_name,
        num_intents=data_module.num_intents,
        num_slots=data_module.num_slots,
        dropout=dropout,
        use_crf=args.use_crf,
        use_intent_slot_attention=args.use_intent_slot_attention,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = JointBERTTrainer(
        model=model,
        train_dataloader=data_module.get_train_dataloader(),
        dev_dataloader=data_module.get_dev_dataloader(),
        id2intent=data_module.id2intent,
        id2slot=data_module.id2slot,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        epochs=epochs,
        patience=patience,
        checkpoint_dir=PROJECT_ROOT / checkpoint_dir,
        device=device,
    )

    # Train
    results = trainer.train()

    # Copy best model to final location
    best_checkpoint = PROJECT_ROOT / checkpoint_dir / "best_model.pt"
    final_model_path = PROJECT_ROOT / best_model_path
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(best_checkpoint), str(final_model_path))
    print(f"\nBest model saved to: {final_model_path}")

    # Final evaluation on test set
    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60)

    # Load best model
    checkpoint = torch.load(str(best_checkpoint), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_dataloader = data_module.get_test_dataloader()
    metrics, report = evaluate_model(
        model=model,
        dataloader=test_dataloader,
        id2intent=data_module.id2intent,
        id2slot=data_module.id2slot,
        device=device,
    )

    print(f"\nTest Set Results:")
    print(f"  Intent Accuracy: {metrics['intent_accuracy']:.4f}")
    print(f"  Slot F1: {metrics['slot_f1']:.4f}")
    print(f"  Sentence Accuracy: {metrics['sentence_accuracy']:.4f}")
    print(f"\nSlot Classification Report:")
    print(report)

    # Save final results
    results_path = PROJECT_ROOT / checkpoint_dir / "test_results.yaml"
    with open(results_path, "w", encoding="utf-8") as f:
        yaml.dump({
            "intent_accuracy": float(metrics["intent_accuracy"]),
            "slot_f1": float(metrics["slot_f1"]),
            "sentence_accuracy": float(metrics["sentence_accuracy"]),
            "best_epoch": results["best_epoch"],
            "training_time_minutes": results["total_time"] / 60,
        }, f)
    print(f"\nResults saved to: {results_path}")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
