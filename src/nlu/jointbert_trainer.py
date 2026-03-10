"""Trainer for JointBERT model with PhoBERT encoder.

Provides:
- Training loop with gradient accumulation
- Validation with early stopping
- Learning rate scheduling with warmup
- Best model checkpointing
- Metrics tracking and logging
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from seqeval.metrics import f1_score as seqeval_f1_score
from seqeval.metrics import classification_report


class JointBERTTrainer:
    """Trainer for JointBERT model.

    Handles:
    - Training loop with optional gradient accumulation
    - Validation with multiple metrics
    - Early stopping based on validation metric
    - Learning rate scheduling with linear warmup
    - Model checkpointing (best model saved)
    - Training history logging

    Args:
        model: JointBERT model instance
        train_dataloader: Training DataLoader
        dev_dataloader: Validation DataLoader
        id2intent: ID to intent label mapping
        id2slot: ID to slot label mapping
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        warmup_ratio: Fraction of training steps for warmup
        epochs: Maximum number of training epochs
        patience: Early stopping patience (epochs without improvement)
        checkpoint_dir: Directory to save checkpoints
        device: Device to train on (auto-detected if None)
        gradient_accumulation_steps: Number of steps to accumulate gradients
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        dev_dataloader: DataLoader,
        id2intent: Dict[int, str],
        id2slot: Dict[int, str],
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        epochs: int = 15,
        patience: int = 3,
        checkpoint_dir: Optional[Path] = None,
        device: Optional[torch.device] = None,
        gradient_accumulation_steps: int = 1,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.id2intent = id2intent
        self.id2slot = id2slot

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.epochs = epochs
        self.patience = patience
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Setup checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("models/checkpoints/jointbert")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)

        # Training state
        self.best_metric = 0.0
        self.best_epoch = 0
        self.no_improvement_count = 0
        self.history: List[Dict] = []

        # Setup optimizer and scheduler
        self._setup_optimizer()

    def _setup_optimizer(self) -> None:
        """Setup AdamW optimizer with weight decay and linear warmup scheduler."""
        # Separate parameters with and without weight decay
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        # Calculate total training steps
        self.total_steps = len(self.train_dataloader) * self.epochs // self.gradient_accumulation_steps
        self.warmup_steps = int(self.total_steps * self.warmup_ratio)

        # Linear warmup scheduler
        self.scheduler = self._get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )

    def _get_linear_schedule_with_warmup(
        self,
        optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
    ):
        """Create linear warmup scheduler."""
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) /
                float(max(1, num_training_steps - num_warmup_steps))
            )

        return LambdaLR(optimizer, lr_lambda)

    def train(self) -> Dict:
        """Run full training loop.

        Returns:
            Dictionary with training history and best metrics
        """
        print(f"\n{'='*60}")
        print(f"Starting JointBERT Training")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.train_dataloader.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Warmup steps: {self.warmup_steps}")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Early stopping patience: {self.patience}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            # Training phase
            train_metrics = self._train_epoch(epoch)

            # Validation phase
            val_metrics = self._validate()

            # Compute combined metric for early stopping
            combined_metric = (
                val_metrics["intent_accuracy"] +
                val_metrics["slot_f1"]
            ) / 2

            # Log epoch results
            self._log_epoch(epoch, train_metrics, val_metrics, combined_metric)

            # Save history
            self.history.append({
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "combined_metric": combined_metric,
            })

            # Check for improvement
            if combined_metric > self.best_metric:
                self.best_metric = combined_metric
                self.best_epoch = epoch
                self.no_improvement_count = 0
                self._save_checkpoint("best_model.pt", epoch, val_metrics)
                print(f"  ** New best model! Combined metric: {combined_metric:.4f}")
            else:
                self.no_improvement_count += 1
                print(f"  No improvement for {self.no_improvement_count} epoch(s)")

            # Early stopping check
            if self.no_improvement_count >= self.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"  Total time: {total_time/60:.2f} minutes")
        print(f"  Best epoch: {self.best_epoch}")
        print(f"  Best combined metric: {self.best_metric:.4f}")
        print(f"{'='*60}\n")

        # Save training history
        self._save_history()

        return {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "history": self.history,
            "total_time": total_time,
        }

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_intent_loss = 0.0
        total_slot_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}/{self.epochs} [Train]",
            leave=False,
        )

        self.optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            intent_labels = batch["intent_label"].to(self.device)
            slot_labels = batch["slot_labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                intent_labels=intent_labels,
                slot_labels=slot_labels,
            )

            loss = outputs["loss"]

            # Scale loss for gradient accumulation
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Track losses
            total_loss += outputs["loss"].item()
            total_intent_loss += outputs["intent_loss"].item()
            total_slot_loss += outputs["slot_loss"].item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{outputs['loss'].item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

        return {
            "loss": total_loss / num_batches,
            "intent_loss": total_intent_loss / num_batches,
            "slot_loss": total_slot_loss / num_batches,
        }

    def _validate(self) -> Dict[str, float]:
        """Run validation.

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        all_intent_preds = []
        all_intent_labels = []
        all_slot_preds = []
        all_slot_labels = []
        all_attention_masks = []

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.dev_dataloader, desc="Validating", leave=False):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                intent_labels = batch["intent_label"].to(self.device)
                slot_labels = batch["slot_labels"].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    intent_labels=intent_labels,
                    slot_labels=slot_labels,
                )

                total_loss += outputs["loss"].item()
                num_batches += 1

                # Get predictions
                intent_preds = torch.argmax(outputs["intent_logits"], dim=-1)
                slot_preds = torch.argmax(outputs["slot_logits"], dim=-1)

                # Collect predictions
                all_intent_preds.extend(intent_preds.cpu().numpy().tolist())
                all_intent_labels.extend(intent_labels.cpu().numpy().tolist())
                all_slot_preds.extend(slot_preds.cpu().numpy().tolist())
                all_slot_labels.extend(slot_labels.cpu().numpy().tolist())
                all_attention_masks.extend(attention_mask.cpu().numpy().tolist())

        # Compute metrics
        metrics = self._compute_metrics(
            all_intent_preds, all_intent_labels,
            all_slot_preds, all_slot_labels,
            all_attention_masks,
        )
        metrics["loss"] = total_loss / num_batches

        return metrics

    def _compute_metrics(
        self,
        intent_preds: List[int],
        intent_labels: List[int],
        slot_preds: List[List[int]],
        slot_labels: List[List[int]],
        attention_masks: List[List[int]],
    ) -> Dict[str, float]:
        """Compute evaluation metrics.

        Args:
            intent_preds: Predicted intent IDs
            intent_labels: True intent IDs
            slot_preds: Predicted slot ID sequences
            slot_labels: True slot ID sequences
            attention_masks: Attention masks for filtering

        Returns:
            Dictionary with computed metrics
        """
        # Intent accuracy
        intent_correct = sum(
            1 for p, l in zip(intent_preds, intent_labels) if p == l
        )
        intent_accuracy = intent_correct / len(intent_labels)

        # Convert slot IDs to labels for seqeval
        pred_slot_labels = []
        true_slot_labels = []

        for preds, labels, mask in zip(slot_preds, slot_labels, attention_masks):
            pred_seq = []
            true_seq = []

            for p, l, m in zip(preds, labels, mask):
                # Skip padding (mask=0) and ignored positions (label=-100)
                if m == 1 and l != -100:
                    pred_label = self.id2slot.get(p, "O")
                    true_label = self.id2slot.get(l, "O")
                    pred_seq.append(pred_label)
                    true_seq.append(true_label)

            if pred_seq:  # Only add non-empty sequences
                pred_slot_labels.append(pred_seq)
                true_slot_labels.append(true_seq)

        # Slot F1 (entity-level)
        slot_f1 = seqeval_f1_score(true_slot_labels, pred_slot_labels, average="micro")

        # Sentence accuracy (both intent and all slots correct)
        sentence_correct = 0
        for i, (ip, il) in enumerate(zip(intent_preds, intent_labels)):
            if ip == il:
                # Check if all slots are correct
                if i < len(pred_slot_labels) and i < len(true_slot_labels):
                    if pred_slot_labels[i] == true_slot_labels[i]:
                        sentence_correct += 1

        sentence_accuracy = sentence_correct / len(intent_labels)

        return {
            "intent_accuracy": intent_accuracy,
            "slot_f1": slot_f1,
            "sentence_accuracy": sentence_accuracy,
        }

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        combined_metric: float,
    ) -> None:
        """Log epoch results."""
        print(f"\nEpoch {epoch}/{self.epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f} "
              f"(Intent: {train_metrics['intent_loss']:.4f}, "
              f"Slot: {train_metrics['slot_loss']:.4f})")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Intent Accuracy: {val_metrics['intent_accuracy']:.4f}")
        print(f"  Slot F1: {val_metrics['slot_f1']:.4f}")
        print(f"  Sentence Accuracy: {val_metrics['sentence_accuracy']:.4f}")
        print(f"  Combined Metric: {combined_metric:.4f}")

    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Dict,
    ) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "model_config": self.model.get_config() if hasattr(self.model, "get_config") else {},
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")

    def _save_history(self) -> None:
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history: {history_path}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Loaded checkpoint from: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Metrics: {checkpoint.get('metrics', {})}")

        return checkpoint


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    id2intent: Dict[int, str],
    id2slot: Dict[int, str],
    device: torch.device,
) -> Tuple[Dict[str, float], str]:
    """Evaluate model on a dataset.

    Args:
        model: Trained JointBERT model
        dataloader: DataLoader to evaluate on
        id2intent: ID to intent mapping
        id2slot: ID to slot mapping
        device: Device to run on

    Returns:
        Tuple of (metrics dict, classification report string)
    """
    model.eval()
    model.to(device)

    all_intent_preds = []
    all_intent_labels = []
    all_slot_preds = []
    all_slot_labels = []
    all_attention_masks = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_labels = batch["intent_label"].to(device)
            slot_labels = batch["slot_labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            intent_preds = torch.argmax(outputs["intent_logits"], dim=-1)
            slot_preds = torch.argmax(outputs["slot_logits"], dim=-1)

            all_intent_preds.extend(intent_preds.cpu().numpy().tolist())
            all_intent_labels.extend(intent_labels.cpu().numpy().tolist())
            all_slot_preds.extend(slot_preds.cpu().numpy().tolist())
            all_slot_labels.extend(slot_labels.cpu().numpy().tolist())
            all_attention_masks.extend(attention_mask.cpu().numpy().tolist())

    # Compute intent accuracy
    intent_correct = sum(1 for p, l in zip(all_intent_preds, all_intent_labels) if p == l)
    intent_accuracy = intent_correct / len(all_intent_labels)

    # Convert slot predictions to labels
    pred_slot_labels = []
    true_slot_labels = []

    for preds, labels, mask in zip(all_slot_preds, all_slot_labels, all_attention_masks):
        pred_seq = []
        true_seq = []

        for p, l, m in zip(preds, labels, mask):
            if m == 1 and l != -100:
                pred_seq.append(id2slot.get(p, "O"))
                true_seq.append(id2slot.get(l, "O"))

        if pred_seq:
            pred_slot_labels.append(pred_seq)
            true_slot_labels.append(true_seq)

    # Slot F1
    slot_f1 = seqeval_f1_score(true_slot_labels, pred_slot_labels, average="micro")

    # Sentence accuracy
    sentence_correct = 0
    for i, (ip, il) in enumerate(zip(all_intent_preds, all_intent_labels)):
        if ip == il and i < len(pred_slot_labels) and i < len(true_slot_labels):
            if pred_slot_labels[i] == true_slot_labels[i]:
                sentence_correct += 1

    sentence_accuracy = sentence_correct / len(all_intent_labels)

    # Classification report
    report = classification_report(true_slot_labels, pred_slot_labels)

    metrics = {
        "intent_accuracy": intent_accuracy,
        "slot_f1": slot_f1,
        "sentence_accuracy": sentence_accuracy,
    }

    return metrics, report
