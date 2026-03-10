"""Unified evaluation metrics for NLU models.

Provides standardized metrics for comparing SVM, JointBERT, and LLM models:
- Intent accuracy and F1 macro
- Slot F1 (entity-level via seqeval and token-level)
- Sentence accuracy (both intent and all slots correct)
- Per-intent metrics breakdown
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

try:
    from seqeval.metrics import (
        classification_report as seqeval_classification_report,
        f1_score as seqeval_f1,
        precision_score as seqeval_precision,
        recall_score as seqeval_recall,
    )
    SEQEVAL_AVAILABLE = True
except ImportError:
    SEQEVAL_AVAILABLE = False


def intent_accuracy(
    y_true: List[int],
    y_pred: List[int],
) -> float:
    """Calculate intent classification accuracy.

    Args:
        y_true: True intent IDs
        y_pred: Predicted intent IDs

    Returns:
        Accuracy score (0-1)
    """
    return accuracy_score(y_true, y_pred)


def intent_f1_macro(
    y_true: List[int],
    y_pred: List[int],
) -> float:
    """Calculate macro-averaged F1 score for intent classification.

    Args:
        y_true: True intent IDs
        y_pred: Predicted intent IDs

    Returns:
        Macro F1 score (0-1)
    """
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def slot_f1_entity(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    average: str = "micro",
) -> float:
    """Calculate entity-level F1 score for slot filling using seqeval.

    This is the standard slot filling metric that considers entity boundaries.
    An entity is only correct if both the type and boundaries match.

    Args:
        y_true: List of true BIO tag sequences
        y_pred: List of predicted BIO tag sequences
        average: Averaging method ('micro', 'macro', 'weighted')

    Returns:
        Entity-level F1 score (0-1)
    """
    if not SEQEVAL_AVAILABLE:
        # Fallback to token-level if seqeval not available
        return slot_f1_token(y_true, y_pred)

    # Filter out empty sequences
    y_true_filtered = []
    y_pred_filtered = []

    for true_seq, pred_seq in zip(y_true, y_pred):
        # Handle both lists and arrays - check length
        true_len = len(true_seq) if hasattr(true_seq, '__len__') else 0
        pred_len = len(pred_seq) if hasattr(pred_seq, '__len__') else 0

        if true_len > 0 and pred_len > 0:
            # Convert to list if needed and ensure same length
            true_list = list(true_seq) if hasattr(true_seq, '__iter__') else [true_seq]
            pred_list = list(pred_seq) if hasattr(pred_seq, '__iter__') else [pred_seq]
            min_len = min(len(true_list), len(pred_list))
            y_true_filtered.append(true_list[:min_len])
            y_pred_filtered.append(pred_list[:min_len])

    if not y_true_filtered:
        return 0.0

    return seqeval_f1(y_true_filtered, y_pred_filtered, average=average, zero_division=0)


def slot_f1_token(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> float:
    """Calculate token-level F1 score for slot filling.

    This metric treats each token independently and doesn't consider
    entity boundaries. Useful as a simpler baseline metric.

    Args:
        y_true: List of true BIO tag sequences
        y_pred: List of predicted BIO tag sequences

    Returns:
        Token-level F1 score (0-1)
    """
    # Flatten sequences
    true_flat = []
    pred_flat = []

    for true_seq, pred_seq in zip(y_true, y_pred):
        # Convert to list if needed
        true_list = list(true_seq) if hasattr(true_seq, '__iter__') and not isinstance(true_seq, str) else [true_seq]
        pred_list = list(pred_seq) if hasattr(pred_seq, '__iter__') and not isinstance(pred_seq, str) else [pred_seq]

        min_len = min(len(true_list), len(pred_list))
        true_flat.extend(true_list[:min_len])
        pred_flat.extend(pred_list[:min_len])

    if not true_flat:
        return 0.0

    return f1_score(true_flat, pred_flat, average="macro", zero_division=0)


def sentence_accuracy(
    intent_true: List[int],
    intent_pred: List[int],
    slots_true: List[List[str]],
    slots_pred: List[List[str]],
) -> float:
    """Calculate sentence-level accuracy.

    A sentence is correct only if BOTH the intent AND all slots are correct.
    This is a strict metric commonly used in task-oriented dialogue evaluation.

    Args:
        intent_true: True intent IDs
        intent_pred: Predicted intent IDs
        slots_true: True BIO tag sequences
        slots_pred: Predicted BIO tag sequences

    Returns:
        Sentence accuracy (0-1)
    """
    correct = 0
    total = len(intent_true)

    for i, (it, ip, st, sp) in enumerate(zip(
        intent_true, intent_pred, slots_true, slots_pred
    )):
        # Check intent
        if it != ip:
            continue

        # Convert to lists if needed
        st_list = list(st) if hasattr(st, '__iter__') and not isinstance(st, str) else [st]
        sp_list = list(sp) if hasattr(sp, '__iter__') and not isinstance(sp, str) else [sp]

        # Check slots (with length alignment)
        if len(st_list) != len(sp_list):
            continue

        if st_list == sp_list:
            correct += 1

    return correct / total if total > 0 else 0.0


def per_intent_metrics(
    y_true: List[int],
    y_pred: List[int],
    id2intent: Dict[int, str],
) -> Dict[str, Dict[str, float]]:
    """Calculate per-intent precision, recall, and F1.

    Args:
        y_true: True intent IDs
        y_pred: Predicted intent IDs
        id2intent: Mapping from intent ID to intent name

    Returns:
        Dictionary mapping intent names to their metrics
    """
    labels = sorted(set(y_true) | set(y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    results = {}
    for i, label_id in enumerate(labels):
        intent_name = id2intent.get(label_id, f"intent_{label_id}")
        results[intent_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return results


def get_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    labels: Optional[List[int]] = None,
) -> np.ndarray:
    """Get confusion matrix for intent classification.

    Args:
        y_true: True intent IDs
        y_pred: Predicted intent IDs
        labels: Optional list of label IDs to include

    Returns:
        Confusion matrix as numpy array
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    return confusion_matrix(y_true, y_pred, labels=labels)


def compute_all_metrics(
    intent_true: List[int],
    intent_pred: List[int],
    slots_true: List[List[str]],
    slots_pred: List[List[str]],
    id2intent: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Compute all evaluation metrics for NLU.

    This is the main function to use for model evaluation, computing
    all standard metrics in one call.

    Args:
        intent_true: True intent IDs
        intent_pred: Predicted intent IDs
        slots_true: True BIO tag sequences
        slots_pred: Predicted BIO tag sequences
        id2intent: Optional mapping from intent ID to intent name

    Returns:
        Dictionary containing all metrics:
            - intent_accuracy: Intent classification accuracy
            - intent_f1_macro: Macro-averaged intent F1
            - slot_f1_entity: Entity-level slot F1 (seqeval)
            - slot_f1_token: Token-level slot F1
            - sentence_accuracy: Both intent and slots correct
            - per_intent: Per-intent metrics (if id2intent provided)
    """
    results = {
        "intent_accuracy": intent_accuracy(intent_true, intent_pred),
        "intent_f1_macro": intent_f1_macro(intent_true, intent_pred),
        "slot_f1_entity": slot_f1_entity(slots_true, slots_pred),
        "slot_f1_token": slot_f1_token(slots_true, slots_pred),
        "sentence_accuracy": sentence_accuracy(
            intent_true, intent_pred, slots_true, slots_pred
        ),
    }

    if id2intent:
        results["per_intent"] = per_intent_metrics(
            intent_true, intent_pred, id2intent
        )

    return results


def format_metrics_table(
    metrics: Dict[str, Any],
    model_name: str = "Model",
) -> str:
    """Format metrics as a readable table string.

    Args:
        metrics: Dictionary of metrics from compute_all_metrics
        model_name: Name to display for the model

    Returns:
        Formatted table string
    """
    lines = [
        f"\n{'='*60}",
        f"  {model_name} Evaluation Results",
        f"{'='*60}",
        "",
        f"  Intent Accuracy:    {metrics['intent_accuracy']:.4f}",
        f"  Intent F1 (macro):  {metrics['intent_f1_macro']:.4f}",
        f"  Slot F1 (entity):   {metrics['slot_f1_entity']:.4f}",
        f"  Slot F1 (token):    {metrics['slot_f1_token']:.4f}",
        f"  Sentence Accuracy:  {metrics['sentence_accuracy']:.4f}",
        "",
        f"{'='*60}",
    ]

    return "\n".join(lines)


def get_slot_classification_report(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> str:
    """Get detailed slot classification report using seqeval.

    Args:
        y_true: True BIO tag sequences
        y_pred: Predicted BIO tag sequences

    Returns:
        Classification report string
    """
    if not SEQEVAL_AVAILABLE:
        return "seqeval not available - install with: pip install seqeval"

    # Filter empty sequences
    y_true_filtered = []
    y_pred_filtered = []

    for true_seq, pred_seq in zip(y_true, y_pred):
        if true_seq and pred_seq:
            min_len = min(len(true_seq), len(pred_seq))
            y_true_filtered.append(true_seq[:min_len])
            y_pred_filtered.append(pred_seq[:min_len])

    if not y_true_filtered:
        return "No valid sequences to evaluate"

    return seqeval_classification_report(y_true_filtered, y_pred_filtered, zero_division=0)


class NLUEvaluator:
    """Unified evaluator for NLU models.

    Provides a consistent interface for evaluating different NLU model types
    (SVM, JointBERT, LLM) on the same test data.
    """

    def __init__(
        self,
        id2intent: Dict[int, str],
        id2slot: Optional[Dict[int, str]] = None,
    ):
        """Initialize the evaluator.

        Args:
            id2intent: Mapping from intent ID to intent name
            id2slot: Optional mapping from slot ID to slot label
        """
        self.id2intent = id2intent
        self.intent2id = {v: k for k, v in id2intent.items()}
        self.id2slot = id2slot or {}

    def evaluate(
        self,
        intent_true: List[int],
        intent_pred: List[int],
        slots_true: List[List[str]],
        slots_pred: List[List[str]],
    ) -> Dict[str, Any]:
        """Evaluate predictions against ground truth.

        Args:
            intent_true: True intent IDs
            intent_pred: Predicted intent IDs
            slots_true: True BIO tag sequences
            slots_pred: Predicted BIO tag sequences

        Returns:
            Dictionary with all metrics
        """
        return compute_all_metrics(
            intent_true=intent_true,
            intent_pred=intent_pred,
            slots_true=slots_true,
            slots_pred=slots_pred,
            id2intent=self.id2intent,
        )

    def get_confusion_matrix(
        self,
        intent_true: List[int],
        intent_pred: List[int],
        top_k: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Get confusion matrix with intent labels.

        Args:
            intent_true: True intent IDs
            intent_pred: Predicted intent IDs
            top_k: If provided, only include top K most frequent intents

        Returns:
            Tuple of (confusion matrix, list of intent names)
        """
        # Get all labels
        all_labels = sorted(set(intent_true) | set(intent_pred))

        if top_k and top_k < len(all_labels):
            # Find top-k most frequent intents in true labels
            from collections import Counter
            label_counts = Counter(intent_true)
            top_labels = [label for label, _ in label_counts.most_common(top_k)]
            all_labels = sorted(top_labels)

        cm = get_confusion_matrix(intent_true, intent_pred, labels=all_labels)
        label_names = [self.id2intent.get(i, f"intent_{i}") for i in all_labels]

        return cm, label_names

    def get_misclassified_examples(
        self,
        texts: List[str],
        intent_true: List[int],
        intent_pred: List[int],
        slots_true: List[List[str]],
        slots_pred: List[List[str]],
        max_examples: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get examples where the model made errors.

        Args:
            texts: Input texts
            intent_true: True intent IDs
            intent_pred: Predicted intent IDs
            slots_true: True BIO tag sequences
            slots_pred: Predicted BIO tag sequences
            max_examples: Maximum number of examples to return

        Returns:
            List of error examples with details
        """
        errors = []

        for i, (text, it, ip, st, sp) in enumerate(zip(
            texts, intent_true, intent_pred, slots_true, slots_pred
        )):
            intent_error = it != ip

            # Check slot errors
            min_len = min(len(st), len(sp))
            slot_error = len(st) != len(sp) or st[:min_len] != sp[:min_len]

            if intent_error or slot_error:
                error_types = []
                if intent_error:
                    error_types.append("intent")
                if slot_error:
                    error_types.append("slot")

                errors.append({
                    "index": i,
                    "text": text,
                    "true_intent": self.id2intent.get(it, f"intent_{it}"),
                    "pred_intent": self.id2intent.get(ip, f"intent_{ip}"),
                    "true_slots": st,
                    "pred_slots": sp,
                    "error_types": error_types,
                })

                if len(errors) >= max_examples:
                    break

        return errors
