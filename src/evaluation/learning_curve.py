"""Learning curve analysis for NLU models.

Evaluates model performance with different training data sizes:
- 20%, 40%, 60%, 80%, 100% of training data
- Plots learning curves for all three models
"""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


# Default data fractions for learning curve
DEFAULT_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]


class LearningCurveEvaluator:
    """Evaluates models at different training data sizes.

    Trains models on subsets of training data and evaluates on full test set
    to understand how performance scales with data size.
    """

    def __init__(
        self,
        fractions: List[float] = None,
        seed: int = 42,
    ):
        """Initialize the evaluator.

        Args:
            fractions: Data fractions to evaluate (default: 0.2, 0.4, 0.6, 0.8, 1.0)
            seed: Random seed for reproducibility
        """
        self.fractions = fractions or DEFAULT_FRACTIONS
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def sample_data(
        self,
        texts: List[str],
        intent_ids: List[int],
        slot_labels: List[List[str]],
        fraction: float,
    ) -> Tuple[List[str], List[int], List[List[str]]]:
        """Sample a fraction of the training data.

        Args:
            texts: Training texts
            intent_ids: Training intent IDs
            slot_labels: Training slot labels
            fraction: Fraction of data to use (0-1)

        Returns:
            Tuple of (sampled_texts, sampled_intents, sampled_slots)
        """
        n_samples = int(len(texts) * fraction)
        indices = self.rng.choice(len(texts), size=n_samples, replace=False)
        indices = sorted(indices)

        return (
            [texts[i] for i in indices],
            [intent_ids[i] for i in indices],
            [slot_labels[i] for i in indices],
        )


def train_and_evaluate_svm(
    train_texts: List[str],
    train_intents: List[int],
    train_slots: List[List[str]],
    test_texts: List[str],
    test_intents: List[int],
    test_slots: List[List[str]],
    id2intent: Dict[int, str],
) -> Dict[str, float]:
    """Train SVM model and evaluate.

    Returns metrics dictionary.
    """
    from src.nlu.svm_nlu import SVMNLU
    from src.evaluation.metrics import compute_all_metrics

    # Train
    start_time = time.time()
    nlu = SVMNLU()
    nlu.fit(train_texts, train_intents, train_slots, id2intent)
    train_time = time.time() - start_time

    # Evaluate
    start_time = time.time()
    results = []
    intent_preds = []
    slot_preds = []

    for text in test_texts:
        pred = nlu.predict(text)
        intent_preds.append(nlu.intent_classifier.intent2id.get(pred["intent"], 0))
        slot_preds.append(pred["slot_labels"])

    inference_time = time.time() - start_time

    # Calculate metrics
    metrics = compute_all_metrics(
        intent_true=test_intents,
        intent_pred=intent_preds,
        slots_true=test_slots,
        slots_pred=slot_preds,
    )

    metrics["train_time"] = train_time
    metrics["inference_time"] = inference_time

    return metrics


def evaluate_svm_learning_curve(
    train_texts: List[str],
    train_intents: List[int],
    train_slots: List[List[str]],
    test_texts: List[str],
    test_intents: List[int],
    test_slots: List[List[str]],
    id2intent: Dict[int, str],
    fractions: List[float] = None,
    seed: int = 42,
) -> Dict[str, List]:
    """Evaluate SVM at different training data sizes.

    Args:
        train_texts: Full training texts
        train_intents: Full training intent IDs
        train_slots: Full training slot labels
        test_texts: Test texts
        test_intents: Test intent IDs
        test_slots: Test slot labels
        id2intent: Intent ID to name mapping
        fractions: Data fractions to evaluate
        seed: Random seed

    Returns:
        Dictionary with lists of metrics at each fraction
    """
    fractions = fractions or DEFAULT_FRACTIONS
    evaluator = LearningCurveEvaluator(fractions=fractions, seed=seed)

    results = {
        "fractions": [],
        "n_samples": [],
        "intent_accuracy": [],
        "slot_f1_entity": [],
        "sentence_accuracy": [],
        "train_time": [],
    }

    for fraction in fractions:
        print(f"  SVM @ {fraction*100:.0f}%...", end=" ", flush=True)

        # Sample training data
        sampled_texts, sampled_intents, sampled_slots = evaluator.sample_data(
            train_texts, train_intents, train_slots, fraction
        )

        # Train and evaluate
        metrics = train_and_evaluate_svm(
            train_texts=sampled_texts,
            train_intents=sampled_intents,
            train_slots=sampled_slots,
            test_texts=test_texts,
            test_intents=test_intents,
            test_slots=test_slots,
            id2intent=id2intent,
        )

        results["fractions"].append(fraction)
        results["n_samples"].append(len(sampled_texts))
        results["intent_accuracy"].append(metrics["intent_accuracy"])
        results["slot_f1_entity"].append(metrics["slot_f1_entity"])
        results["sentence_accuracy"].append(metrics["sentence_accuracy"])
        results["train_time"].append(metrics["train_time"])

        print(f"acc={metrics['intent_accuracy']:.3f}")

    return results


def get_mock_bert_learning_curve(
    fractions: List[float] = None,
) -> Dict[str, List]:
    """Get mock learning curve results for JointBERT.

    Used when actual training is too expensive or not available.
    Based on typical PhoBERT performance characteristics.
    """
    fractions = fractions or DEFAULT_FRACTIONS

    # Simulated metrics based on typical transformer learning curves
    # JointBERT typically shows logarithmic improvement with data
    base_intent_acc = 0.97
    base_slot_f1 = 0.95
    base_sentence_acc = 0.85

    results = {
        "fractions": fractions,
        "n_samples": [int(4478 * f) for f in fractions],  # Assuming ~4478 train samples
        "intent_accuracy": [],
        "slot_f1_entity": [],
        "sentence_accuracy": [],
        "train_time": [],
    }

    for fraction in fractions:
        # Logarithmic scaling: performance approaches asymptote
        scale = np.log(1 + 4 * fraction) / np.log(5)

        results["intent_accuracy"].append(0.80 + (base_intent_acc - 0.80) * scale)
        results["slot_f1_entity"].append(0.70 + (base_slot_f1 - 0.70) * scale)
        results["sentence_accuracy"].append(0.55 + (base_sentence_acc - 0.55) * scale)
        results["train_time"].append(300 * fraction)  # ~5 min per 100%

    return results


def get_mock_llm_learning_curve(
    fractions: List[float] = None,
) -> Dict[str, List]:
    """Get mock learning curve results for LLM (zero-shot).

    LLM is zero-shot, so performance doesn't change with training data.
    """
    fractions = fractions or DEFAULT_FRACTIONS

    # Zero-shot LLM performance is constant
    intent_acc = 0.75
    slot_f1 = 0.65
    sentence_acc = 0.45

    n_fractions = len(fractions)

    return {
        "fractions": fractions,
        "n_samples": [int(4478 * f) for f in fractions],
        "intent_accuracy": [intent_acc] * n_fractions,
        "slot_f1_entity": [slot_f1] * n_fractions,
        "sentence_accuracy": [sentence_acc] * n_fractions,
        "train_time": [0.0] * n_fractions,  # No training for zero-shot
    }


def plot_learning_curves(
    svm_results: Optional[Dict[str, List]] = None,
    bert_results: Optional[Dict[str, List]] = None,
    llm_results: Optional[Dict[str, List]] = None,
    output_path: str = "results/figures/learning_curve.png",
    figsize: Tuple[int, int] = (14, 5),
) -> Optional[plt.Figure]:
    """Plot learning curves for all models.

    Args:
        svm_results: SVM learning curve results
        bert_results: JointBERT learning curve results
        llm_results: LLM learning curve results
        output_path: Path to save the figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib not available")
        return None

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    metrics = [
        ("intent_accuracy", "Intent Accuracy"),
        ("slot_f1_entity", "Slot F1 (Entity)"),
        ("sentence_accuracy", "Sentence Accuracy"),
    ]

    colors = {"SVM": "#2ecc71", "JointBERT": "#3498db", "LLM": "#9b59b6"}
    markers = {"SVM": "o", "JointBERT": "s", "LLM": "^"}

    for ax, (metric_key, metric_name) in zip(axes, metrics):
        if svm_results:
            fractions = [f * 100 for f in svm_results["fractions"]]
            ax.plot(
                fractions,
                svm_results[metric_key],
                marker=markers["SVM"],
                color=colors["SVM"],
                linewidth=2,
                markersize=8,
                label="SVM",
            )

        if bert_results:
            fractions = [f * 100 for f in bert_results["fractions"]]
            ax.plot(
                fractions,
                bert_results[metric_key],
                marker=markers["JointBERT"],
                color=colors["JointBERT"],
                linewidth=2,
                markersize=8,
                label="JointBERT",
            )

        if llm_results:
            fractions = [f * 100 for f in llm_results["fractions"]]
            ax.plot(
                fractions,
                llm_results[metric_key],
                marker=markers["LLM"],
                color=colors["LLM"],
                linewidth=2,
                markersize=8,
                linestyle="--",
                label="LLM (zero-shot)",
            )

        ax.set_xlabel("Training Data (%)", fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight="bold")
        ax.set_xticks([20, 40, 60, 80, 100])
        ax.set_ylim(0.3, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)

    plt.suptitle("Learning Curves: Performance vs Training Data Size", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved learning curve: {output_path}")

    return fig


def save_learning_curve_data(
    svm_results: Optional[Dict] = None,
    bert_results: Optional[Dict] = None,
    llm_results: Optional[Dict] = None,
    output_path: str = "results/tables/learning_curve_data.json",
) -> str:
    """Save learning curve data to JSON.

    Args:
        svm_results: SVM learning curve results
        bert_results: JointBERT learning curve results
        llm_results: LLM learning curve results
        output_path: Path to save the data

    Returns:
        Path to saved file
    """
    data = {}

    if svm_results:
        data["svm"] = svm_results
    if bert_results:
        data["bert"] = bert_results
    if llm_results:
        data["llm"] = llm_results

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved learning curve data: {output_path}")

    return str(output_path)


def generate_learning_curve_report(
    svm_results: Optional[Dict] = None,
    bert_results: Optional[Dict] = None,
    llm_results: Optional[Dict] = None,
) -> str:
    """Generate markdown table of learning curve results.

    Returns:
        Markdown-formatted table string
    """
    lines = [
        "## Learning Curve Results",
        "",
        "Performance at different training data sizes:",
        "",
        "### Intent Accuracy",
        "",
        "| Data % | SVM | JointBERT | LLM |",
        "|--------|-----|-----------|-----|",
    ]

    fractions = svm_results["fractions"] if svm_results else DEFAULT_FRACTIONS

    for i, frac in enumerate(fractions):
        svm_val = f"{svm_results['intent_accuracy'][i]:.3f}" if svm_results else "-"
        bert_val = f"{bert_results['intent_accuracy'][i]:.3f}" if bert_results else "-"
        llm_val = f"{llm_results['intent_accuracy'][i]:.3f}" if llm_results else "-"

        lines.append(f"| {frac*100:.0f}% | {svm_val} | {bert_val} | {llm_val} |")

    lines.extend([
        "",
        "### Slot F1 (Entity-level)",
        "",
        "| Data % | SVM | JointBERT | LLM |",
        "|--------|-----|-----------|-----|",
    ])

    for i, frac in enumerate(fractions):
        svm_val = f"{svm_results['slot_f1_entity'][i]:.3f}" if svm_results else "-"
        bert_val = f"{bert_results['slot_f1_entity'][i]:.3f}" if bert_results else "-"
        llm_val = f"{llm_results['slot_f1_entity'][i]:.3f}" if llm_results else "-"

        lines.append(f"| {frac*100:.0f}% | {svm_val} | {bert_val} | {llm_val} |")

    return "\n".join(lines)
