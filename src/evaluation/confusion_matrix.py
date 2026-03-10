"""Confusion matrix visualization for NLU intent classification.

Generates publication-quality confusion matrices for comparing
different NLU models (SVM, JointBERT, LLM).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str = "Intent Confusion Matrix",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "Blues",
    normalize: bool = True,
    annot_fontsize: int = 8,
    show_values: bool = True,
) -> Optional[plt.Figure]:
    """Plot a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array (n_classes x n_classes)
        labels: List of class labels
        title: Plot title
        output_path: Path to save the figure (optional)
        figsize: Figure size as (width, height)
        cmap: Colormap name
        normalize: Whether to normalize by row (true labels)
        annot_fontsize: Font size for cell annotations
        show_values: Whether to show values in cells

    Returns:
        matplotlib Figure object, or None if plotting not available
    """
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")
        return None

    # Normalize if requested
    if normalize:
        cm_plot = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        fmt = '.2f'
        vmin, vmax = 0, 1
    else:
        cm_plot = cm
        fmt = 'd'
        vmin, vmax = None, None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        cm_plot,
        annot=show_values,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        annot_kws={'fontsize': annot_fontsize},
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
    )

    # Labels
    ax.set_xlabel('Predicted Intent', fontsize=12)
    ax.set_ylabel('True Intent', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix: {output_path}")

    return fig


def plot_top_k_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    id2intent: Dict[int, str],
    top_k: int = 15,
    title: str = "Intent Confusion Matrix (Top 15)",
    output_path: Optional[str] = None,
    **kwargs,
) -> Optional[plt.Figure]:
    """Plot confusion matrix for top-k most frequent intents.

    Args:
        y_true: True intent IDs
        y_pred: Predicted intent IDs
        id2intent: Mapping from intent ID to intent name
        top_k: Number of top intents to include
        title: Plot title
        output_path: Path to save the figure
        **kwargs: Additional arguments passed to plot_confusion_matrix

    Returns:
        matplotlib Figure object
    """
    from collections import Counter
    from sklearn.metrics import confusion_matrix

    # Find top-k most frequent intents
    label_counts = Counter(y_true)
    top_labels = [label for label, _ in label_counts.most_common(top_k)]
    top_labels_set = set(top_labels)

    # Filter data to only include top-k intents
    filtered_true = []
    filtered_pred = []

    for t, p in zip(y_true, y_pred):
        if t in top_labels_set:
            filtered_true.append(t)
            # If prediction is not in top-k, map to "other"
            if p in top_labels_set:
                filtered_pred.append(p)
            else:
                filtered_pred.append(t)  # Keep as true for confusion matrix

    # Get confusion matrix for filtered data
    cm = confusion_matrix(filtered_true, filtered_pred, labels=top_labels)

    # Get label names
    label_names = [id2intent.get(i, f"intent_{i}") for i in top_labels]

    # Truncate long label names
    label_names = [name[:20] + "..." if len(name) > 20 else name for name in label_names]

    return plot_confusion_matrix(
        cm=cm,
        labels=label_names,
        title=title,
        output_path=output_path,
        **kwargs,
    )


def plot_comparison_confusion_matrices(
    results: Dict[str, Dict],
    id2intent: Dict[int, str],
    top_k: int = 10,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 5),
) -> Optional[plt.Figure]:
    """Plot confusion matrices for multiple models side by side.

    Args:
        results: Dictionary mapping model name to {'y_true': [...], 'y_pred': [...]}
        id2intent: Mapping from intent ID to intent name
        top_k: Number of top intents to include
        output_path: Path to save the figure
        figsize: Figure size as (width, height)

    Returns:
        matplotlib Figure object
    """
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib/seaborn not available")
        return None

    from collections import Counter
    from sklearn.metrics import confusion_matrix

    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    # Find top-k intents across all models (use first model's true labels)
    first_model = list(results.values())[0]
    label_counts = Counter(first_model['y_true'])
    top_labels = [label for label, _ in label_counts.most_common(top_k)]
    label_names = [id2intent.get(i, f"intent_{i}")[:15] for i in top_labels]

    for ax, (model_name, data) in zip(axes, results.items()):
        y_true = data['y_true']
        y_pred = data['y_pred']

        # Filter to top-k
        filtered_true = []
        filtered_pred = []
        top_labels_set = set(top_labels)

        for t, p in zip(y_true, y_pred):
            if t in top_labels_set:
                filtered_true.append(t)
                filtered_pred.append(p if p in top_labels_set else t)

        cm = confusion_matrix(filtered_true, filtered_pred, labels=top_labels)

        # Normalize
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=label_names,
            yticklabels=label_names,
            ax=ax,
            vmin=0,
            vmax=1,
            annot_kws={'fontsize': 7},
            cbar=False,
        )

        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison confusion matrices: {output_path}")

    return fig


def generate_all_confusion_matrices(
    svm_results: Optional[Dict] = None,
    bert_results: Optional[Dict] = None,
    llm_results: Optional[Dict] = None,
    id2intent: Dict[int, str] = None,
    output_dir: str = "results/figures",
    top_k: int = 15,
) -> Dict[str, str]:
    """Generate confusion matrices for all models.

    Args:
        svm_results: SVM results with 'y_true' and 'y_pred'
        bert_results: JointBERT results with 'y_true' and 'y_pred'
        llm_results: LLM results with 'y_true' and 'y_pred'
        id2intent: Mapping from intent ID to intent name
        output_dir: Directory to save figures
        top_k: Number of top intents to include

    Returns:
        Dictionary mapping model name to output file path
    """
    output_paths = {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Individual confusion matrices
    if svm_results and id2intent:
        path = str(output_dir / "intent_confusion_svm.png")
        plot_top_k_confusion_matrix(
            y_true=svm_results['y_true'],
            y_pred=svm_results['y_pred'],
            id2intent=id2intent,
            top_k=top_k,
            title="SVM Intent Confusion Matrix (Top 15)",
            output_path=path,
        )
        output_paths['svm'] = path
        plt.close()

    if bert_results and id2intent:
        path = str(output_dir / "intent_confusion_bert.png")
        plot_top_k_confusion_matrix(
            y_true=bert_results['y_true'],
            y_pred=bert_results['y_pred'],
            id2intent=id2intent,
            top_k=top_k,
            title="JointBERT Intent Confusion Matrix (Top 15)",
            output_path=path,
        )
        output_paths['bert'] = path
        plt.close()

    if llm_results and id2intent:
        path = str(output_dir / "intent_confusion_llm.png")
        plot_top_k_confusion_matrix(
            y_true=llm_results['y_true'],
            y_pred=llm_results['y_pred'],
            id2intent=id2intent,
            top_k=top_k,
            title="LLM Intent Confusion Matrix (Top 15)",
            output_path=path,
        )
        output_paths['llm'] = path
        plt.close()

    # Comparison figure (if multiple models)
    available_results = {}
    if svm_results:
        available_results['SVM'] = svm_results
    if bert_results:
        available_results['JointBERT'] = bert_results
    if llm_results:
        available_results['LLM'] = llm_results

    if len(available_results) > 1 and id2intent:
        path = str(output_dir / "intent_confusion_comparison.png")
        plot_comparison_confusion_matrices(
            results=available_results,
            id2intent=id2intent,
            top_k=min(10, top_k),
            output_path=path,
        )
        output_paths['comparison'] = path
        plt.close()

    return output_paths


def analyze_confusion_patterns(
    cm: np.ndarray,
    labels: List[str],
    threshold: float = 0.1,
) -> List[Dict]:
    """Analyze common confusion patterns in the confusion matrix.

    Args:
        cm: Confusion matrix array
        labels: List of class labels
        threshold: Minimum proportion to consider as a pattern

    Returns:
        List of confusion patterns with details
    """
    patterns = []

    # Normalize by row
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype('float') / (row_sums + 1e-10)

    n_classes = len(labels)

    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm_norm[i, j] >= threshold:
                patterns.append({
                    'true_class': labels[i],
                    'pred_class': labels[j],
                    'confusion_rate': float(cm_norm[i, j]),
                    'count': int(cm[i, j]),
                    'true_class_total': int(row_sums[i, 0]),
                })

    # Sort by confusion rate
    patterns.sort(key=lambda x: x['confusion_rate'], reverse=True)

    return patterns


def format_confusion_analysis(
    patterns: List[Dict],
    top_k: int = 10,
) -> str:
    """Format confusion analysis as a readable report.

    Args:
        patterns: List of confusion patterns from analyze_confusion_patterns
        top_k: Number of top patterns to show

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "  Top Confusion Patterns",
        "=" * 60,
        "",
    ]

    for i, pattern in enumerate(patterns[:top_k], 1):
        lines.append(
            f"  {i}. {pattern['true_class']} -> {pattern['pred_class']}"
        )
        lines.append(
            f"     Rate: {pattern['confusion_rate']:.2%} "
            f"({pattern['count']}/{pattern['true_class_total']})"
        )
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)
