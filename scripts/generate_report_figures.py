#!/usr/bin/env python3
"""Generate publication-quality figures for the NLU report.

Creates:
- Model comparison bar charts (intent accuracy, slot F1)
- Training time comparison
- Confusion matrices (side-by-side)
- Learning curves
- Per-intent performance breakdown

Usage:
    python scripts/generate_report_figures.py
    python scripts/generate_report_figures.py --results-dir results/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    import seaborn as sns

    # Set publication-quality defaults
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available")


# Color palette for models
MODEL_COLORS = {
    "SVM": "#2ecc71",      # Green
    "JointBERT": "#3498db", # Blue
    "LLM": "#9b59b6",      # Purple
}

MODEL_HATCHES = {
    "SVM": "",
    "JointBERT": "//",
    "LLM": "\\\\",
}


def load_comparison_results(results_dir: Path) -> Optional[pd.DataFrame]:
    """Load comparison.csv if available."""
    csv_path = results_dir / "tables" / "comparison.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def create_mock_results() -> Dict[str, Dict[str, float]]:
    """Create mock results for demonstration."""
    return {
        "SVM": {
            "intent_accuracy": 0.942,
            "intent_f1_macro": 0.891,
            "slot_f1_entity": 0.923,
            "slot_f1_token": 0.935,
            "sentence_accuracy": 0.823,
            "inference_time": 2.5,
            "train_time": 15.0,
        },
        "JointBERT": {
            "intent_accuracy": 0.971,
            "intent_f1_macro": 0.943,
            "slot_f1_entity": 0.952,
            "slot_f1_token": 0.961,
            "sentence_accuracy": 0.891,
            "inference_time": 45.0,
            "train_time": 300.0,
        },
        "LLM": {
            "intent_accuracy": 0.754,
            "intent_f1_macro": 0.712,
            "slot_f1_entity": 0.651,
            "slot_f1_token": 0.673,
            "sentence_accuracy": 0.456,
            "inference_time": 120.0,
            "train_time": 0.0,
        },
    }


def plot_model_comparison_bars(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plot bar chart comparing model performance metrics.

    Args:
        results: Model results dictionary
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if not PLOTTING_AVAILABLE:
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.25

    # Left plot: Accuracy metrics
    ax1 = axes[0]
    metrics = ["intent_accuracy", "slot_f1_entity", "sentence_accuracy"]
    labels = ["Intent Acc", "Slot F1", "Sentence Acc"]

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        values = [results[m][metric] for m in models]
        bars = ax1.bar(
            x + i * width - width,
            values,
            width,
            label=label,
            color=plt.cm.Blues(0.4 + i * 0.2),
            edgecolor='black',
            linewidth=0.5,
        )
        # Add value labels
        for bar, val in zip(bars, values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.2f}',
                ha='center',
                va='bottom',
                fontsize=8,
            )

    ax1.set_xlabel('Model')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3)

    # Right plot: Per-model comparison
    ax2 = axes[1]

    for i, model in enumerate(models):
        metrics_vals = [
            results[model]["intent_accuracy"],
            results[model]["slot_f1_entity"],
            results[model]["sentence_accuracy"],
        ]
        ax2.bar(
            x + i * width - width,
            metrics_vals,
            width,
            label=model,
            color=MODEL_COLORS.get(model, f'C{i}'),
            edgecolor='black',
            linewidth=0.5,
        )

    ax2.set_xlabel('Metric')
    ax2.set_ylabel('Score')
    ax2.set_title('Model Comparison by Metric', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Intent Acc", "Slot F1", "Sentence Acc"])
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='lower right')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved: {output_path}")

    return fig


def plot_training_time_comparison(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """Plot training and inference time comparison.

    Args:
        results: Model results dictionary
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if not PLOTTING_AVAILABLE:
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    models = list(results.keys())

    # Training time
    ax1 = axes[0]
    train_times = [results[m].get("train_time", 0) / 60 for m in models]  # Convert to minutes
    colors = [MODEL_COLORS.get(m, 'gray') for m in models]

    bars = ax1.bar(models, train_times, color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, train_times):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{val:.1f}m',
            ha='center',
            va='bottom',
            fontsize=10,
        )

    ax1.set_xlabel('Model')
    ax1.set_ylabel('Training Time (minutes)')
    ax1.set_title('Training Time', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Inference time
    ax2 = axes[1]
    inference_times = [results[m].get("inference_time", 0) for m in models]

    bars = ax2.bar(models, inference_times, color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, inference_times):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{val:.1f}s',
            ha='center',
            va='bottom',
            fontsize=10,
        )

    ax2.set_xlabel('Model')
    ax2.set_ylabel('Inference Time (seconds)')
    ax2.set_title('Test Set Inference Time', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved: {output_path}")

    return fig


def plot_radar_chart(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """Plot radar/spider chart comparing models across metrics.

    Args:
        results: Model results dictionary
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if not PLOTTING_AVAILABLE:
        return None

    categories = ['Intent Acc', 'Intent F1', 'Slot F1', 'Token F1', 'Sent Acc']
    metric_keys = ['intent_accuracy', 'intent_f1_macro', 'slot_f1_entity', 'slot_f1_token', 'sentence_accuracy']

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Number of variables
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    for model, data in results.items():
        values = [data[key] for key in metric_keys]
        values += values[:1]  # Complete the loop

        ax.plot(
            angles,
            values,
            'o-',
            linewidth=2,
            label=model,
            color=MODEL_COLORS.get(model, 'gray'),
        )
        ax.fill(
            angles,
            values,
            alpha=0.15,
            color=MODEL_COLORS.get(model, 'gray'),
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Model Performance Comparison', fontweight='bold', pad=20)

    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved: {output_path}")

    return fig


def plot_per_intent_heatmap(
    per_intent_path: str,
    output_path: str,
    top_k: int = 15,
    figsize: Tuple[int, int] = (12, 8),
) -> Optional[plt.Figure]:
    """Plot heatmap of per-intent F1 scores.

    Args:
        per_intent_path: Path to per_intent_metrics.csv
        output_path: Path to save figure
        top_k: Number of intents to show
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if not PLOTTING_AVAILABLE:
        return None

    if not Path(per_intent_path).exists():
        print(f"Warning: {per_intent_path} not found. Skipping heatmap.")
        return None

    df = pd.read_csv(per_intent_path)

    # Get F1 columns
    f1_cols = [col for col in df.columns if 'F1' in col]
    if not f1_cols:
        print("No F1 columns found in per-intent metrics")
        return None

    # Convert F1 values to numeric
    for col in f1_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sort by average F1 and get top-k
    df['avg_f1'] = df[f1_cols].mean(axis=1)
    df_sorted = df.nsmallest(top_k, 'avg_f1')  # Show worst-performing intents

    # Create heatmap data
    heatmap_data = df_sorted.set_index('Intent')[f1_cols]

    # Rename columns for display
    heatmap_data.columns = [col.replace(' F1', '') for col in heatmap_data.columns]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        ax=ax,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={'label': 'F1 Score'},
    )

    ax.set_title(f'Per-Intent F1 Scores (Bottom {top_k} Intents)', fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('Intent')

    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved: {output_path}")

    return fig


def generate_summary_table(
    results: Dict[str, Dict[str, float]],
    output_path: str,
) -> str:
    """Generate LaTeX/Markdown summary table.

    Args:
        results: Model results dictionary
        output_path: Path to save table

    Returns:
        Table as string
    """
    # Markdown table
    lines = [
        "# Model Comparison Summary",
        "",
        "| Model | Intent Acc | Intent F1 | Slot F1 | Sent Acc | Train Time | Inference |",
        "|-------|-----------|-----------|---------|----------|------------|-----------|",
    ]

    for model, data in results.items():
        train_time = f"{data.get('train_time', 0)/60:.1f}m"
        if data.get('train_time', 0) == 0:
            train_time = "N/A"

        lines.append(
            f"| {model} | "
            f"{data['intent_accuracy']:.3f} | "
            f"{data['intent_f1_macro']:.3f} | "
            f"{data['slot_f1_entity']:.3f} | "
            f"{data['sentence_accuracy']:.3f} | "
            f"{train_time} | "
            f"{data.get('inference_time', 0):.1f}s |"
        )

    table_str = "\n".join(lines)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(table_str)

    print(f"Saved: {output_path}")

    return table_str


def main():
    parser = argparse.ArgumentParser(description="Generate report figures")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results directory",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock results if no data available",
    )

    args = parser.parse_args()

    results_dir = PROJECT_ROOT / args.results_dir
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Generating Report Figures")
    print("=" * 60)

    # Try to load results from CSV
    df = load_comparison_results(results_dir)

    if df is not None:
        # Convert DataFrame to results dict
        results = {}
        for _, row in df.iterrows():
            model = row['Model']
            results[model] = {
                'intent_accuracy': float(row['Intent Accuracy']),
                'intent_f1_macro': float(row['Intent F1 (macro)']),
                'slot_f1_entity': float(row['Slot F1 (entity)']),
                'slot_f1_token': float(row['Slot F1 (token)']),
                'sentence_accuracy': float(row['Sentence Accuracy']),
                'inference_time': float(row.get('Inference Time (s)', '0').replace('s', '')),
                'train_time': 0,  # Not in CSV
            }
        print("  Loaded results from comparison.csv")
    else:
        print("  Using mock results (run evaluate_all.py first for real data)")
        results = create_mock_results()

    # Generate figures
    print("\n  Generating figures...")

    # Model comparison bars
    plot_model_comparison_bars(
        results,
        str(figures_dir / "model_comparison.png"),
    )

    # Training time comparison
    plot_training_time_comparison(
        results,
        str(figures_dir / "time_comparison.png"),
    )

    # Radar chart
    plot_radar_chart(
        results,
        str(figures_dir / "radar_comparison.png"),
    )

    # Per-intent heatmap (if available)
    per_intent_path = tables_dir / "per_intent_metrics.csv"
    if per_intent_path.exists():
        plot_per_intent_heatmap(
            str(per_intent_path),
            str(figures_dir / "per_intent_heatmap.png"),
        )

    # Learning curve (check if data exists)
    lc_data_path = tables_dir / "learning_curve_data.json"
    if lc_data_path.exists():
        with open(lc_data_path) as f:
            lc_data = json.load(f)

        from src.evaluation.learning_curve import plot_learning_curves
        plot_learning_curves(
            svm_results=lc_data.get("svm"),
            bert_results=lc_data.get("bert"),
            llm_results=lc_data.get("llm"),
            output_path=str(figures_dir / "learning_curve.png"),
        )
    else:
        # Generate mock learning curve
        print("  Generating mock learning curve...")
        from src.evaluation.learning_curve import (
            get_mock_bert_learning_curve,
            get_mock_llm_learning_curve,
            plot_learning_curves,
        )

        # Simple mock SVM curve
        svm_lc = {
            "fractions": [0.2, 0.4, 0.6, 0.8, 1.0],
            "n_samples": [896, 1791, 2687, 3582, 4478],
            "intent_accuracy": [0.85, 0.90, 0.92, 0.93, 0.94],
            "slot_f1_entity": [0.82, 0.87, 0.90, 0.91, 0.92],
            "sentence_accuracy": [0.68, 0.75, 0.78, 0.80, 0.82],
            "train_time": [3, 6, 9, 12, 15],
        }

        plot_learning_curves(
            svm_results=svm_lc,
            bert_results=get_mock_bert_learning_curve(),
            llm_results=get_mock_llm_learning_curve(),
            output_path=str(figures_dir / "learning_curve.png"),
        )

    # Summary table
    generate_summary_table(
        results,
        str(tables_dir / "summary_table.md"),
    )

    print("\n" + "=" * 60)
    print("  Figure Generation Complete!")
    print("=" * 60)
    print(f"\nGenerated files in {figures_dir}:")
    for f in figures_dir.glob("*.png"):
        print(f"  - {f.name}")
    print("")


if __name__ == "__main__":
    main()
