"""Error analysis for NLU model evaluation.

Categorizes and analyzes errors to understand model weaknesses:
- Ambiguous intent: Semantically similar intents confused
- Rare intent: Low-frequency intents with poor performance
- Slot boundary: Entity boundary detection errors
- OOV (Out of Vocabulary): Errors on unseen words/patterns
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import json


# Define semantically related intent groups for ambiguity detection
AMBIGUOUS_INTENT_GROUPS = [
    {"flight", "airfare", "flight_time"},
    {"airline", "airline#flight_no"},
    {"flight#flight_no", "flight_no"},
    {"ground_fare", "ground_service", "ground_fare#ground_service"},
    {"city", "city#flight_time"},
    {"aircraft", "aircraft#flight#flight_no"},
    {"airfare#flight", "airfare"},
]


class ErrorAnalyzer:
    """Analyzes NLU model errors and categorizes them.

    Error categories:
    - ambiguous_intent: Confusion between semantically similar intents
    - rare_intent: Errors on low-frequency intents (< 50 samples)
    - slot_boundary: Wrong entity boundaries (B-/I- tag errors)
    - oov_related: Potential OOV-related errors (rare words in text)
    - other: Uncategorized errors
    """

    def __init__(
        self,
        id2intent: Dict[int, str],
        intent2id: Optional[Dict[str, int]] = None,
        id2slot: Optional[Dict[int, str]] = None,
        rare_threshold: int = 50,
        train_texts: Optional[List[str]] = None,
    ):
        """Initialize the error analyzer.

        Args:
            id2intent: Mapping from intent ID to intent name
            intent2id: Mapping from intent name to ID (derived if not provided)
            id2slot: Mapping from slot ID to slot label
            rare_threshold: Intents with fewer samples are considered rare
            train_texts: Training texts for building vocabulary (OOV detection)
        """
        self.id2intent = id2intent
        self.intent2id = intent2id or {v: k for k, v in id2intent.items()}
        self.id2slot = id2slot or {}
        self.rare_threshold = rare_threshold

        # Build ambiguity map
        self._ambiguous_map = self._build_ambiguity_map()

        # Build vocabulary from training data
        self._vocab: Set[str] = set()
        if train_texts:
            for text in train_texts:
                self._vocab.update(text.lower().split())

        # Store intent frequencies (set during analyze)
        self._intent_counts: Dict[int, int] = {}

    def _build_ambiguity_map(self) -> Dict[str, Set[str]]:
        """Build mapping from intent to its ambiguous counterparts."""
        ambiguous_map = defaultdict(set)

        for group in AMBIGUOUS_INTENT_GROUPS:
            for intent in group:
                ambiguous_map[intent].update(group - {intent})

        return dict(ambiguous_map)

    def analyze(
        self,
        texts: List[str],
        intent_true: List[int],
        intent_pred: List[int],
        slots_true: List[List[str]],
        slots_pred: List[List[str]],
        confidences: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Analyze all errors and categorize them.

        Args:
            texts: Input texts
            intent_true: True intent IDs
            intent_pred: Predicted intent IDs
            slots_true: True BIO tag sequences
            slots_pred: Predicted BIO tag sequences
            confidences: Prediction confidence scores (optional)

        Returns:
            Dictionary with error analysis results
        """
        # Calculate intent frequencies
        self._intent_counts = Counter(intent_true)

        # Collect errors
        errors = {
            "ambiguous_intent": [],
            "rare_intent": [],
            "slot_boundary": [],
            "oov_related": [],
            "other": [],
        }

        all_errors = []

        for i, (text, it, ip, st, sp) in enumerate(zip(
            texts, intent_true, intent_pred, slots_true, slots_pred
        )):
            intent_error = it != ip
            slot_error = self._has_slot_error(st, sp)

            if not intent_error and not slot_error:
                continue

            # Build error record
            error = {
                "index": i,
                "text": text,
                "true_intent": self.id2intent.get(it, f"intent_{it}"),
                "pred_intent": self.id2intent.get(ip, f"intent_{ip}"),
                "true_intent_id": it,
                "pred_intent_id": ip,
                "true_slots": st,
                "pred_slots": sp,
                "has_intent_error": intent_error,
                "has_slot_error": slot_error,
                "categories": [],
            }

            if confidences and i < len(confidences):
                error["confidence"] = confidences[i]

            # Categorize the error
            if intent_error:
                categories = self._categorize_intent_error(text, it, ip)
                error["categories"].extend(categories)

            if slot_error:
                slot_categories = self._categorize_slot_error(st, sp)
                error["categories"].extend(slot_categories)

            # If no specific category, mark as other
            if not error["categories"]:
                error["categories"].append("other")

            # Add to appropriate buckets
            for category in set(error["categories"]):
                errors[category].append(error)

            all_errors.append(error)

        # Calculate statistics
        stats = self._calculate_statistics(all_errors, intent_true)

        return {
            "errors": errors,
            "all_errors": all_errors,
            "statistics": stats,
            "total_samples": len(texts),
            "total_errors": len(all_errors),
            "error_rate": len(all_errors) / len(texts) if texts else 0,
        }

    def _has_slot_error(
        self,
        true_slots: List[str],
        pred_slots: List[str],
    ) -> bool:
        """Check if there's any slot error."""
        if len(true_slots) != len(pred_slots):
            return True

        return any(t != p for t, p in zip(true_slots, pred_slots))

    def _categorize_intent_error(
        self,
        text: str,
        true_id: int,
        pred_id: int,
    ) -> List[str]:
        """Categorize an intent error.

        Returns list of applicable categories.
        """
        categories = []

        true_intent = self.id2intent.get(true_id, "")
        pred_intent = self.id2intent.get(pred_id, "")

        # Check if ambiguous
        if pred_intent in self._ambiguous_map.get(true_intent, set()):
            categories.append("ambiguous_intent")

        # Check if rare intent
        if self._intent_counts.get(true_id, 0) < self.rare_threshold:
            categories.append("rare_intent")

        # Check for OOV-related (if vocabulary available)
        if self._vocab:
            words = text.lower().split()
            oov_words = [w for w in words if w not in self._vocab]
            if len(oov_words) > 0 and len(oov_words) / len(words) > 0.2:
                categories.append("oov_related")

        return categories

    def _categorize_slot_error(
        self,
        true_slots: List[str],
        pred_slots: List[str],
    ) -> List[str]:
        """Categorize slot errors.

        Returns list of applicable categories.
        """
        categories = []

        min_len = min(len(true_slots), len(pred_slots))

        # Check for boundary errors (B-/I- confusion)
        boundary_errors = 0
        type_errors = 0

        for t, p in zip(true_slots[:min_len], pred_slots[:min_len]):
            if t == p:
                continue

            # Extract prefix and type
            t_prefix = t.split("-")[0] if "-" in t else t
            p_prefix = p.split("-")[0] if "-" in p else p
            t_type = t.split("-")[1] if "-" in t else t
            p_type = p.split("-")[1] if "-" in p else p

            # Boundary error: same type but different prefix
            if t_type == p_type and t_prefix != p_prefix:
                boundary_errors += 1
            elif t_type != p_type and t != "O" and p != "O":
                type_errors += 1

        if boundary_errors > 0:
            categories.append("slot_boundary")

        return categories

    def _calculate_statistics(
        self,
        all_errors: List[Dict],
        intent_true: List[int],
    ) -> Dict[str, Any]:
        """Calculate error statistics."""
        category_counts = defaultdict(int)
        intent_error_counts = defaultdict(int)

        for error in all_errors:
            for category in error["categories"]:
                category_counts[category] += 1

            if error["has_intent_error"]:
                true_intent = error["true_intent"]
                intent_error_counts[true_intent] += 1

        # Calculate per-category rates
        total_errors = len(all_errors) if all_errors else 1
        category_rates = {
            k: v / total_errors for k, v in category_counts.items()
        }

        # Find most confused intent pairs
        confusion_pairs = defaultdict(int)
        for error in all_errors:
            if error["has_intent_error"]:
                pair = (error["true_intent"], error["pred_intent"])
                confusion_pairs[pair] += 1

        top_confusion_pairs = sorted(
            confusion_pairs.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        return {
            "category_counts": dict(category_counts),
            "category_rates": category_rates,
            "intent_error_counts": dict(intent_error_counts),
            "top_confusion_pairs": [
                {"true": p[0], "pred": p[1], "count": c}
                for (p, c) in top_confusion_pairs
            ],
        }

    def generate_report(
        self,
        analysis: Dict[str, Any],
        max_examples_per_category: int = 5,
    ) -> str:
        """Generate a markdown report of the error analysis.

        Args:
            analysis: Results from analyze()
            max_examples_per_category: Max examples to show per category

        Returns:
            Markdown-formatted report string
        """
        lines = [
            "# NLU Error Analysis Report",
            "",
            "## Overview",
            "",
            f"- **Total samples**: {analysis['total_samples']}",
            f"- **Total errors**: {analysis['total_errors']}",
            f"- **Error rate**: {analysis['error_rate']:.2%}",
            "",
            "## Error Categories",
            "",
        ]

        stats = analysis["statistics"]
        category_counts = stats["category_counts"]
        category_rates = stats["category_rates"]

        # Category table
        lines.append("| Category | Count | Rate |")
        lines.append("|----------|-------|------|")

        for category in ["ambiguous_intent", "rare_intent", "slot_boundary", "oov_related", "other"]:
            count = category_counts.get(category, 0)
            rate = category_rates.get(category, 0)
            lines.append(f"| {category} | {count} | {rate:.1%} |")

        lines.append("")

        # Top confusion pairs
        lines.append("## Top Confusion Pairs")
        lines.append("")
        lines.append("| True Intent | Predicted Intent | Count |")
        lines.append("|-------------|------------------|-------|")

        for pair in stats["top_confusion_pairs"][:10]:
            lines.append(f"| {pair['true']} | {pair['pred']} | {pair['count']} |")

        lines.append("")

        # Example errors per category
        for category in ["ambiguous_intent", "rare_intent", "slot_boundary", "oov_related"]:
            errors = analysis["errors"].get(category, [])
            if not errors:
                continue

            lines.append(f"## {category.replace('_', ' ').title()} Examples")
            lines.append("")

            for i, error in enumerate(errors[:max_examples_per_category], 1):
                lines.append(f"### Example {i}")
                lines.append("")
                lines.append(f"**Text**: {error['text']}")
                lines.append(f"**True Intent**: {error['true_intent']}")
                lines.append(f"**Predicted Intent**: {error['pred_intent']}")

                if error.get("confidence"):
                    lines.append(f"**Confidence**: {error['confidence']:.3f}")

                if error["has_slot_error"]:
                    lines.append(f"**True Slots**: {' '.join(error['true_slots'][:10])}...")
                    lines.append(f"**Pred Slots**: {' '.join(error['pred_slots'][:10])}...")

                lines.append("")

        return "\n".join(lines)


def save_error_analysis(
    analysis: Dict[str, Any],
    output_dir: str,
    model_name: str = "model",
) -> Dict[str, str]:
    """Save error analysis results to files.

    Args:
        analysis: Results from ErrorAnalyzer.analyze()
        output_dir: Directory to save results
        model_name: Model name for file naming

    Returns:
        Dictionary mapping file type to path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save full analysis as JSON
    json_path = output_dir / f"error_analysis_{model_name}.json"
    # Convert to serializable format
    serializable = {
        "total_samples": analysis["total_samples"],
        "total_errors": analysis["total_errors"],
        "error_rate": analysis["error_rate"],
        "statistics": analysis["statistics"],
        "errors_by_category": {
            category: len(errors)
            for category, errors in analysis["errors"].items()
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    paths["json"] = str(json_path)

    return paths


def generate_combined_error_report(
    svm_analysis: Optional[Dict] = None,
    bert_analysis: Optional[Dict] = None,
    llm_analysis: Optional[Dict] = None,
    output_path: str = "results/error_analysis.md",
) -> str:
    """Generate combined error analysis report for all models.

    Args:
        svm_analysis: SVM error analysis results
        bert_analysis: JointBERT error analysis results
        llm_analysis: LLM error analysis results
        output_path: Path to save the report

    Returns:
        Path to saved report
    """
    lines = [
        "# NLU Error Analysis Report",
        "",
        "Comparative error analysis for SVM, JointBERT, and LLM models.",
        "",
        "## Summary",
        "",
        "| Model | Total Errors | Error Rate | Ambiguous | Rare | Slot Boundary | OOV |",
        "|-------|-------------|------------|-----------|------|---------------|-----|",
    ]

    analyses = [
        ("SVM", svm_analysis),
        ("JointBERT", bert_analysis),
        ("LLM", llm_analysis),
    ]

    for name, analysis in analyses:
        if analysis is None:
            continue

        stats = analysis["statistics"]
        cat_counts = stats["category_counts"]

        lines.append(
            f"| {name} | {analysis['total_errors']} | "
            f"{analysis['error_rate']:.1%} | "
            f"{cat_counts.get('ambiguous_intent', 0)} | "
            f"{cat_counts.get('rare_intent', 0)} | "
            f"{cat_counts.get('slot_boundary', 0)} | "
            f"{cat_counts.get('oov_related', 0)} |"
        )

    lines.append("")

    # Detailed analysis per model
    for name, analysis in analyses:
        if analysis is None:
            continue

        lines.append(f"## {name} Model Analysis")
        lines.append("")
        lines.append(f"- Total samples: {analysis['total_samples']}")
        lines.append(f"- Total errors: {analysis['total_errors']}")
        lines.append(f"- Error rate: {analysis['error_rate']:.2%}")
        lines.append("")

        # Top confusion pairs
        top_pairs = analysis["statistics"]["top_confusion_pairs"][:5]
        if top_pairs:
            lines.append("### Top Confusion Pairs")
            lines.append("")
            for pair in top_pairs:
                lines.append(f"- {pair['true']} -> {pair['pred']} ({pair['count']} errors)")
            lines.append("")

        # Sample errors
        lines.append("### Sample Errors")
        lines.append("")

        all_errors = analysis.get("all_errors", [])[:5]
        for i, error in enumerate(all_errors, 1):
            lines.append(f"**{i}. {error['text'][:60]}...**")
            lines.append(f"   - True: {error['true_intent']}, Pred: {error['pred_intent']}")
            if error.get("categories"):
                lines.append(f"   - Categories: {', '.join(error['categories'])}")
            lines.append("")

    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved error analysis report: {output_path}")

    return str(output_path)
