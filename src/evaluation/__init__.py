"""Evaluation modules for Vietnamese NLU models.

Provides comprehensive evaluation tools for comparing SVM, JointBERT, and LLM models:
- Unified metrics (intent accuracy, slot F1, sentence accuracy)
- Confusion matrix visualization
- Error analysis and categorization
- Learning curve evaluation
"""

from .metrics import (
    NLUEvaluator,
    compute_all_metrics,
    format_metrics_table,
    get_confusion_matrix,
    get_slot_classification_report,
    intent_accuracy,
    intent_f1_macro,
    per_intent_metrics,
    sentence_accuracy,
    slot_f1_entity,
    slot_f1_token,
)

from .confusion_matrix import (
    analyze_confusion_patterns,
    format_confusion_analysis,
    generate_all_confusion_matrices,
    plot_comparison_confusion_matrices,
    plot_confusion_matrix,
    plot_top_k_confusion_matrix,
)

from .error_analysis import (
    ErrorAnalyzer,
    generate_combined_error_report,
    save_error_analysis,
)

from .learning_curve import (
    LearningCurveEvaluator,
    evaluate_svm_learning_curve,
    generate_learning_curve_report,
    get_mock_bert_learning_curve,
    get_mock_llm_learning_curve,
    plot_learning_curves,
    save_learning_curve_data,
)

__all__ = [
    # Metrics
    "NLUEvaluator",
    "compute_all_metrics",
    "format_metrics_table",
    "get_confusion_matrix",
    "get_slot_classification_report",
    "intent_accuracy",
    "intent_f1_macro",
    "per_intent_metrics",
    "sentence_accuracy",
    "slot_f1_entity",
    "slot_f1_token",
    # Confusion matrix
    "analyze_confusion_patterns",
    "format_confusion_analysis",
    "generate_all_confusion_matrices",
    "plot_comparison_confusion_matrices",
    "plot_confusion_matrix",
    "plot_top_k_confusion_matrix",
    # Error analysis
    "ErrorAnalyzer",
    "generate_combined_error_report",
    "save_error_analysis",
    # Learning curve
    "LearningCurveEvaluator",
    "evaluate_svm_learning_curve",
    "generate_learning_curve_report",
    "get_mock_bert_learning_curve",
    "get_mock_llm_learning_curve",
    "plot_learning_curves",
    "save_learning_curve_data",
]
