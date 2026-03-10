# Phase 8: Evaluation & Report

**Status:** Pending
**Priority:** High
**Depends On:** Phase 2, 3, 4

---

## Context

Compare 3 NLU approaches, generate figures for academic report.

---

## Requirements

1. Unified evaluation script
2. Comparison metrics table
3. Visualization figures
4. Error analysis

---

## Implementation Steps

### 1. Evaluation Framework
```
src/evaluation/evaluator.py
```
```python
class NLUEvaluator:
    def evaluate(model, test_data) -> {
        intent_accuracy,
        intent_f1_macro,
        slot_f1,
        sentence_accuracy,
        confusion_matrix
    }
```

### 2. Comparison Script
```
scripts/run_evaluation.py
```
- Load all 3 models
- Evaluate on same test set
- Save results to `results/comparison.json`

### 3. Visualization
```
scripts/generate_figures.py
```
Figures:
- Intent accuracy bar chart
- Slot F1 comparison
- Confusion matrices (per model)
- Latency comparison
- Error type distribution

### 4. Error Analysis
```
src/evaluation/error_analysis.py
```
- Misclassified examples
- Slot boundary errors
- Intent confusion pairs

---

## Output Files

```
results/
├── comparison.json          # Raw metrics
├── figures/
│   ├── intent_accuracy.png
│   ├── slot_f1.png
│   ├── confusion_svm.png
│   ├── confusion_bert.png
│   ├── confusion_llm.png
│   └── latency.png
└── error_analysis.json
```

---

## Metrics

| Metric | SVM Target | BERT Target | LLM Target |
|--------|------------|-------------|------------|
| Intent Acc | >90% | >94% | >85% |
| Intent F1 | >85% | >92% | >80% |
| Slot F1 | >85% | >92% | >75% |
| Latency | <100ms | <200ms | <800ms |

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/evaluation/evaluator.py` | Metrics computation |
| `src/evaluation/error_analysis.py` | Error breakdown |
| `scripts/run_evaluation.py` | Unified evaluation |
| `scripts/generate_figures.py` | Chart generation |

---

## Success Criteria

- [ ] All 3 models evaluated on same test set
- [ ] Comparison table generated
- [ ] Figures saved as PNG
- [ ] Error analysis identifies top failure modes
