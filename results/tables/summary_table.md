# Model Comparison Summary

| Model | Intent Acc | Intent F1 | Slot F1 | Sent Acc | Train Time | Inference |
|-------|-----------|-----------|---------|----------|------------|-----------|
| SVM | 0.942 | 0.891 | 0.923 | 0.823 | 0.2m | 2.5s |
| JointBERT | 0.971 | 0.943 | 0.952 | 0.891 | 5.0m | 45.0s |
| LLM | 0.754 | 0.712 | 0.651 | 0.456 | N/A | 120.0s |