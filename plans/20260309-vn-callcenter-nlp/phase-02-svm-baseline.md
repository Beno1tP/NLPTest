# Phase 2: TF-IDF + SVM Baseline

**Status:** Pending
**Priority:** High
**Depends On:** Phase 1
**Research:** `docs/research/svm-baseline-research.md`

---

## Context

Baseline NLU using traditional ML. LinearSVC for intent, CRF for slots.

---

## Requirements

1. TF-IDF vectorizer with Vietnamese tokenization
2. LinearSVC intent classifier
3. CRF slot filling model
4. Training and evaluation scripts

---

## Implementation Steps

### 1. Intent Classifier
```
src/nlu/svm_intent.py
```
- underthesea word_tokenize
- TfidfVectorizer(ngram_range=(1,2), max_features=10000)
- CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced'))
- save/load with joblib

### 2. Slot Filler
```
src/nlu/crf_slot.py
```
- sklearn-crfsuite CRF
- word2features: word, prefix, suffix, position
- BIO label handling

### 3. Combined Pipeline
```
src/nlu/svm_nlu.py
```
- SVMIntentClassifier + CRFSlotFiller
- Joint predict(text) -> {intent, slots}

### 4. Training Script
```
scripts/train_svm.py
```
- Load data from Phase 1
- Train both models
- Save to `models/svm_intent.joblib`, `models/crf_slot.joblib`

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/nlu/svm_intent.py` | Intent classifier |
| `src/nlu/crf_slot.py` | Slot filler |
| `src/nlu/svm_nlu.py` | Combined pipeline |
| `scripts/train_svm.py` | Training script |

---

## Success Criteria

- [ ] Intent accuracy >90% on test set
- [ ] Slot F1 >85% on test set
- [ ] Models saved to `models/`
- [ ] Inference time <100ms
