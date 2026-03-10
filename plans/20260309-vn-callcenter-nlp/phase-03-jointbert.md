# Phase 3: JointBERT + PhoBERT

**Status:** Pending
**Priority:** High
**Depends On:** Phase 1
**Research:** `docs/research/jointbert-phobert-research.md`

---

## Context

State-of-the-art joint intent+slot model using PhoBERT encoder.

---

## Requirements

1. JointBERT model architecture
2. Subword alignment for slot labels
3. CRF layer for slot prediction
4. Training with early stopping

---

## Implementation Steps

### 1. Model Architecture
```
src/nlu/jointbert_model.py
```
- phobert: AutoModel("vinai/phobert-base-v2")
- intent_classifier: Linear(768, num_intents)
- slot_classifier: Linear(768, num_slots)
- crf: CRF(num_slots)

### 2. Data Processing
```
src/nlu/jointbert_data.py
```
- First-token alignment strategy
- -100 for non-first subwords
- Attention mask handling

### 3. Trainer
```
src/nlu/jointbert_trainer.py
```
- AdamW optimizer
- Linear warmup scheduler
- Joint loss: intent_loss + slot_loss
- Early stopping on dev F1

### 4. Training Script
```
scripts/train_jointbert.py
```
- Load config from `configs/joint_bert_config.yaml`
- Train with checkpointing
- Save best model to `models/best_jointbert.pt`

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/nlu/jointbert_model.py` | Model class |
| `src/nlu/jointbert_data.py` | Dataset class |
| `src/nlu/jointbert_trainer.py` | Training logic |
| `scripts/train_jointbert.py` | Entry point |

---

## Success Criteria

- [ ] Intent accuracy >94% on test set
- [ ] Slot F1 >92% on test set
- [ ] Model checkpointed during training
- [ ] Training completes in <2 hours (GPU)
