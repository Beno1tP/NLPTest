# Phase 1: Data Pipeline

**Status:** Pending
**Priority:** Critical
**Blocks:** Phase 2, 3, 4, 8

---

## Context

PhoATIS dataset must be downloaded, processed, and formatted for all 3 NLU approaches.

---

## Requirements

1. Download PhoATIS dataset from VinAI
2. Create train/dev/test splits
3. Build label vocabularies (intents, slots)
4. Implement data loaders for each model type

---

## Implementation Steps

### 1. Dataset Download
```
scripts/download_phoatis.py
```
- Fetch from VinAI/PhoATIS GitHub
- Verify integrity

### 2. Data Processing
```
src/data/processor.py
```
- Parse seq.in, seq.out, label files
- Word segmentation (VnCoreNLP)
- Build intent2id, slot2id mappings
- Save to `data/processed/`

### 3. Data Loaders
```
src/data/loaders.py
```
- `SVMDataLoader`: Return tokenized texts + labels
- `BERTDataLoader`: Return tensors with subword alignment
- `LLMDataLoader`: Return raw texts for prompting

### 4. Label Files
```
data/processed/
├── intent_labels.txt
├── slot_labels.txt
├── train/
├── dev/
└── test/
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `scripts/download_phoatis.py` | Dataset download |
| `src/data/__init__.py` | Package init |
| `src/data/processor.py` | Data processing |
| `src/data/loaders.py` | Model-specific loaders |

---

## Success Criteria

- [ ] PhoATIS downloaded to `data/raw/`
- [ ] Processed data in `data/processed/`
- [ ] 28 intent labels extracted
- [ ] 82+ slot labels extracted (with BIO)
- [ ] All loaders return correct formats
