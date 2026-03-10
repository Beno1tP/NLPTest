# Vietnamese AI Call Center - Implementation Plan

**Created:** 2026-03-09
**Status:** In Progress
**Timeline:** 2 weeks

---

## Overview

Task-oriented dialogue system with 3-way NLU comparison for Vietnamese air travel domain.

**Pipeline:** STT → NLU → DST → Policy → NLG → TTS

---

## Phases

| Phase | Name | Status | Dependencies |
|-------|------|--------|--------------|
| 1 | [Data Pipeline](./phase-01-data-pipeline.md) | Pending | None |
| 2 | [SVM Baseline](./phase-02-svm-baseline.md) | Pending | Phase 1 |
| 3 | [JointBERT](./phase-03-jointbert.md) | Pending | Phase 1 |
| 4 | [LLM Zero-Shot](./phase-04-llm-zeroshot.md) | Pending | Phase 1 |
| 5 | [Dialogue System](./phase-05-dialogue-system.md) | Pending | Phase 2,3,4 |
| 6 | [Speech Integration](./phase-06-speech-integration.md) | Pending | None |
| 7 | [Streamlit Demo](./phase-07-streamlit-demo.md) | Pending | Phase 5,6 |
| 8 | [Evaluation](./phase-08-evaluation.md) | Pending | Phase 2,3,4 |

---

## Key Deliverables

1. **NLU Models:** 3 trained/configured models with comparison metrics
2. **Full Pipeline:** End-to-end STT→NLU→DST→Policy→NLG→TTS
3. **Demo UI:** Streamlit app with voice/text modes
4. **Report:** Academic paper with evaluation figures

---

## Research References

- `docs/research/jointbert-phobert-research.md`
- `docs/research/svm-baseline-research.md`
- `docs/research/llm-zeroshot-research.md`
- `docs/research/speech-apis-research.md`
- `docs/research/streamlit-demo-research.md`

---

## Success Criteria

- Intent accuracy >90% for best model
- Slot F1 >85% for best model
- Demo processes voice input end-to-end
- Comparison figures generated
