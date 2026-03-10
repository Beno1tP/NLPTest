# Phase 5: Dialogue System (DST + Policy + NLG)

**Status:** Pending
**Priority:** Medium
**Depends On:** Phase 2, 3, 4

---

## Context

Complete dialogue management: track state, select actions, generate responses.

---

## Requirements

1. Dialogue State Tracker (rule-based)
2. Policy module (rule-based action selection)
3. NLG module (template-based Vietnamese responses)
4. End-to-end pipeline orchestration

---

## Implementation Steps

### 1. Dialogue State Tracker
```
src/dst/tracker.py
```
```python
class DialogueState:
    intent_history: list
    slots: dict  # accumulated
    turn_count: int
    completed: bool

class StateTracker:
    def update(nlu_output) -> DialogueState
```

### 2. Policy Module
```
src/policy/rule_policy.py
```
Actions:
- `confirm_slot`: Confirm extracted value
- `request_slot`: Ask for missing required slot
- `api_call`: Execute booking/query
- `respond`: Generate final response

### 3. NLG Module
```
src/nlg/templates.py
```
Vietnamese templates:
```python
TEMPLATES = {
    "confirm_slot": "Bạn muốn bay từ {fromloc} đến {toloc}, đúng không?",
    "request_slot": "Bạn muốn bay đi đâu?",
    "flight_result": "Có {count} chuyến bay từ {fromloc} đến {toloc}..."
}
```

### 4. Pipeline Orchestrator
```
src/pipeline/orchestrator.py
```
```python
class DialoguePipeline:
    def __init__(nlu_model, tracker, policy, nlg)
    def process(user_input) -> response
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/dst/tracker.py` | State tracking |
| `src/policy/rule_policy.py` | Action selection |
| `src/nlg/templates.py` | Response generation |
| `src/pipeline/orchestrator.py` | Full pipeline |

---

## Success Criteria

- [ ] Multi-turn dialogue maintains state
- [ ] Required slots requested when missing
- [ ] Vietnamese responses grammatically correct
- [ ] Pipeline processes input→output in <2s
