# Phase 7: Streamlit Demo

**Status:** Pending
**Priority:** High
**Depends On:** Phase 5, 6
**Research:** `docs/research/streamlit-demo-research.md`

---

## Context

Interactive demo with voice/text input, model selection, pipeline visualization.

---

## Requirements

1. Voice input (st.audio_input)
2. Text chat interface
3. Model selection (SVM/JointBERT/LLM)
4. Pipeline stage visualization
5. Vietnamese text display

---

## Implementation Steps

### 1. Main App
```
app/main.py
```
- st.set_page_config(layout="wide")
- Sidebar: model selection, settings
- Main: audio input, chat, results

### 2. Components
```
app/components/
├── audio_input.py    # Voice recording
├── chat.py           # Chat interface
├── pipeline_viz.py   # Stage visualization
└── metrics.py        # Performance display
```

### 3. Session State
```python
st.session_state:
  - messages: []
  - current_model: "jointbert"
  - pipeline_state: {}
  - audio_bytes: None
```

### 4. Pipeline Integration
```
app/pipeline.py
```
- Load models on startup (cached)
- Process audio → text → NLU → response → audio

---

## Layout

```
┌─────────────────────────────────────────────────┐
│ Sidebar          │  Main Content                │
│ ─────────────────│──────────────────────────────│
│ Model: [▼]       │  🎤 [Record Audio]           │
│ ○ SVM            │  ──────────────────────────  │
│ ● JointBERT      │  Pipeline Status:            │
│ ○ LLM            │  [STT ✓] [NLU ✓] [DST →]    │
│                  │  ──────────────────────────  │
│ Settings:        │  Chat:                       │
│ Confidence: 0.7  │  User: Tôi muốn đặt vé...   │
│ Show details: ☑  │  Bot: Bạn muốn bay đi đâu?  │
└─────────────────────────────────────────────────┘
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `app/main.py` | Entry point |
| `app/components/audio_input.py` | Audio recording |
| `app/components/chat.py` | Chat interface |
| `app/components/pipeline_viz.py` | Pipeline status |
| `app/pipeline.py` | Backend integration |

---

## Run Command

```bash
streamlit run app/main.py
```

---

## Success Criteria

- [ ] Voice input transcribes correctly
- [ ] All 3 models selectable
- [ ] Pipeline stages visualized
- [ ] Vietnamese displays correctly
- [ ] Response time <5s end-to-end
