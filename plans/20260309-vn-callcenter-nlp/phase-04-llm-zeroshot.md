# Phase 4: LLM Zero-Shot Evaluation

**Status:** Pending
**Priority:** High
**Depends On:** Phase 1
**Research:** `docs/research/llm-zeroshot-research.md`

---

## Context

Evaluate LLM zero-shot/few-shot on PhoATIS without training.

---

## Requirements

1. Structured prompt for intent+slot extraction
2. API clients for Claude and GPT
3. JSON output parsing with tool use
4. Batch evaluation with rate limiting

---

## Implementation Steps

### 1. Prompt Templates
```
src/nlu/llm_prompts.py
```
- System prompt with intent/slot definitions
- User template: "{utterance}" → JSON output
- Few-shot examples (optional)

### 2. API Clients
```
src/nlu/llm_client.py
```
- AnthropicClient: tool_use for structured output
- OpenAIClient: JSON mode
- Retry logic, rate limiting

### 3. LLM NLU Wrapper
```
src/nlu/llm_nlu.py
```
```python
class LLMNLUClassifier:
    def predict(text) -> {intent, confidence, slots}
```

### 4. Evaluation Script
```
scripts/evaluate_llm.py
```
- Load test set
- Batch processing (rate limited)
- Save results to `results/llm_evaluation.json`

---

## Config

`configs/llm_config.yaml`:
```yaml
provider: anthropic
models:
  anthropic:
    model: claude-3-haiku-20240307
    temperature: 0.0
evaluation:
  batch_size: 10
  delay_seconds: 0.5
  max_samples: 500
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/nlu/llm_prompts.py` | Prompt templates |
| `src/nlu/llm_client.py` | API wrappers |
| `src/nlu/llm_nlu.py` | NLU interface |
| `scripts/evaluate_llm.py` | Batch evaluation |

---

## Success Criteria

- [ ] Zero-shot intent accuracy >85%
- [ ] Structured JSON output (no parsing failures)
- [ ] Evaluation completes within API budget
- [ ] Results saved with confidence scores
