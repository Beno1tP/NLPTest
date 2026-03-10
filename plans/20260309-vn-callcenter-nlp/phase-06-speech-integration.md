# Phase 6: Speech Integration (STT + TTS)

**Status:** Pending
**Priority:** Medium
**Depends On:** None (parallel track)
**Research:** `docs/research/speech-apis-research.md`

---

## Context

Voice I/O using Google Cloud APIs with free alternatives.

---

## Requirements

1. STT wrapper (Google Cloud + faster-whisper fallback)
2. TTS wrapper (Google Cloud + edge-tts fallback)
3. Audio format handling (WAV, MP3)
4. Streaming support (optional)

---

## Implementation Steps

### 1. STT Module
```
src/speech/stt.py
```
```python
class GoogleSTT:
    def transcribe(audio_bytes) -> str
    # V2 API, Chirp 3 model, vi-VN

class WhisperSTT:
    def transcribe(audio_path) -> str
    # faster-whisper, large-v3-turbo
```

### 2. TTS Module
```
src/speech/tts.py
```
```python
class GoogleTTS:
    def synthesize(text) -> bytes
    # Neural2-A voice, MP3

class EdgeTTS:
    async def synthesize(text) -> bytes
    # vi-VN-HoaiMyNeural
```

### 3. Audio Utilities
```
src/speech/audio_utils.py
```
- WAV ↔ MP3 conversion
- Sample rate conversion (to 16kHz)
- pydub helpers

### 4. Unified Interface
```
src/speech/speech_service.py
```
```python
class SpeechService:
    def __init__(config)
    def speech_to_text(audio) -> str
    def text_to_speech(text) -> bytes
```

---

## Config

`configs/speech_config.yaml`:
```yaml
stt:
  provider: google_cloud  # or whisper
tts:
  provider: edge_tts  # or google_cloud
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/speech/stt.py` | STT implementations |
| `src/speech/tts.py` | TTS implementations |
| `src/speech/audio_utils.py` | Audio helpers |
| `src/speech/speech_service.py` | Unified interface |

---

## Success Criteria

- [ ] STT accuracy matches API benchmarks
- [ ] TTS produces natural Vietnamese speech
- [ ] Fallback to free alternatives works
- [ ] Round-trip latency <3s
