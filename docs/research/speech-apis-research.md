# Speech-to-Text & Text-to-Speech APIs for Vietnamese

## 1. Google Cloud Speech-to-Text (STT)

### Configuration
- **Language Code:** `vi-VN` (BCP-47)
- **Model:** `chirp_3` (recommended, GA)
- **API Version:** V2
- **Regional Endpoint:** `asia-southeast1` (lowest latency for Vietnam)

### Python Example (V2 API)
```python
from google.cloud import speech_v2 as speech

client = speech.SpeechClient()

config = speech.RecognitionConfig(
    auto_decoding_config=speech.AutoDetectDecodingConfig(),
    language_codes=["vi-VN"],
    model="chirp_3",
    features=speech.RecognitionFeatures(
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
    ),
)

# Batch transcription
request = speech.RecognizeRequest(
    recognizer=f"projects/{PROJECT_ID}/locations/asia-southeast1/recognizers/_",
    config=config,
    content=audio_content,  # bytes
)
response = client.recognize(request=request)

for result in response.results:
    print(result.alternatives[0].transcript)
```

### Real-time Streaming
```python
def stream_generator(audio_chunks):
    yield speech.StreamingRecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/asia-southeast1/recognizers/_",
        streaming_config=speech.StreamingRecognitionConfig(
            config=config,
            streaming_features=speech.StreamingRecognitionFeatures(
                interim_results=True,
            ),
        ),
    )
    for chunk in audio_chunks:
        yield speech.StreamingRecognizeRequest(audio=chunk)

responses = client.streaming_recognize(requests=stream_generator(chunks))
```

### Vietnamese Feature Support
| Feature | Status |
|---------|--------|
| Chirp 3 Model | GA |
| Auto Punctuation | Supported |
| Denoiser | Supported |
| Speech Adaptation (hints) | Up to 1,000 phrases |
| Speaker Diarization | NOT Supported |

### Audio Formats
- LINEAR16, FLAC, MP3, OGG_OPUS, WEBM_OPUS
- Sample rate: 16kHz recommended

---

## 2. Google Cloud Text-to-Speech (TTS)

### Vietnamese Voices
| Tier | Voice IDs | Notes |
|------|-----------|-------|
| Neural2 | `vi-VN-Neural2-A/C` (F), `vi-VN-Neural2-B/D` (M) | A/C = Northern, B/D = Southern |
| WaveNet | `vi-VN-Wavenet-A/C` (F), `vi-VN-Wavenet-B/D` (M) | DeepMind-powered |
| Standard | `vi-VN-Standard-A/C` (F), `vi-VN-Standard-B/D` (M) | Cost-effective |

### Python Example
```python
from google.cloud import texttospeech

client = texttospeech.TextToSpeechClient()

input_text = texttospeech.SynthesisInput(text="Xin chao Vietnam")

voice = texttospeech.VoiceSelectionParams(
    language_code="vi-VN",
    name="vi-VN-Neural2-A",
    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
)

audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3,
    speaking_rate=1.0,
    pitch=0.0,
)

response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

with open("output.mp3", "wb") as f:
    f.write(response.audio_content)
```

### SSML Support
```python
ssml_text = """
<speak>
    <prosody rate="slow" pitch="+2st">Xin chao</prosody>
    <break time="500ms"/>
    <say-as interpret-as="telephone">0901234567</say-as>
    <sub alias="Thanh pho Ho Chi Minh">TPHCM</sub>
</speak>
"""
input_text = texttospeech.SynthesisInput(ssml=ssml_text)
```

---

## 3. Free Alternatives

### gTTS (Google Translate TTS)
```python
from gtts import gTTS

tts = gTTS(text="Xin chao", lang='vi')
tts.save("output.mp3")
```
- Free, no API key
- Single female voice, robotic quality
- Rate-limited

### edge-tts (Microsoft Neural Voices) - RECOMMENDED
```python
import edge_tts
import asyncio

async def speak(text):
    communicate = edge_tts.Communicate(text, "vi-VN-HoaiMyNeural")
    await communicate.save("output.mp3")

asyncio.run(speak("Xin chao Vietnam"))
```
- **Voices:** `vi-VN-HoaiMyNeural` (F), `vi-VN-NamMinhNeural` (M)
- Free, high-quality neural voices
- No API key required

### SpeechRecognition + Whisper (Free STT)
```python
import whisper

model = whisper.load_model("large-v3-turbo")
result = model.transcribe("audio.wav", language="vi")
print(result["text"])
```
- Offline capable
- ~8.81% WER on Vietnamese
- Requires GPU (6GB+ VRAM)

### faster-whisper (Optimized)
```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.wav", language="vi")
for segment in segments:
    print(segment.text)
```

---

## 4. Streamlit Audio Recording

### st.audio_input (Built-in)
```python
import streamlit as st

audio_bytes = st.audio_input("Record voice message")

if audio_bytes:
    st.audio(audio_bytes)
    # Process with STT
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes.getvalue())
```

### streamlit-webrtc (Real-time)
```python
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import queue

audio_queue = queue.Queue()

def audio_frame_callback(frame: av.AudioFrame):
    audio_queue.put(frame.to_ndarray())
    return frame

ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": False, "audio": True},
    audio_frame_callback=audio_frame_callback,
)

if ctx.state.playing:
    # Process frames from audio_queue
    pass
```

### Full Integration Example
```python
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import edge_tts
import asyncio
from faster_whisper import WhisperModel

# STT
model = WhisperModel("base", device="cpu")

audio = st.audio_input("Speak in Vietnamese")
if audio:
    with open("input.wav", "wb") as f:
        f.write(audio.getvalue())
    segments, _ = model.transcribe("input.wav", language="vi")
    text = " ".join([s.text for s in segments])
    st.write(f"Transcription: {text}")

    # TTS response
    async def respond():
        comm = edge_tts.Communicate(f"Ban da noi: {text}", "vi-VN-HoaiMyNeural")
        await comm.save("response.mp3")
    asyncio.run(respond())
    st.audio("response.mp3")
```

---

## 5. Comparison Summary

| Solution | Cost | Quality | Offline | Best For |
|----------|------|---------|---------|----------|
| Google Cloud STT | Paid | Excellent | No | Production apps |
| Google Cloud TTS | Paid | Excellent | No | Production apps |
| Whisper/faster-whisper | Free | Very Good | Yes | Offline STT |
| edge-tts | Free | Very Good | No | Free TTS |
| gTTS | Free | Basic | No | Simple prototypes |
| Vosk | Free | Moderate | Yes | Edge/embedded |

## Unresolved Questions
1. Google Cloud pricing tiers for Vietnamese specifically (usage-based)
2. Exact latency benchmarks for streaming STT in Vietnam region
3. PhoWhisper vs Whisper accuracy on Southern Vietnamese accent
