# Streamlit AI Call Center Demo - Best Practices Research

## 1. Layout Patterns

### Sidebar for Controls
```python
import streamlit as st

st.sidebar.title("Call Center Controls")
st.sidebar.selectbox("Model", ["PhoWhisper", "Whisper-large"])
st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7)
st.sidebar.toggle("Auto-transcribe", value=True)
```

### Columns for Metrics Dashboard
```python
col1, col2, col3 = st.columns(3)
col1.metric("Calls Processed", "127", "+12%")
col2.metric("Avg. Duration", "2:34", "-8%")
col3.metric("Accuracy", "94.2%", "+1.5%")
```

### Container-based Pipeline View
```python
with st.container(border=True):
    st.subheader("NLP Pipeline")
    # nested content here
```

## 2. Chat Interface

### Basic Chat with History
```python
# Initialize
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input handling
if prompt := st.chat_input("Type message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(generate_response(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})
```

### Streaming Response Generator
```python
def stream_response(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.03)
```

## 3. Audio Recording

### Option A: Native `st.audio_input` (Recommended)
Streamlit 1.34+. Batch mode (record -> stop -> process).
```python
audio = st.audio_input("Record your message")
if audio:
    audio_bytes = audio.read()
    transcript = transcribe(audio_bytes)  # your ASR function
```

### Option B: `audio-recorder-streamlit`
Has auto-stop on silence detection.
```python
from audio_recorder_streamlit import audio_recorder

audio_bytes = audio_recorder(
    pause_threshold=2.0,  # seconds of silence to auto-stop
    sample_rate=16000
)
if audio_bytes:
    process_audio(audio_bytes)
```

### Option C: `streamlit-webrtc` (Real-time)
For live streaming; complex setup, needs STUN/TURN for deployment.
```python
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

def audio_callback(frame: av.AudioFrame):
    # Process audio chunk in real-time
    return frame

webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    audio_frame_callback=audio_callback,
    media_stream_constraints={"audio": True, "video": False}
)
```

**Recommendation**: Use `st.audio_input` for demo simplicity. Reserve webrtc for production real-time needs.

## 4. Session State Management

### Pattern: Pipeline State
```python
# Initialize all state at once
if "initialized" not in st.session_state:
    st.session_state.update({
        "messages": [],
        "current_audio": None,
        "pipeline_stage": "idle",  # idle | recording | processing | complete
        "transcript": "",
        "entities": [],
        "intent": None
    })
    st.session_state.initialized = True
```

### Pattern: Callback-based Updates
```python
def on_model_change():
    st.session_state.pipeline_stage = "idle"
    st.session_state.transcript = ""

st.selectbox("Model", options, on_change=on_model_change, key="model")
```

## 5. Pipeline Visualization

### Progress Bar with Text
```python
progress = st.progress(0, text="Initializing...")
for i, stage in enumerate(["ASR", "NER", "Intent", "Summary"]):
    progress.progress((i+1)*25, text=f"Processing: {stage}")
    run_stage(stage)
progress.empty()
```

### Expandable Stage Details
```python
with st.expander("ASR Output", expanded=True):
    st.text(st.session_state.transcript)
    st.caption(f"Confidence: {confidence:.2%}")

with st.expander("Named Entities"):
    for ent in st.session_state.entities:
        st.markdown(f"- **{ent['text']}** ({ent['label']})")
```

### Status Container for Long Tasks
```python
with st.status("Processing call...", expanded=True) as status:
    st.write("Transcribing audio...")
    transcript = transcribe(audio)

    st.write("Extracting entities...")
    entities = extract_ner(transcript)

    st.write("Classifying intent...")
    intent = classify_intent(transcript)

    status.update(label="Complete!", state="complete", expanded=False)
```

### Toast Notifications
```python
st.toast("Audio uploaded successfully")
st.toast("Processing complete", icon="check")
```

## 6. Vietnamese Text Display

### Font Configuration
Add to `.streamlit/config.toml`:
```toml
[theme]
font = "sans serif"
```

Or inject CSS for specific Vietnamese fonts:
```python
st.markdown("""
<style>
    .stMarkdown, .stText, .stChatMessage {
        font-family: 'Noto Sans', 'Roboto', sans-serif;
    }
</style>
""", unsafe_allow_html=True)
```

### Display Patterns
```python
# Use st.markdown for rich Vietnamese text
st.markdown("**Nội dung cuộc gọi:** Khach hang yeu cau ho tro")

# For code/technical output, use monospace
st.code("Entity: [LOC] Ha Noi", language=None)
```

### Handling Mixed Content
```python
# Vietnamese with entities highlighted
text = "Khach hang o **Ha Noi** can ho tro ve **bao hiem**"
st.markdown(text)
```

## 7. Recommended App Structure

```
app.py
├── Sidebar: Model selection, settings
├── Main Area:
│   ├── Header + Metrics (columns)
│   ├── Audio Input Section
│   ├── Pipeline Status (st.status)
│   └── Chat/Results Display
└── Session State: messages, pipeline_stage, results
```

### Minimal Skeleton
```python
import streamlit as st

st.set_page_config(page_title="AI Call Center", layout="wide")

# Sidebar
with st.sidebar:
    st.title("Settings")
    model = st.selectbox("ASR Model", ["PhoWhisper", "Whisper"])

# Main
st.title("AI Call Center Demo")

col1, col2 = st.columns([2, 1])
with col1:
    audio = st.audio_input("Record or upload audio")
    if audio and st.button("Process"):
        with st.status("Processing...") as status:
            # Pipeline stages
            status.update(label="Done", state="complete")

with col2:
    st.subheader("Results")
    with st.expander("Transcript", expanded=True):
        st.write(st.session_state.get("transcript", ""))
```

## Key Takeaways

| Component | Recommendation |
|-----------|----------------|
| Audio | `st.audio_input` (native, simple) |
| Chat | `st.chat_message` + `st.chat_input` |
| Progress | `st.status` for multi-step, `st.progress` for single |
| Layout | Sidebar for controls, columns for metrics |
| State | Single init block, callbacks for updates |
| Vietnamese | Default fonts work; inject CSS for custom |

## Dependencies
```
streamlit>=1.34.0
audio-recorder-streamlit  # optional, for silence detection
```
