"""Vietnamese AI Call Center - Streamlit Demo Application.

Interactive demo showcasing the NLU dialogue system with voice and text modes.

Run with:
    streamlit run app/streamlit_app.py

Layout:
    +----------------------------------------------------+
    |  Vietnamese AI Call Center                          |
    |  [Voice Mode] [Text Mode]                          |
    +------------------------+---------------------------+
    |                        | Pipeline Visualization    |
    |   Conversation         |                           |
    |                        | STT: "dat ve di da nang"  |
    |   User: ...            | NLU:                      |
    |   Bot: ...             |   Intent: flight 96%      |
    |   User: ...            |   Slots: {toloc: Da Nang} |
    |   Bot: ...             | State: {toloc: Da Nang}   |
    |                        | Policy: request_slot      |
    |   [Playing...]         | NLG: "Bay tu dau a?"      |
    |                        |                           |
    +------------------------+---------------------------+
    |  [Record] or [Type message...] [Send]              |
    +----------------------------------------------------+
"""

import base64
import io
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Vietnamese AI Call Center",
    page_icon="telephone_receiver",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }

    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    .chat-message.bot {
        background-color: #f5f5f5;
        margin-right: 2rem;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        margin-right: 0.5rem;
    }
    .chat-message.user .avatar {
        background-color: #1976d2;
        color: white;
    }
    .chat-message.bot .avatar {
        background-color: #388e3c;
        color: white;
    }

    /* Pipeline visualization */
    .pipeline-box {
        background-color: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .pipeline-box h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        color: #333;
    }

    /* Confidence bar */
    .confidence-bar {
        height: 8px;
        background-color: #e0e0e0;
        border-radius: 4px;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background-color: #4caf50;
        transition: width 0.3s ease;
    }

    /* Slot chips */
    .slot-chip {
        display: inline-block;
        background-color: #e8f5e9;
        border: 1px solid #81c784;
        border-radius: 1rem;
        padding: 0.25rem 0.75rem;
        margin: 0.125rem;
        font-size: 0.85rem;
    }

    /* Header styling */
    .app-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Audio player styling */
    audio {
        width: 100%;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Model Loading (cached)
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_svm_model():
    """Load SVM NLU model."""
    try:
        from src.nlu import SVMNLU
        model_dir = PROJECT_ROOT / "models" / "svm_nlu"
        if model_dir.exists():
            return SVMNLU.load(str(model_dir))
        logger.warning("SVM model not found at %s", model_dir)
        return None
    except Exception as e:
        logger.error("Failed to load SVM model: %s", e)
        return None


@st.cache_resource(show_spinner=False)
def load_jointbert_model():
    """Load JointBERT NLU model."""
    try:
        from src.nlu import JointBERTNLU
        model_path = PROJECT_ROOT / "models" / "best_jointbert.pt"
        data_dir = PROJECT_ROOT / "data" / "processed"

        if model_path.exists() and data_dir.exists():
            return JointBERTNLU(
                model_path=str(model_path),
                data_dir=data_dir,
            )
        logger.warning("JointBERT model not found at %s", model_path)
        return None
    except Exception as e:
        logger.error("Failed to load JointBERT model: %s", e)
        return None


@st.cache_resource(show_spinner=False)
def load_llm_model():
    """Load LLM NLU classifier."""
    try:
        from src.nlu import create_llm_classifier
        # Use mock by default, can switch to anthropic/openai if API key available
        import os
        if os.environ.get("ANTHROPIC_API_KEY"):
            return create_llm_classifier(provider="anthropic")
        elif os.environ.get("OPENAI_API_KEY"):
            return create_llm_classifier(provider="openai")
        else:
            return create_llm_classifier(provider="mock")
    except Exception as e:
        logger.error("Failed to load LLM model: %s", e)
        return None


@st.cache_resource(show_spinner=False)
def load_speech_service():
    """Load speech service (STT + TTS)."""
    try:
        from src.speech.speech_service import SpeechService
        return SpeechService()
    except Exception as e:
        logger.error("Failed to load speech service: %s", e)
        return None


@st.cache_resource(show_spinner=False)
def get_pipeline(_nlu_model, _speech_service=None, config=None):
    """Create dialogue pipeline with given NLU model.

    Note: Leading underscore in _nlu_model and _speech_service prevents
    Streamlit from trying to hash these objects.
    """
    try:
        from src.pipeline.orchestrator import DialoguePipeline, PipelineConfig

        pipeline_config = None
        if config:
            pipeline_config = PipelineConfig(
                confidence_threshold=config.get("confidence_threshold", 0.5),
                enable_tts=config.get("enable_tts", True),
                enable_stt=config.get("enable_stt", True),
            )

        stt = _speech_service.stt if _speech_service else None
        tts = _speech_service.tts if _speech_service else None

        return DialoguePipeline(
            nlu_model=_nlu_model,
            stt=stt,
            tts=tts,
            config=pipeline_config,
        )
    except Exception as e:
        logger.error("Failed to create pipeline: %s", e)
        return None


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------

def init_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pipeline_data" not in st.session_state:
        st.session_state.pipeline_data = {
            "stt_transcript": "",
            "nlu_output": None,
            "state": None,
            "action": None,
            "nlg_response": "",
        }

    if "mode" not in st.session_state:
        st.session_state.mode = "text"  # "text" or "voice"

    if "model_type" not in st.session_state:
        # Default to Demo Mode if JointBERT not available
        jointbert_available = load_jointbert_model() is not None
        st.session_state.model_type = "JointBERT" if jointbert_available else "Demo Mode (Mock)"

    if "tts_enabled" not in st.session_state:
        st.session_state.tts_enabled = True

    if "confidence_threshold" not in st.session_state:
        st.session_state.confidence_threshold = 0.5

    if "current_audio" not in st.session_state:
        st.session_state.current_audio = None

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None


def get_nlu_model(model_type: str):
    """Get NLU model by type."""
    if model_type == "SVM Baseline":
        return load_svm_model()
    elif model_type == "JointBERT":
        return load_jointbert_model()
    elif model_type == "LLM Zero-Shot":
        return load_llm_model()
    elif model_type == "Demo Mode (Mock)":
        # Return None to use pipeline's built-in mock NLU
        return None
    return None


def check_model_availability() -> Dict[str, bool]:
    """Check which models are available."""
    return {
        "JointBERT": load_jointbert_model() is not None,
        "SVM Baseline": load_svm_model() is not None,
        "LLM Zero-Shot": load_llm_model() is not None,
        "Demo Mode (Mock)": True,  # Always available
    }


# -----------------------------------------------------------------------------
# UI Components
# -----------------------------------------------------------------------------

def render_header():
    """Render application header."""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("## Vietnamese AI Call Center")
        st.markdown("*Demo NLU dialogue system for air travel domain*")

    with col2:
        # Mode selection
        mode = st.radio(
            "Input Mode",
            ["Text", "Voice"],
            horizontal=True,
            key="mode_radio",
        )
        st.session_state.mode = mode.lower()


def render_sidebar():
    """Render sidebar with settings."""
    with st.sidebar:
        st.markdown("### Settings")

        # Model selection
        st.markdown("#### NLU Model")

        # Check model availability
        availability = check_model_availability()

        # Build model options with availability info
        all_models = ["JointBERT", "SVM Baseline", "LLM Zero-Shot", "Demo Mode (Mock)"]
        model_options = []
        for model in all_models:
            if availability.get(model, False):
                model_options.append(model)
            else:
                model_options.append(f"{model} (not available)")

        # Find current index (handle unavailable models)
        current_model = st.session_state.model_type
        try:
            current_index = all_models.index(current_model)
        except ValueError:
            current_index = all_models.index("Demo Mode (Mock)")

        model_type = st.selectbox(
            "Select Model",
            model_options,
            index=current_index,
            key="model_select",
        )

        # Extract actual model name (remove " (not available)" suffix)
        actual_model = model_type.replace(" (not available)", "")

        if actual_model != st.session_state.model_type:
            # Check if model is actually available
            if availability.get(actual_model, False):
                st.session_state.model_type = actual_model
                st.session_state.pipeline = None  # Force pipeline recreation
                st.rerun()
            else:
                st.error(f"{actual_model} is not available. Please train the model first.")

        # Model status
        if availability.get(st.session_state.model_type, False):
            if st.session_state.model_type == "Demo Mode (Mock)":
                st.info("Demo Mode: Using built-in mock NLU (keyword-based)")
            else:
                st.success(f"{st.session_state.model_type} loaded")
        else:
            st.warning(f"{st.session_state.model_type} not available")

        st.markdown("---")

        # TTS toggle
        st.markdown("#### Audio Settings")
        tts_enabled = st.toggle(
            "Enable TTS (Text-to-Speech)",
            value=st.session_state.tts_enabled,
            key="tts_toggle",
        )
        st.session_state.tts_enabled = tts_enabled

        # Confidence threshold
        st.markdown("#### Confidence Threshold")
        threshold = st.slider(
            "Min confidence for intent",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.confidence_threshold,
            step=0.05,
            key="threshold_slider",
        )
        st.session_state.confidence_threshold = threshold

        st.markdown("---")

        # Reset button
        if st.button("Reset Conversation", type="secondary", use_container_width=True):
            reset_conversation()
            st.rerun()

        st.markdown("---")

        # System info
        with st.expander("System Info"):
            speech_service = load_speech_service()
            st.markdown(f"- **STT Provider**: {speech_service.stt.provider if speech_service else 'N/A'}")
            st.markdown(f"- **TTS Provider**: {speech_service.tts.provider if speech_service else 'N/A'}")
            st.markdown(f"- **STT Available**: {speech_service.stt_available if speech_service else False}")
            st.markdown(f"- **TTS Available**: {speech_service.tts_available if speech_service else False}")


def reset_conversation():
    """Reset conversation state."""
    st.session_state.messages = []
    st.session_state.pipeline_data = {
        "stt_transcript": "",
        "nlu_output": None,
        "state": None,
        "action": None,
        "nlg_response": "",
    }
    st.session_state.current_audio = None
    if st.session_state.pipeline:
        st.session_state.pipeline.reset()


def render_chat_message(role: str, content: str, audio_bytes: Optional[bytes] = None):
    """Render a chat message."""
    if role == "user":
        avatar = "person"
        name = "Khach"
    else:
        avatar = "robot_face"
        name = "Bot"

    with st.chat_message(role, avatar=avatar):
        st.markdown(content)

        # Play audio if available
        if audio_bytes and len(audio_bytes) > 0:
            st.audio(audio_bytes, format="audio/mp3")


def render_conversation():
    """Render conversation history."""
    for msg in st.session_state.messages:
        render_chat_message(
            role=msg["role"],
            content=msg["content"],
            audio_bytes=msg.get("audio"),
        )


def render_pipeline_visualization():
    """Render pipeline state visualization."""
    data = st.session_state.pipeline_data

    st.markdown("### Pipeline Visualization")

    # STT Transcript (if voice mode)
    if st.session_state.mode == "voice" and data.get("stt_transcript"):
        st.markdown("#### STT Transcript")
        st.info(f'"{data["stt_transcript"]}"')

    # NLU Output
    st.markdown("#### NLU Output")
    nlu = data.get("nlu_output")
    if nlu:
        intent = nlu.get("intent", "unknown")
        confidence = nlu.get("confidence", 0.0)
        slots = nlu.get("slots", {})

        # Intent with confidence bar
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Intent**: `{intent}`")
        with col2:
            st.markdown(f"**{confidence:.0%}**")

        # Confidence bar
        st.progress(confidence)

        # Slots
        if slots:
            st.markdown("**Slots**:")
            slot_html = " ".join([
                f'<span class="slot-chip"><b>{k}</b>: {v}</span>'
                for k, v in slots.items()
            ])
            st.markdown(slot_html, unsafe_allow_html=True)
        else:
            st.markdown("*No slots detected*")
    else:
        st.markdown("*Waiting for input...*")

    st.markdown("---")

    # Dialogue State
    st.markdown("#### Belief State (DST)")
    state = data.get("state")
    if state:
        slots = state.get("slots", {})
        if slots:
            for slot_type, value in slots.items():
                st.markdown(f"- **{slot_type}**: {value}")
        else:
            st.markdown("*Empty state*")

        # Show turn count
        st.caption(f"Turn: {state.get('turn_count', 0)}")
    else:
        st.markdown("*No state yet*")

    st.markdown("---")

    # Policy Action
    st.markdown("#### Policy Action")
    action = data.get("action")
    if action:
        action_type = action.get("type", "unknown")
        st.markdown(f"**Action**: `{action_type}`")

        if action_type == "request_slot":
            st.markdown(f"**Requesting**: `{action.get('slot', 'N/A')}`")
        elif action_type == "confirm":
            st.markdown("**Confirming parameters**")
        elif action_type == "execute":
            st.markdown("**Executing booking**")
    else:
        st.markdown("*No action yet*")

    st.markdown("---")

    # NLG Response
    st.markdown("#### NLG Response")
    if data.get("nlg_response"):
        st.success(data["nlg_response"])
    else:
        st.markdown("*No response yet*")


def process_user_input(user_input: str, is_audio: bool = False, audio_bytes: bytes = None):
    """Process user input through the pipeline."""
    # Get or create pipeline
    nlu_model = get_nlu_model(st.session_state.model_type)
    speech_service = load_speech_service() if st.session_state.tts_enabled else None

    # For Demo Mode, nlu_model is None intentionally (uses pipeline's mock NLU)
    if nlu_model is None and st.session_state.model_type != "Demo Mode (Mock)":
        st.error(f"NLU model ({st.session_state.model_type}) not available. Please check model files.")
        return

    # Create pipeline if needed
    if st.session_state.pipeline is None:
        config = {
            "confidence_threshold": st.session_state.confidence_threshold,
            "enable_tts": st.session_state.tts_enabled,
            "enable_stt": is_audio,
        }
        st.session_state.pipeline = get_pipeline(nlu_model, speech_service, config)

    pipeline = st.session_state.pipeline
    if not pipeline:
        st.error("Failed to create dialogue pipeline.")
        return

    # Process input
    try:
        if is_audio and audio_bytes:
            result = pipeline.process_audio(audio_bytes)
            transcript = result.transcript or user_input
            st.session_state.pipeline_data["stt_transcript"] = transcript
        else:
            result = pipeline.process(user_input)
            st.session_state.pipeline_data["stt_transcript"] = ""

        # Update pipeline visualization data
        st.session_state.pipeline_data["nlu_output"] = result.nlu_output
        st.session_state.pipeline_data["state"] = result.state
        st.session_state.pipeline_data["action"] = result.action
        st.session_state.pipeline_data["nlg_response"] = result.response

        # Add messages to history
        st.session_state.messages.append({
            "role": "user",
            "content": result.transcript if is_audio and result.transcript else user_input,
        })

        st.session_state.messages.append({
            "role": "assistant",
            "content": result.response,
            "audio": result.audio_response if st.session_state.tts_enabled else None,
        })

        # Store current audio for playback
        if result.audio_response and st.session_state.tts_enabled:
            st.session_state.current_audio = result.audio_response

    except Exception as e:
        logger.error("Error processing input: %s", e)
        st.error(f"Error: {str(e)}")


def render_voice_input():
    """Render voice input controls."""
    st.markdown("### Voice Input")

    # Check if audio recorder is available
    try:
        from streamlit_audiorecorder import audiorecorder

        st.markdown("*Click the microphone to start recording, click again to stop.*")

        audio = audiorecorder(
            "Click to record",
            "Recording... Click to stop",
            key="audio_recorder",
        )

        if len(audio) > 0:
            # Convert to bytes
            audio_bytes = audio.export().read()

            # Show recorded audio
            st.audio(audio_bytes, format="audio/wav")

            # Process button
            if st.button("Process Recording", type="primary"):
                with st.spinner("Processing audio..."):
                    process_user_input("", is_audio=True, audio_bytes=audio_bytes)
                st.rerun()

    except ImportError:
        st.warning("Audio recorder not available. Please install: `pip install streamlit-audiorecorder`")

        # Fallback: file upload
        st.markdown("**Alternative: Upload audio file**")
        uploaded_file = st.file_uploader(
            "Upload audio file (WAV, MP3)",
            type=["wav", "mp3", "m4a"],
            key="audio_upload",
        )

        if uploaded_file:
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes)

            if st.button("Process Audio", type="primary"):
                with st.spinner("Processing audio..."):
                    process_user_input("", is_audio=True, audio_bytes=audio_bytes)
                st.rerun()


def render_text_input():
    """Render text input controls."""
    # Use chat input for text mode
    user_input = st.chat_input(
        "Nhap tin nhan... (e.g., 'Toi muon dat ve di Da Nang')",
        key="chat_input",
    )

    if user_input:
        with st.spinner("Processing..."):
            process_user_input(user_input)
        st.rerun()


def render_quick_actions():
    """Render quick action buttons for testing."""
    st.markdown("### Quick Test Phrases")

    test_phrases = [
        "Xin chao",
        "Toi muon dat ve di Da Nang",
        "Bay tu Ha Noi den Ho Chi Minh",
        "Gia ve bao nhieu",
        "Dung roi",
    ]

    cols = st.columns(len(test_phrases))
    for i, phrase in enumerate(test_phrases):
        with cols[i]:
            if st.button(phrase, key=f"quick_{i}", use_container_width=True):
                with st.spinner("Processing..."):
                    process_user_input(phrase)
                st.rerun()


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------

def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()

    # Render sidebar
    render_sidebar()

    # Render header
    render_header()

    # Main layout: conversation + pipeline visualization
    col_chat, col_pipeline = st.columns([3, 2])

    with col_chat:
        st.markdown("### Conversation")

        # Conversation container
        chat_container = st.container(height=400)
        with chat_container:
            if not st.session_state.messages:
                st.info("Start a conversation by typing a message or using voice input.")
            else:
                render_conversation()

        st.markdown("---")

        # Input area
        if st.session_state.mode == "voice":
            render_voice_input()
        else:
            render_text_input()

        # Quick actions
        with st.expander("Quick Test Phrases", expanded=False):
            render_quick_actions()

    with col_pipeline:
        render_pipeline_visualization()

    # Footer
    st.markdown("---")
    st.caption(
        "Vietnamese AI Call Center Demo | "
        "NLP Course Project | "
        "Models: SVM, JointBERT (PhoBERT), LLM Zero-Shot"
    )


if __name__ == "__main__":
    main()
