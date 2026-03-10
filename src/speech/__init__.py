"""Speech module — STT, TTS, audio utilities, and unified service.

Exports:
    SpeechToText    — Speech-to-Text (google_free / whisper_api / fallback)
    TextToSpeech    — Text-to-Speech (gtts / edge_tts / fallback)
    SpeechService   — Unified STT+TTS service with config support
    audio_utils     — WAV/MP3 conversion, sample rate, duration, validation
"""

from .speech_service import SpeechService
from .stt import SpeechToText
from .tts import TextToSpeech

__all__ = [
    "SpeechToText",
    "TextToSpeech",
    "SpeechService",
]
