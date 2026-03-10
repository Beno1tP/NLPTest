"""Speech-to-Text module with multiple provider backends.

Supports:
    1. "google_free" — SpeechRecognition library with free Google Web Speech API (default)
    2. "whisper_api" — OpenAI Whisper API (requires API key)
    3. "fallback"    — No transcription, forces text input mode

All providers handle Vietnamese (vi-VN) and fail gracefully
if dependencies are missing.
"""

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Vietnamese language codes per provider
_LANG_GOOGLE = "vi-VN"
_LANG_WHISPER = "vi"


class SpeechToText:
    """Speech-to-Text with multiple backend support.

    Usage::

        stt = SpeechToText(provider="google_free")
        text = stt.transcribe("recording.wav")
        text = stt.transcribe(audio_bytes)
        text = stt.transcribe_from_mic(duration=5)
    """

    PROVIDERS = ("google_free", "whisper_api", "fallback")

    def __init__(self, provider: str = "google_free"):
        """Initialize STT with the given provider.

        Args:
            provider: One of "google_free", "whisper_api", "fallback".
                      Falls back automatically if the chosen provider's
                      dependencies are not available.
        """
        if provider not in self.PROVIDERS:
            logger.warning("Unknown STT provider '%s', falling back to 'google_free'", provider)
            provider = "google_free"

        self.provider = provider
        self._recognizer = None
        self._openai_client = None

        self._init_provider()

    def _init_provider(self) -> None:
        """Initialize the selected provider, falling back if needed."""
        if self.provider == "google_free":
            if not self._init_google_free():
                logger.warning("SpeechRecognition not available. Trying whisper_api.")
                self.provider = "whisper_api"
                self._init_provider()
                return

        elif self.provider == "whisper_api":
            if not self._init_whisper_api():
                logger.warning("OpenAI Whisper API not available. Falling back to text-only mode.")
                self.provider = "fallback"

        if self.provider == "fallback":
            logger.info("STT running in fallback mode (no speech recognition).")

    def _init_google_free(self) -> bool:
        """Try to initialize SpeechRecognition."""
        try:
            import speech_recognition as sr

            self._recognizer = sr.Recognizer()
            logger.info("STT provider: google_free (SpeechRecognition)")
            return True
        except ImportError:
            return False

    def _init_whisper_api(self) -> bool:
        """Try to initialize OpenAI Whisper API client."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.debug("OPENAI_API_KEY not set.")
            return False
        try:
            import openai

            self._openai_client = openai.OpenAI(api_key=api_key)
            logger.info("STT provider: whisper_api (OpenAI Whisper)")
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_input: Union[str, Path, bytes]) -> str:
        """Transcribe audio to Vietnamese text.

        Args:
            audio_input: File path (str/Path) or raw audio bytes.

        Returns:
            Transcribed text, or empty string on failure / fallback mode.
        """
        if self.provider == "fallback":
            logger.debug("STT in fallback mode — returning empty string.")
            return ""

        if self.provider == "google_free":
            return self._transcribe_google(audio_input)

        if self.provider == "whisper_api":
            return self._transcribe_whisper(audio_input)

        return ""

    def transcribe_from_mic(self, duration: float = 5.0) -> str:
        """Record from microphone and transcribe.

        Args:
            duration: Recording duration in seconds.

        Returns:
            Transcribed text, or empty string on failure.

        Note:
            Requires PyAudio to be installed. Import is deferred
            to avoid module-level dependency.
        """
        if self.provider == "fallback":
            logger.info("STT fallback mode — mic recording not available.")
            return ""

        if self._recognizer is None:
            logger.error("Microphone recording requires SpeechRecognition (google_free provider).")
            return ""

        try:
            import speech_recognition as sr

            # PyAudio is imported inside sr.Microphone — not at module level
            with sr.Microphone(sample_rate=16000) as source:
                logger.info("Adjusting for ambient noise...")
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
                logger.info("Recording for %.1f seconds...", duration)
                audio = self._recognizer.record(source, duration=duration)

            return self._recognize_google(audio)
        except OSError as e:
            logger.error("Microphone not available: %s", e)
            return ""
        except Exception as e:
            logger.error("Mic transcription error: %s", e)
            return ""

    @property
    def is_available(self) -> bool:
        """Whether STT is functional (not in fallback mode)."""
        return self.provider != "fallback"

    # ------------------------------------------------------------------
    # Google Free (SpeechRecognition)
    # ------------------------------------------------------------------

    def _transcribe_google(self, audio_input: Union[str, Path, bytes]) -> str:
        """Transcribe using SpeechRecognition + free Google Web Speech API."""
        import speech_recognition as sr

        try:
            audio_data = self._load_audio_for_sr(audio_input)
            return self._recognize_google(audio_data)
        except Exception as e:
            logger.error("Google STT error: %s", e)
            return ""

    def _load_audio_for_sr(self, audio_input: Union[str, Path, bytes]):
        """Load audio into SpeechRecognition AudioData format."""
        import speech_recognition as sr

        if isinstance(audio_input, (str, Path)):
            path = Path(audio_input)
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {path}")

            with sr.AudioFile(str(path)) as source:
                return self._recognizer.record(source)

        # Raw bytes — write to temp WAV for SpeechRecognition
        audio_bytes = self._ensure_wav_bytes(audio_input)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            with sr.AudioFile(tmp.name) as source:
                return self._recognizer.record(source)

    def _recognize_google(self, audio_data) -> str:
        """Run Google Web Speech API recognition."""
        import speech_recognition as sr

        try:
            text = self._recognizer.recognize_google(audio_data, language=_LANG_GOOGLE)
            logger.info("Transcribed (google_free): %s", text[:80])
            return text
        except sr.UnknownValueError:
            logger.warning("Google STT could not understand the audio.")
            return ""
        except sr.RequestError as e:
            logger.error("Google STT API request failed: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Whisper API (OpenAI)
    # ------------------------------------------------------------------

    def _transcribe_whisper(self, audio_input: Union[str, Path, bytes]) -> str:
        """Transcribe using OpenAI Whisper API."""
        try:
            if isinstance(audio_input, (str, Path)):
                path = Path(audio_input)
                if not path.exists():
                    raise FileNotFoundError(f"Audio file not found: {path}")
                with open(path, "rb") as f:
                    response = self._openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        language=_LANG_WHISPER,
                    )
            else:
                # Bytes — wrap in a file-like object with a name
                buf = io.BytesIO(audio_input)
                buf.name = "audio.wav"
                response = self._openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=buf,
                    language=_LANG_WHISPER,
                )

            text = response.text.strip()
            logger.info("Transcribed (whisper_api): %s", text[:80])
            return text
        except Exception as e:
            logger.error("Whisper API error: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_wav_bytes(audio_bytes: bytes) -> bytes:
        """Ensure audio bytes are in WAV format for SpeechRecognition."""
        # Already WAV?
        if audio_bytes[:4] == b"RIFF":
            return audio_bytes

        # Try converting via pydub
        try:
            from .audio_utils import audio_bytes_to_wav

            return audio_bytes_to_wav(audio_bytes)
        except Exception:
            logger.warning("Could not convert audio bytes to WAV. Passing as-is.")
            return audio_bytes

    def __repr__(self) -> str:
        return f"SpeechToText(provider='{self.provider}', available={self.is_available})"
