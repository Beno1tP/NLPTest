"""Unified speech service combining STT and TTS.

Provides a single entry point for all speech operations,
configured via a dictionary or the project's speech_config.yaml.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from .stt import SpeechToText
from .tts import TextToSpeech

logger = logging.getLogger(__name__)

# Default config path relative to project root
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "speech_config.yaml"


def _load_yaml_config(path: Union[str, Path]) -> dict:
    """Load YAML configuration file."""
    path = Path(path)
    if not path.exists():
        logger.warning("Speech config not found at %s, using defaults.", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _map_stt_provider(raw: str) -> str:
    """Map config provider names to SpeechToText provider names."""
    mapping = {
        "google_cloud": "google_free",  # config says google_cloud but we use free API
        "google_free": "google_free",
        "speech_recognition": "google_free",
        "whisper_api": "whisper_api",
        "whisper": "whisper_api",
        "fallback": "fallback",
    }
    return mapping.get(raw, "google_free")


def _map_tts_provider(raw: str) -> str:
    """Map config provider names to TextToSpeech provider names."""
    mapping = {
        "google_cloud": "gtts",  # config says google_cloud but we use gTTS
        "gtts": "gtts",
        "edge_tts": "edge_tts",
        "fallback": "fallback",
    }
    return mapping.get(raw, "gtts")


class SpeechService:
    """Unified speech service wrapping STT and TTS.

    Usage::

        # From config dict
        service = SpeechService({"stt_provider": "google_free", "tts_provider": "gtts"})

        # From YAML config file
        service = SpeechService.from_config("configs/speech_config.yaml")

        # Default (auto-detect from project config)
        service = SpeechService()

        text = service.speech_to_text(audio_bytes)
        audio = service.text_to_speech("Xin chào!")
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize speech service.

        Args:
            config: Configuration dict. Supported keys:
                - stt_provider: STT provider name (default: "google_free")
                - tts_provider: TTS provider name (default: "gtts")
                If None, loads from configs/speech_config.yaml.
        """
        if config is None:
            config = self._load_default_config()

        stt_provider = config.get("stt_provider", "google_free")
        tts_provider = config.get("tts_provider", "gtts")

        self.stt = SpeechToText(provider=stt_provider)
        self.tts = TextToSpeech(provider=tts_provider)

        logger.info(
            "SpeechService initialized: STT=%s, TTS=%s",
            self.stt.provider,
            self.tts.provider,
        )

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "SpeechService":
        """Create SpeechService from a YAML config file.

        Args:
            config_path: Path to speech_config.yaml.

        Returns:
            Configured SpeechService instance.
        """
        raw = _load_yaml_config(config_path)
        config = cls._parse_yaml_config(raw)
        return cls(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speech_to_text(self, audio: Union[str, Path, bytes]) -> str:
        """Transcribe audio to Vietnamese text.

        Args:
            audio: Audio file path or raw bytes.

        Returns:
            Transcribed text, or empty string on failure.
        """
        return self.stt.transcribe(audio)

    def text_to_speech(self, text: str) -> bytes:
        """Convert Vietnamese text to audio bytes (MP3).

        Args:
            text: Vietnamese text string.

        Returns:
            MP3 audio bytes, or empty bytes on failure.
        """
        return self.tts.synthesize(text)

    def speech_to_text_from_mic(self, duration: float = 5.0) -> str:
        """Record from microphone and transcribe.

        Args:
            duration: Recording duration in seconds.

        Returns:
            Transcribed text.
        """
        return self.stt.transcribe_from_mic(duration=duration)

    def save_speech(self, text: str, filepath: Union[str, Path]) -> Path:
        """Synthesize and save speech audio to file.

        Args:
            text: Vietnamese text.
            filepath: Output file path.

        Returns:
            Path to saved audio file.
        """
        return self.tts.save(text, filepath)

    @property
    def stt_available(self) -> bool:
        """Whether STT is functional."""
        return self.stt.is_available

    @property
    def tts_available(self) -> bool:
        """Whether TTS is functional."""
        return self.tts.is_available

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_yaml_config(raw: dict) -> dict:
        """Parse the project's speech_config.yaml into a flat config dict."""
        config = {}

        # STT provider
        stt_section = raw.get("stt", {})
        raw_stt_provider = stt_section.get("provider", "google_free")
        config["stt_provider"] = _map_stt_provider(raw_stt_provider)

        # TTS provider
        tts_section = raw.get("tts", {})
        raw_tts_provider = tts_section.get("provider", "gtts")
        config["tts_provider"] = _map_tts_provider(raw_tts_provider)

        return config

    @classmethod
    def _load_default_config(cls) -> dict:
        """Load default config from project's speech_config.yaml."""
        if _DEFAULT_CONFIG_PATH.exists():
            raw = _load_yaml_config(_DEFAULT_CONFIG_PATH)
            return cls._parse_yaml_config(raw)
        return {"stt_provider": "google_free", "tts_provider": "gtts"}

    def __repr__(self) -> str:
        return (
            f"SpeechService(stt={self.stt.provider!r}, tts={self.tts.provider!r}, "
            f"stt_ok={self.stt_available}, tts_ok={self.tts_available})"
        )
