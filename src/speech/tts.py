"""Text-to-Speech module with multiple provider backends.

Supports:
    1. "gtts"     — gTTS (free, good Vietnamese support) — default
    2. "edge_tts" — Microsoft Edge TTS (free, higher quality, async)
    3. "fallback" — No audio output

All providers target Vietnamese and fall back gracefully
if dependencies are missing.
"""

import io
import logging
import tempfile
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Vietnamese voice/language settings per provider
_GTTS_LANG = "vi"
_EDGE_TTS_VOICE = "vi-VN-HoaiMyNeural"


class TextToSpeech:
    """Text-to-Speech with multiple backend support.

    Usage::

        tts = TextToSpeech(provider="gtts")
        audio_bytes = tts.synthesize("Xin chào!")
        tts.save("Xin chào!", "greeting.mp3")
    """

    PROVIDERS = ("gtts", "edge_tts", "fallback")

    def __init__(self, provider: str = "gtts"):
        """Initialize TTS with the given provider.

        Args:
            provider: One of "gtts", "edge_tts", "fallback".
                      Falls back automatically if the chosen provider's
                      dependencies are not available.
        """
        if provider not in self.PROVIDERS:
            logger.warning("Unknown TTS provider '%s', falling back to 'gtts'", provider)
            provider = "gtts"

        self.provider = provider
        self._init_provider()

    def _init_provider(self) -> None:
        """Initialize the selected provider, falling back if needed."""
        if self.provider == "gtts":
            if not self._check_gtts():
                logger.warning("gTTS not available. Trying edge_tts.")
                self.provider = "edge_tts"
                self._init_provider()
                return

        elif self.provider == "edge_tts":
            if not self._check_edge_tts():
                logger.warning("edge-tts not available. Falling back to no-audio mode.")
                self.provider = "fallback"

        if self.provider == "fallback":
            logger.info("TTS running in fallback mode (no audio output).")

    @staticmethod
    def _check_gtts() -> bool:
        """Check if gTTS is importable."""
        try:
            import gtts  # noqa: F401

            logger.info("TTS provider: gtts")
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_edge_tts() -> bool:
        """Check if edge-tts is importable."""
        try:
            import edge_tts  # noqa: F401

            logger.info("TTS provider: edge_tts")
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, text: str) -> bytes:
        """Convert Vietnamese text to audio bytes (MP3).

        Args:
            text: Vietnamese text to synthesize.

        Returns:
            MP3 audio bytes, or empty bytes in fallback mode.
        """
        if not text or not text.strip():
            logger.warning("Empty text passed to TTS.")
            return b""

        if self.provider == "fallback":
            logger.debug("TTS in fallback mode — returning empty bytes.")
            return b""

        if self.provider == "gtts":
            return self._synthesize_gtts(text)

        if self.provider == "edge_tts":
            return self._synthesize_edge_tts(text)

        return b""

    def save(self, text: str, filepath: Union[str, Path]) -> Path:
        """Synthesize and save audio to a file.

        Args:
            text: Vietnamese text to synthesize.
            filepath: Output file path (MP3 recommended).

        Returns:
            Path to the saved audio file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        audio_bytes = self.synthesize(text)
        if not audio_bytes:
            logger.warning("No audio generated for: '%s'", text[:50])
            # Write empty file so callers don't get FileNotFoundError
            filepath.touch()
            return filepath

        filepath.write_bytes(audio_bytes)
        logger.info("Saved TTS audio: %s (%.1f KB)", filepath, len(audio_bytes) / 1024)
        return filepath

    @property
    def is_available(self) -> bool:
        """Whether TTS is functional (not in fallback mode)."""
        return self.provider != "fallback"

    # ------------------------------------------------------------------
    # gTTS
    # ------------------------------------------------------------------

    def _synthesize_gtts(self, text: str) -> bytes:
        """Synthesize using gTTS (Google Translate TTS)."""
        try:
            from gtts import gTTS

            tts = gTTS(text=text, lang=_GTTS_LANG, slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            audio_bytes = buf.getvalue()
            logger.debug("gTTS synthesized %d bytes for: '%s'", len(audio_bytes), text[:50])
            return audio_bytes
        except Exception as e:
            logger.error("gTTS synthesis error: %s", e)
            return b""

    # ------------------------------------------------------------------
    # Edge TTS (async → sync wrapper)
    # ------------------------------------------------------------------

    def _synthesize_edge_tts(self, text: str) -> bytes:
        """Synthesize using Edge TTS (async, wrapped for sync interface)."""
        try:
            import asyncio

            import edge_tts

            async def _generate() -> bytes:
                communicate = edge_tts.Communicate(text, _EDGE_TTS_VOICE)
                chunks = []
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        chunks.append(chunk["data"])
                return b"".join(chunks)

            # Handle running event loops (e.g., inside Jupyter/Streamlit)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Already in an async context — use nest_asyncio or thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _generate())
                    audio_bytes = future.result(timeout=30)
            else:
                audio_bytes = asyncio.run(_generate())

            logger.debug("Edge TTS synthesized %d bytes for: '%s'", len(audio_bytes), text[:50])
            return audio_bytes
        except Exception as e:
            logger.error("Edge TTS synthesis error: %s", e)
            return b""

    def __repr__(self) -> str:
        return f"TextToSpeech(provider='{self.provider}', available={self.is_available})"
