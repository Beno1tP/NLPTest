"""Audio utility functions for format conversion and preprocessing.

Uses pydub for audio manipulation with graceful fallback
if ffmpeg/pydub is not available.
"""

import io
import logging
import struct
import wave
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Target sample rate for STT engines
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("pydub not installed. Audio conversion features limited.")


def wav_to_mp3(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    bitrate: str = "128k",
) -> Path:
    """Convert WAV file to MP3.

    Args:
        input_path: Path to input WAV file.
        output_path: Path for output MP3. Defaults to same name with .mp3 extension.
        bitrate: MP3 bitrate (default "128k").

    Returns:
        Path to the output MP3 file.

    Raises:
        RuntimeError: If pydub is not available.
        FileNotFoundError: If input file doesn't exist.
    """
    if not PYDUB_AVAILABLE:
        raise RuntimeError("pydub is required for WAV→MP3 conversion. Install: pip install pydub")

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_suffix(".mp3")
    output_path = Path(output_path)

    audio = AudioSegment.from_wav(str(input_path))
    audio.export(str(output_path), format="mp3", bitrate=bitrate)
    logger.info("Converted WAV→MP3: %s → %s", input_path, output_path)
    return output_path


def mp3_to_wav(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
) -> Path:
    """Convert MP3 file to WAV (mono, 16kHz by default for STT).

    Args:
        input_path: Path to input MP3 file.
        output_path: Path for output WAV. Defaults to same name with .wav extension.
        sample_rate: Target sample rate in Hz.
        channels: Number of audio channels.

    Returns:
        Path to the output WAV file.

    Raises:
        RuntimeError: If pydub is not available.
        FileNotFoundError: If input file doesn't exist.
    """
    if not PYDUB_AVAILABLE:
        raise RuntimeError("pydub is required for MP3→WAV conversion. Install: pip install pydub")

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_suffix(".wav")
    output_path = Path(output_path)

    audio = AudioSegment.from_mp3(str(input_path))
    audio = audio.set_frame_rate(sample_rate).set_channels(channels)
    audio.export(str(output_path), format="wav")
    logger.info("Converted MP3→WAV: %s → %s (rate=%d, ch=%d)", input_path, output_path, sample_rate, channels)
    return output_path


def convert_sample_rate(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    target_rate: int = TARGET_SAMPLE_RATE,
    target_channels: int = TARGET_CHANNELS,
) -> Path:
    """Convert audio file to target sample rate and channel count.

    Supports any format pydub/ffmpeg can read.

    Args:
        input_path: Path to input audio file.
        output_path: Path for output file. Defaults to overwriting input.
        target_rate: Target sample rate in Hz.
        target_channels: Target number of channels.

    Returns:
        Path to the output file.
    """
    if not PYDUB_AVAILABLE:
        raise RuntimeError("pydub is required for sample rate conversion. Install: pip install pydub")

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = input_path

    output_path = Path(output_path)
    suffix = output_path.suffix.lstrip(".")
    fmt = suffix if suffix else "wav"

    audio = AudioSegment.from_file(str(input_path))
    audio = audio.set_frame_rate(target_rate).set_channels(target_channels)
    audio.export(str(output_path), format=fmt)
    logger.info("Resampled: %s → %dHz, %dch", output_path, target_rate, target_channels)
    return output_path


def get_audio_duration(audio_input: Union[str, Path, bytes]) -> float:
    """Get audio duration in seconds.

    Args:
        audio_input: File path or raw audio bytes.

    Returns:
        Duration in seconds.
    """
    if isinstance(audio_input, bytes):
        return _duration_from_bytes(audio_input)

    path = Path(audio_input)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if PYDUB_AVAILABLE:
        audio = AudioSegment.from_file(str(path))
        return len(audio) / 1000.0

    # Fallback: WAV-only using stdlib
    if path.suffix.lower() == ".wav":
        return _wav_duration(path)

    raise RuntimeError(f"pydub required for non-WAV duration check. File: {path}")


def _wav_duration(path: Path) -> float:
    """Get WAV duration using stdlib wave module."""
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate) if rate > 0 else 0.0


def _duration_from_bytes(audio_bytes: bytes) -> float:
    """Get duration from audio bytes (tries WAV header first, then pydub)."""
    # Try WAV header
    if audio_bytes[:4] == b"RIFF":
        try:
            buf = io.BytesIO(audio_bytes)
            with wave.open(buf, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / float(rate) if rate > 0 else 0.0
        except wave.Error:
            pass

    if PYDUB_AVAILABLE:
        buf = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_file(buf)
        return len(audio) / 1000.0

    raise RuntimeError("Cannot determine duration from bytes without pydub.")


def audio_bytes_to_wav(
    audio_bytes: bytes,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
) -> bytes:
    """Convert arbitrary audio bytes to WAV format suitable for STT.

    Args:
        audio_bytes: Raw audio data in any format.
        sample_rate: Target sample rate.
        channels: Target channel count.

    Returns:
        WAV-formatted bytes.
    """
    if not PYDUB_AVAILABLE:
        # If already WAV, return as-is
        if audio_bytes[:4] == b"RIFF":
            return audio_bytes
        raise RuntimeError("pydub required to convert non-WAV audio bytes.")

    buf_in = io.BytesIO(audio_bytes)
    audio = AudioSegment.from_file(buf_in)
    audio = audio.set_frame_rate(sample_rate).set_channels(channels)

    buf_out = io.BytesIO()
    audio.export(buf_out, format="wav")
    return buf_out.getvalue()


def validate_audio(
    audio_input: Union[str, Path, bytes],
    max_duration: float = 30.0,
) -> dict:
    """Validate audio input and return metadata.

    Args:
        audio_input: File path or raw audio bytes.
        max_duration: Maximum allowed duration in seconds.

    Returns:
        Dict with keys: valid, duration, error (if invalid).
    """
    try:
        duration = get_audio_duration(audio_input)
        if duration <= 0:
            return {"valid": False, "duration": 0.0, "error": "Audio has zero duration"}
        if duration > max_duration:
            return {
                "valid": False,
                "duration": duration,
                "error": f"Audio too long: {duration:.1f}s (max {max_duration:.0f}s)",
            }
        return {"valid": True, "duration": duration, "error": None}
    except Exception as e:
        return {"valid": False, "duration": 0.0, "error": str(e)}
