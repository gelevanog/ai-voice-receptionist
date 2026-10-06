"""PCM helpers: the pipeline works on mono float32 in [-1, 1]; transports speak 16-bit little-endian PCM."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt

Audio = npt.NDArray[np.float32]

PIPELINE_RATE = 16_000  # VAD and STT run at 16 kHz


def int16_bytes_to_float(data: bytes) -> Audio:
    samples = np.frombuffer(data, dtype="<i2").astype(np.float32)
    return samples / 32768.0


def float_to_int16(audio: Audio) -> npt.NDArray[np.int16]:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).round().astype(np.int16)


def float_to_int16_bytes(audio: Audio) -> bytes:
    return float_to_int16(audio).astype("<i2").tobytes()


def silence(seconds: float, rate: int = PIPELINE_RATE) -> Audio:
    return np.zeros(round(seconds * rate), dtype=np.float32)


def duration(audio: Audio, rate: int = PIPELINE_RATE) -> float:
    return len(audio) / rate


def rms_dbfs(audio: Audio) -> float:
    if len(audio) == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * float(np.log10(max(rms, 1e-6)))


def write_wav(path: Path | io.BytesIO, audio: Audio, rate: int, channels: int = 1) -> None:
    """Write float audio (shape (n,) or (n, channels)) as 16-bit PCM WAV."""
    pcm = float_to_int16(audio)
    with wave.open(path if isinstance(path, io.BytesIO) else str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.astype("<i2").tobytes())


def wav_bytes(audio: Audio, rate: int, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    write_wav(buffer, audio, rate, channels)
    return buffer.getvalue()


def read_wav(path: Path) -> tuple[Audio, int]:
    """Read a 16-bit PCM WAV as mono float32 (channels are averaged)."""
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())
    audio = int16_bytes_to_float(frames)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.float32)
    return audio, rate
