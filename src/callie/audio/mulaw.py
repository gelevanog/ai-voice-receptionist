"""G.711 μ-law codec (what Twilio Media Streams carry: `audio/x-mulaw`, 8 kHz, mono).

Implemented with lookup tables instead of `audioop`, which is deprecated in Python 3.12 and gone in 3.13.
The tables follow the reference algorithm (Sun Microsystems g711.c): bias 0x84, clip at 32635, bits inverted.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from callie.audio.pcm import Audio

_BIAS = 0x84
_CLIP = 32635


def _build_encode_table() -> npt.NDArray[np.uint8]:
    # g711.c linear2ulaw: work on the 14-bit value (arithmetic shift first, then take the magnitude).
    linear = np.arange(-32768, 32768, dtype=np.int32) >> 2
    mask = np.where(linear < 0, 0x7F, 0xFF).astype(np.int32)
    magnitude = np.minimum(np.abs(linear), _CLIP >> 2) + (_BIAS >> 2)
    segment_ends = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32)
    segment = np.searchsorted(segment_ends, magnitude, side="left").astype(np.int32)
    encoded = np.where(
        segment >= 8,
        0x7F ^ mask,
        ((segment << 4) | ((magnitude >> (np.minimum(segment, 7) + 1)) & 0x0F)) ^ mask,
    )
    return (encoded & 0xFF).astype(np.uint8)


def _build_decode_table() -> npt.NDArray[np.int16]:
    codes = ~np.arange(256, dtype=np.int32) & 0xFF
    sign = codes & 0x80
    exponent = (codes >> 4) & 0x07
    mantissa = codes & 0x0F
    magnitude = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
    return np.where(sign != 0, -magnitude, magnitude).astype(np.int16)


_ENCODE = _build_encode_table()  # index: linear sample + 32768
_DECODE = _build_decode_table()  # index: μ-law byte


def encode_int16(samples: npt.NDArray[np.int16]) -> bytes:
    return bytes(_ENCODE[samples.astype(np.int32) + 32768].tobytes())


def decode_to_int16(data: bytes) -> npt.NDArray[np.int16]:
    return _DECODE[np.frombuffer(data, dtype=np.uint8)]


def encode(audio: Audio) -> bytes:
    """Float audio in [-1, 1] -> μ-law bytes (one byte per sample)."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype(np.int16)
    return encode_int16(pcm)


def decode(data: bytes) -> Audio:
    """μ-law bytes -> float audio in [-1, 1]."""
    return (decode_to_int16(data).astype(np.float32) / 32768.0).astype(np.float32)
