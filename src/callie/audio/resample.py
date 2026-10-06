"""Sample-rate conversion: whole buffers (polyphase) and chunked streams (stateful FIR, no edge clicks)."""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy import signal

from callie.audio.pcm import Audio


def resample(audio: Audio, src_rate: int, dst_rate: int) -> Audio:
    """Resample a whole buffer with a polyphase anti-aliasing filter."""
    if src_rate == dst_rate or len(audio) == 0:
        return audio.astype(np.float32, copy=False)
    divisor = gcd(src_rate, dst_rate)
    out = signal.resample_poly(audio.astype(np.float64), dst_rate // divisor, src_rate // divisor)
    return np.asarray(out, dtype=np.float32)


_ONE = np.array([1.0])


class StreamResampler:
    """Resamples consecutive chunks as one continuous signal.

    Resampling each 20 ms telephony frame on its own restarts the filter every frame and adds a click at every
    boundary. This keeps the FIR state and the decimation phase across calls instead. Meant for small ratios
    (8k <-> 16k, 24k -> 16k / 8k); the cost grows with the upsampling factor.
    """

    def __init__(self, src_rate: int, dst_rate: int, taps_per_phase: int = 24) -> None:
        divisor = gcd(src_rate, dst_rate)
        self.up = dst_rate // divisor
        self.down = src_rate // divisor
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self.passthrough = self.up == self.down
        factor = max(self.up, self.down, 2)
        numtaps = taps_per_phase * factor + 1
        self._taps = (signal.firwin(numtaps, 1.0 / factor, window=("kaiser", 6.0)) * self.up).astype(np.float64)
        self._zi = np.zeros(len(self._taps) - 1, dtype=np.float64)
        self._offset = 0  # samples of the upsampled signal produced so far
        self.delay_samples = (numtaps - 1) // 2 / self.up  # group delay, in input samples

    def process(self, chunk: Audio) -> Audio:
        if self.passthrough or len(chunk) == 0:
            return chunk.astype(np.float32, copy=False)
        upsampled = np.zeros(len(chunk) * self.up, dtype=np.float64)
        upsampled[:: self.up] = chunk
        filtered, self._zi = signal.lfilter(self._taps, _ONE, upsampled, zi=self._zi)
        first = (-self._offset) % self.down
        self._offset += len(filtered)
        return np.asarray(filtered[first :: self.down], dtype=np.float32)

    def reset(self) -> None:
        self._zi[:] = 0.0
        self._offset = 0
