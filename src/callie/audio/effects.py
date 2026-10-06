"""Channel simulation for the evaluation: background noise at a target SNR and a phone line (8 kHz μ-law)."""

from __future__ import annotations

import numpy as np
from scipy import signal

from callie.audio import mulaw
from callie.audio.pcm import Audio
from callie.audio.resample import resample


def pink_noise(n: int, rng: np.random.Generator) -> Audio:
    """1/f noise via spectral shaping of white noise, normalized to unit RMS."""
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.arange(len(spectrum), dtype=np.float64)
    freqs[0] = 1.0
    pink = np.fft.irfft(spectrum / np.sqrt(freqs), n)
    pink /= np.sqrt(np.mean(pink**2)) + 1e-12
    return pink.astype(np.float32)


def babble_noise(n: int, rate: int, rng: np.random.Generator) -> Audio:
    """Speech-shaped, syllable-rate modulated noise: a crude stand-in for a busy room or a street."""
    base = pink_noise(n, rng)
    sos = signal.butter(4, [200, 3000], btype="bandpass", fs=rate, output="sos")
    shaped = signal.sosfilt(sos, base)
    t = np.arange(n) / rate
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3.7 * t + rng.uniform(0, 6.28)) * np.sin(2 * np.pi * 0.9 * t)
    out = shaped * envelope
    out /= np.sqrt(np.mean(out**2)) + 1e-12
    return out.astype(np.float32)


def add_noise(audio: Audio, snr_db: float, rate: int, rng: np.random.Generator, kind: str = "babble") -> Audio:
    """Mix noise at `snr_db` relative to the RMS of the speech (computed over the non-silent part)."""
    if len(audio) == 0:
        return audio
    active = audio[np.abs(audio) > 1e-3]
    speech_rms = float(np.sqrt(np.mean(np.square(active if len(active) else audio))))
    noise = babble_noise(len(audio), rate, rng) if kind == "babble" else pink_noise(len(audio), rng)
    noise_rms = speech_rms / (10 ** (snr_db / 20))
    mixed = audio + noise * noise_rms
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


def phone_channel(audio: Audio, rate: int) -> Audio:
    """Band-limit to 300-3400 Hz, resample to 8 kHz, μ-law encode/decode and return at the original rate.

    This is the same codec path the Twilio transport uses, so the "phone" evaluation condition hears what
    a caller on a real phone line would sound like to the pipeline (minus packet loss and jitter).
    """
    sos = signal.butter(6, [300, 3400], btype="bandpass", fs=rate, output="sos")
    banded = signal.sosfilt(sos, audio).astype(np.float32)
    narrow = resample(banded, rate, 8000)
    coded = mulaw.decode(mulaw.encode(narrow))
    return resample(coded, 8000, rate)
