"""μ-law codec, resampling (whole-buffer and streaming), PCM helpers and channel effects."""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
import pytest

from callie.audio import mulaw
from callie.audio.effects import add_noise, phone_channel
from callie.audio.pcm import float_to_int16_bytes, int16_bytes_to_float, read_wav, rms_dbfs, write_wav
from callie.audio.resample import StreamResampler, resample


def tone(freq: float, seconds: float, rate: int, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(round(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_frequency(audio: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    return float(np.fft.rfftfreq(len(audio), 1 / rate)[np.argmax(spectrum)])


class TestMulaw:
    def test_known_codes(self) -> None:
        # G.711 μ-law: silence encodes to 0xFF, the extremes to 0x00 / 0x80.
        assert mulaw.encode_int16(np.array([0], dtype=np.int16)) == b"\xff"
        assert mulaw.encode_int16(np.array([32767], dtype=np.int16)) == b"\x80"
        assert mulaw.encode_int16(np.array([-32768], dtype=np.int16)) == b"\x00"
        assert mulaw.decode_to_int16(b"\xff")[0] == 0
        assert mulaw.decode_to_int16(b"\x80")[0] == 32124
        assert mulaw.decode_to_int16(b"\x00")[0] == -32124

    def test_matches_reference_codec_on_every_sample(self) -> None:
        audioop = pytest.importorskip("audioop")  # stdlib up to Python 3.12: an independent reference
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            every = np.arange(-32768, 32768, dtype=np.int16)
            assert mulaw.encode_int16(every) == audioop.lin2ulaw(every.tobytes(), 2)
            codes = bytes(range(256))
            reference = np.frombuffer(audioop.ulaw2lin(codes, 2), dtype=np.int16)
            assert np.array_equal(mulaw.decode_to_int16(codes), reference)

    def test_round_trip_error_is_small_relative_to_signal(self) -> None:
        audio = tone(440, 0.5, 8000)
        decoded = mulaw.decode(mulaw.encode(audio))
        snr = 10 * np.log10(np.sum(audio**2) / np.sum((audio - decoded) ** 2))
        assert snr > 30  # μ-law gives ~38 dB SQNR for a loud tone
        assert len(mulaw.encode(audio)) == len(audio)  # one byte per sample

    def test_clipping(self) -> None:
        loud = np.array([2.0, -2.0], dtype=np.float32)
        assert mulaw.encode(loud) == mulaw.encode(np.array([1.0, -1.0], dtype=np.float32))


class TestResample:
    @pytest.mark.parametrize(
        ("src", "dst"), [(8000, 16000), (16000, 8000), (24000, 16000), (24000, 8000), (22050, 16000)]
    )
    def test_whole_buffer_keeps_frequency_and_length(self, src: int, dst: int) -> None:
        audio = tone(700, 1.0, src)
        out = resample(audio, src, dst)
        assert abs(len(out) - dst) <= 1
        assert dominant_frequency(out, dst) == pytest.approx(700, abs=3)

    @pytest.mark.parametrize(
        ("src", "dst", "chunk"), [(8000, 16000, 160), (24000, 16000, 480), (24000, 8000, 960), (16000, 8000, 320)]
    )
    def test_stream_resampler_matches_rate_and_has_no_boundary_clicks(self, src: int, dst: int, chunk: int) -> None:
        audio = tone(440, 1.0, src)
        resampler = StreamResampler(src, dst)
        pieces = [resampler.process(audio[i : i + chunk]) for i in range(0, len(audio), chunk)]
        out = np.concatenate(pieces)
        assert len(out) == len(audio) * dst // src
        steady = out[200:-200]
        assert dominant_frequency(steady, dst) == pytest.approx(440, abs=3)
        # A click at a chunk boundary shows up as a sample-to-sample jump far above a 440 Hz sine's slope.
        max_step = 2 * np.pi * 440 / dst * 0.5 * 1.2
        assert float(np.max(np.abs(np.diff(steady)))) < max_step

    def test_stream_resampler_is_chunk_size_independent(self) -> None:
        audio = tone(300, 0.5, 8000)
        a, b = StreamResampler(8000, 16000), StreamResampler(8000, 16000)
        out_a = np.concatenate([a.process(audio[i : i + 160]) for i in range(0, len(audio), 160)])
        out_b = np.concatenate([b.process(audio[i : i + 37]) for i in range(0, len(audio), 37)])
        assert np.allclose(out_a, out_b, atol=1e-5)

    def test_passthrough(self) -> None:
        audio = tone(300, 0.1, 16000)
        assert np.array_equal(StreamResampler(16000, 16000).process(audio), audio)


class TestPcmAndEffects:
    def test_int16_round_trip(self) -> None:
        audio = tone(440, 0.1, 16000)
        assert np.allclose(int16_bytes_to_float(float_to_int16_bytes(audio)), audio, atol=1e-4)

    def test_wav_in_memory_has_pcm16_size(self) -> None:
        audio = tone(440, 0.2, 16000)
        buffer = io.BytesIO()
        write_wav(buffer, audio, 16000)
        assert buffer.getbuffer().nbytes == 44 + 2 * len(audio)

    def test_read_wav(self, tmp_path: Path) -> None:
        audio = tone(440, 0.2, 16000)
        write_wav(tmp_path / "a.wav", audio, 16000)
        loaded, rate = read_wav(tmp_path / "a.wav")
        assert rate == 16000 and np.allclose(loaded, audio, atol=1e-4)

    def test_noise_hits_target_snr(self) -> None:
        clean = tone(300, 2.0, 16000, amplitude=0.3)
        noisy = add_noise(clean, 10.0, 16000, np.random.default_rng(1))
        noise = noisy - clean
        snr = 10 * np.log10(np.mean(clean**2) / np.mean(noise**2))
        assert snr == pytest.approx(10.0, abs=0.5)

    def test_phone_channel_removes_high_frequencies(self) -> None:
        high = tone(6000, 0.5, 16000)
        speech_band = tone(1000, 0.5, 16000)
        assert rms_dbfs(phone_channel(high, 16000)) < rms_dbfs(high) - 30
        assert rms_dbfs(phone_channel(speech_band, 16000)) > rms_dbfs(speech_band) - 2
