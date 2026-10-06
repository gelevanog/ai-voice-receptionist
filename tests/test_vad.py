"""VAD and endpointing on synthetic signals (energy VAD; Silero when the model is downloaded)."""

from __future__ import annotations

import numpy as np
import pytest

from callie.config import Settings
from callie.vad import FRAME, FRAME_S, Endpointer, EnergyVAD, SileroVAD
from tests.helpers import quiet, speech_like


def run(endpointer: Endpointer, audio: np.ndarray, chunk: int = 320) -> list:  # type: ignore[type-arg]
    events = []
    for start in range(0, len(audio), chunk):
        events.extend(endpointer.feed(audio[start : start + chunk]))
    return events


def test_detects_one_utterance_with_timing() -> None:
    endpointer = Endpointer(EnergyVAD(), min_speech_ms=96, end_silence_ms=500)
    audio = np.concatenate([quiet(1.0), speech_like(1.2), quiet(1.0)])
    events = run(endpointer, audio)
    kinds = [e.kind for e in events]
    assert kinds[0] == "speech_start" and kinds[-1] == "speech_end" and kinds.count("speech_end") == 1
    start = events[0]
    end = events[-1]
    assert start.onset == pytest.approx(1.0, abs=0.1)
    assert end.last_speech_t == pytest.approx(2.2, abs=0.1)
    # End of turn fires after the configured silence, not before.
    assert end.t - end.last_speech_t == pytest.approx(0.5, abs=FRAME_S * 1.5)
    # The utterance audio includes the onset (pre-roll) and a short tail.
    assert 1.2 <= len(end.audio) / 16000 <= 1.2 + 0.32 + 0.25


def test_short_pause_does_not_end_the_turn() -> None:
    endpointer = Endpointer(EnergyVAD(), end_silence_ms=550)
    audio = np.concatenate([quiet(0.5), speech_like(0.8), quiet(0.3), speech_like(0.8, seed=2), quiet(0.8)])
    ends = [e for e in run(endpointer, audio) if e.kind == "speech_end"]
    assert len(ends) == 1 and ends[0].speech_s > 1.4


def test_long_pause_splits_turns() -> None:
    endpointer = Endpointer(EnergyVAD(), end_silence_ms=400)
    audio = np.concatenate([quiet(0.5), speech_like(0.6), quiet(0.8), speech_like(0.6, seed=4), quiet(0.8)])
    assert sum(e.kind == "speech_end" for e in run(endpointer, audio)) == 2


def test_clicks_and_silence_are_not_speech() -> None:
    endpointer = Endpointer(EnergyVAD(), min_speech_ms=120)
    click = np.zeros(16000, dtype=np.float32)
    click[8000:8040] = 0.8  # 2.5 ms click
    assert run(endpointer, np.concatenate([quiet(1.0), click, quiet(1.0)])) == []


def test_ongoing_events_report_speech_duration() -> None:
    endpointer = Endpointer(EnergyVAD(), min_speech_ms=96)
    events = run(endpointer, np.concatenate([quiet(0.5), speech_like(1.0)]))
    ongoing = [e.speech_s for e in events if e.kind == "speech_ongoing"]
    assert ongoing == sorted(ongoing) and ongoing[-1] == pytest.approx(1.0, abs=0.1)


def test_chunk_size_does_not_matter() -> None:
    audio = np.concatenate([quiet(0.5), speech_like(0.8), quiet(0.8)])
    a = [(e.kind, round(e.t, 3)) for e in run(Endpointer(EnergyVAD()), audio, chunk=160)]
    b = [(e.kind, round(e.t, 3)) for e in run(Endpointer(EnergyVAD()), audio, chunk=1000)]
    assert a == b


def test_max_utterance_forces_an_end() -> None:
    endpointer = Endpointer(EnergyVAD(), max_utterance_s=2.0)
    ends = [e for e in run(endpointer, np.concatenate([quiet(0.3), speech_like(5.0)])) if e.kind == "speech_end"]
    assert ends and ends[0].speech_s <= 2.1


def test_energy_vad_adapts_to_noise_floor() -> None:
    vad = EnergyVAD()
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(16000 * 3) * 0.003).astype(np.float32)
    probs = [vad.prob(noise[i : i + FRAME]) for i in range(0, len(noise) - FRAME, FRAME)]
    assert np.mean(probs[-20:]) < 0.2
    assert vad.prob(speech_like(0.032)) > 0.8


@pytest.mark.models
def test_silero_detects_synthetic_speech_and_rejects_noise() -> None:
    path = Settings().models_dir / "silero_vad.onnx"
    if not path.exists():
        pytest.skip("Silero VAD model not downloaded (callie download-models)")
    vad = SileroVAD(path)
    noise = (np.random.default_rng(0).standard_normal(16000) * 0.01).astype(np.float32)
    probs = [vad.prob(noise[i : i + FRAME]) for i in range(0, len(noise) - FRAME, FRAME)]
    assert max(probs) < 0.5
