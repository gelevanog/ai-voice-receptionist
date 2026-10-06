"""Export a recorded call for listening: an MP3 of both sides and a timestamped transcript.

The transcript is made from the recording itself: each channel of the stereo WAV (left: caller, right: Callie)
is transcribed with faster-whisper and the segments are merged by start time, so the timestamps are where the
words are actually audible. Names in demo calls are fictional (simulated callers).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np

from callie.audio.pcm import int16_bytes_to_float


def _channels(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        frames = int16_bytes_to_float(handle.readframes(handle.getnframes()))
    stereo = frames.reshape(-1, 2)
    return stereo[:, 0].copy(), stereo[:, 1].copy(), rate


DEFAULT_NOTE = "Simulated caller (LLM-written lines, Piper voice); Callie as in the evaluation run."


def export_call(
    recording: Path, out_dir: Path, name: str, title: str, model: str = "small.en", note: str = DEFAULT_NOTE
) -> dict[str, Any]:
    from faster_whisper import WhisperModel

    out_dir.mkdir(parents=True, exist_ok=True)
    caller, agent, rate = _channels(recording)
    whisper = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=4)
    lines: list[dict[str, Any]] = []
    for speaker, audio in (("Caller", caller), ("Callie", agent)):
        for begin, finish in _segments(audio, rate):
            clip = audio[int(begin * rate) : int(finish * rate)]
            segments, _ = whisper.transcribe(clip, language="en", beam_size=5, condition_on_previous_text=False)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            if text:
                lines.append({"start": round(begin, 2), "end": round(finish, 2), "speaker": speaker, "text": text})
    lines.sort(key=lambda line: line["start"])
    mp3 = out_dir / f"{name}.mp3"
    if shutil.which("ffmpeg"):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(recording),
                "-ac",
                "2",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "64k",
                str(mp3),
            ],
            check=True,
        )
    markdown = [
        f"# {title}",
        "",
        f"Audio: [{mp3.name}]({mp3.name}) (stereo: caller left, Callie right). "
        f"{note} "
        f"Transcribed from the recording with faster-whisper {model}, so a few words may be misheard.",
        "",
    ]
    markdown += [f"- `{_clock(line['start'])}` **{line['speaker']}:** {line['text']}" for line in lines]
    (out_dir / f"{name}.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (out_dir / f"{name}.json").write_text(json.dumps({"title": title, "lines": lines}, indent=1), encoding="utf-8")
    return {"mp3": str(mp3), "lines": len(lines), "seconds": round(len(caller) / rate, 1)}


def _segments(audio: np.ndarray, rate: int, min_gap: float = 0.7) -> list[tuple[float, float]]:
    """Stretches of sound separated by at least `min_gap` seconds of silence (100 ms energy windows)."""
    step = rate // 10
    energy = [float(np.sqrt(np.mean(np.square(audio[i : i + step])))) for i in range(0, len(audio), step)]
    spans: list[tuple[float, float]] = []
    start: float | None = None
    quiet = 0
    for index, value in enumerate(energy):
        if value > 0.01:
            if start is None:
                start = index / 10
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet / 10 >= min_gap:
                spans.append((max(0.0, start - 0.1), (index - quiet + 1) / 10 + 0.1))
                start, quiet = None, 0
    if start is not None:
        spans.append((max(0.0, start - 0.1), len(energy) / 10))
    return spans


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"
