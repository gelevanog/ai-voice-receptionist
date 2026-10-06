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


def export_call(recording: Path, out_dir: Path, name: str, title: str, model: str = "small.en") -> dict[str, Any]:
    from faster_whisper import WhisperModel

    out_dir.mkdir(parents=True, exist_ok=True)
    caller, agent, rate = _channels(recording)
    whisper = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=4)
    lines: list[dict[str, Any]] = []
    for speaker, audio in (("Caller", caller), ("Callie", agent)):
        segments, _ = whisper.transcribe(
            audio, language="en", beam_size=5, vad_filter=True, condition_on_previous_text=False
        )
        lines.extend(
            {"start": round(s.start, 2), "end": round(s.end, 2), "speaker": speaker, "text": s.text.strip()}
            for s in segments
            if s.text.strip()
        )
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
        "Simulated caller (LLM-written lines, Piper voice); Callie as in the evaluation run. "
        f"Transcribed from the recording with faster-whisper {model}, so a few words may be misheard.",
        "",
    ]
    markdown += [f"- `{_clock(line['start'])}` **{line['speaker']}:** {line['text']}" for line in lines]
    (out_dir / f"{name}.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (out_dir / f"{name}.json").write_text(json.dumps({"title": title, "lines": lines}, indent=1), encoding="utf-8")
    return {"mp3": str(mp3), "lines": len(lines), "seconds": round(len(caller) / rate, 1)}


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"
