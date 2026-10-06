"""Building the VAD / STT / TTS stack from settings, and downloading the model files.

Weights are never committed or baked into the image by default: `callie download-models` fetches them into
`CALLIE_MODELS_DIR` (and Whisper into the Hugging Face cache). Each entry lists its source and license.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from callie.config import Settings
from callie.stt import STT, FakeSTT, WhisperSTT
from callie.tts import TTS, FakeTTS, KokoroTTS, PiperTTS
from callie.vad import Endpointer, EnergyVAD, SileroVAD, VADModel


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    license: str
    used_for: str


MODEL_FILES = [
    ModelFile(
        "silero_vad.onnx",
        "https://github.com/snakers4/silero-vad/raw/v6.2.3/src/silero_vad/data/silero_vad.onnx",
        "MIT (Silero Team)",
        "voice activity detection",
    ),
    ModelFile(
        "kokoro-v1.0.onnx",
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        "Apache-2.0 (Kokoro-82M weights, hexgrad)",
        "agent voice (TTS)",
    ),
    ModelFile(
        "voices-v1.0.bin",
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        "Apache-2.0 (Kokoro-82M voices)",
        "agent voice (TTS)",
    ),
]
PIPER_FILES = [
    ModelFile(
        "en_US-libritts_r-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx",
        "voice trained on LibriTTS-R (CC BY 4.0), fine-tuned from Piper's lessac voice",
        "simulated callers in the evaluation",
    ),
    ModelFile(
        "en_US-libritts_r-medium.onnx.json",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json",
        "as above",
        "simulated callers in the evaluation",
    ),
]


def download_models(
    settings: Settings, *, piper: bool = False, whisper: list[str] | None = None, log: Callable[[str], None] = print
) -> None:
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    for item in MODEL_FILES + (PIPER_FILES if piper else []):
        target = settings.models_dir / item.name
        if target.exists() and target.stat().st_size > 0:
            log(f"ok        {item.name} ({item.license})")
            continue
        log(f"download  {item.name} <- {item.url}")
        partial = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(item.url, partial)
        partial.rename(target)
    for name in whisper if whisper is not None else [settings.stt_model]:
        from faster_whisper import download_model

        log(f"whisper   {name} -> {download_model(name)}")


@dataclass
class SpeechStack:
    stt: STT
    tts: TTS
    vad_factory: Callable[[], VADModel]
    settings: Settings

    def endpointer(self) -> Endpointer:
        s = self.settings
        return Endpointer(
            self.vad_factory(),
            threshold=s.vad_threshold,
            min_speech_ms=s.min_speech_ms,
            end_silence_ms=s.endpoint_silence_ms,
        )

    def describe(self) -> dict[str, str]:
        return {"stt": self.stt.name, "tts": self.tts.name, "vad": self.settings.vad_provider}


def build_speech(settings: Settings, *, fake_lines: list[str] | None = None) -> SpeechStack:
    models = settings.models_dir
    stt: STT
    tts: TTS
    if settings.stt_provider == "whisper":
        stt = WhisperSTT(settings.stt_model, threads=settings.stt_threads)
    else:
        stt = FakeSTT(fake_lines or [])
    if settings.tts_provider == "kokoro":
        tts = KokoroTTS(
            models / "kokoro-v1.0.onnx",
            models / "voices-v1.0.bin",
            voice=settings.tts_voice,
            speed=settings.tts_speed,
            threads=settings.tts_threads,
        )
    elif settings.tts_provider == "piper":
        tts = PiperTTS(models / "en_US-libritts_r-medium.onnx", speaker_id=None)
    else:
        tts = FakeTTS()
    vad_path = models / "silero_vad.onnx"

    def vad_factory() -> VADModel:
        if settings.vad_provider == "silero":
            return SileroVAD(vad_path)
        return EnergyVAD()

    return SpeechStack(stt=stt, tts=tts, vad_factory=vad_factory, settings=settings)


def models_present(settings: Settings) -> dict[str, bool]:
    return {item.name: (settings.models_dir / item.name).exists() for item in MODEL_FILES + PIPER_FILES}


def model_path(settings: Settings, name: str) -> Path:
    return settings.models_dir / name
